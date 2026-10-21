import json

import pytest
import torch

from comfy.cli_args import args
args.cpu = True

import comfy.model_detection
import comfy.ops
import comfy.sd
from comfy.ldm.fibo2.model import FeedForward
from comfy.text_encoders import fibo2, qwen3vl
from comfy_extras.nodes_fibo2 import TextEncodeFibo2
import nodes


def test_clip_type_does_not_alias_existing_fork_models():
    expected = {"MOSS_AUDIO": 38, "AUK": 39, "AUDIO_FLAMINGO": 40, "QWEN3_OMNI": 41, "MOSS_IMAGE": 42}
    for name, value in expected.items():
        assert comfy.sd.CLIPType[name].value == value
        assert comfy.sd.CLIPType.FIBO2 is not comfy.sd.CLIPType[name]
    assert "fibo2" in nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]
    assert len(comfy.sd.CLIPType.__members__) == len(list(comfy.sd.CLIPType))


def fibo_state(fused=True, packed=False):
    width = 64 if packed else 128
    state = {
        "perceiver.latent_queries": torch.empty(512, 128, device="meta"),
        "layers.0.attention.qkv.weight": torch.empty(384, width, device="meta"),
        "final_layer.linear.weight": torch.empty(128, width, device="meta"),
        "layers.0.cross_attn_gate": torch.empty(1, device="meta"),
        "noise_refiner.0.attention.qkv.weight": torch.empty(384, width, device="meta"),
        "perceiver.layers.0.cross_attention.to_q.weight": torch.empty(128, width, device="meta"),
    }
    key = "w13" if fused else "w1"
    state[f"layers.0.feed_forward.{key}.weight"] = torch.empty(512 if fused else 256, width, device="meta")
    return state


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("packed", [True, False])
def test_detection_uses_output_dimensions_for_packed_weights(fused, packed):
    config = comfy.model_detection.detect_unet_config(fibo_state(fused, packed), "")
    assert config["image_model"] == "fibo2"
    assert (config["dim"], config["n_heads"], config["in_channels"], config["ffn_dim"]) == (128, 1, 128, 256)
    assert config["injection_layers"] == [0]
    assert config["n_layers"] == config["n_refiner_layers"] == config["perceiver_layers"] == 1
    assert config.get("fused_ffn", True) is fused


@pytest.mark.parametrize("key", list(fibo_state()))
def test_incomplete_signature_does_not_dereference_missing_keys(key):
    state = fibo_state()
    del state[key]
    config = comfy.model_detection.detect_unet_config(state, "")
    if key in {"perceiver.latent_queries", "layers.0.attention.qkv.weight", "final_layer.linear.weight", "layers.0.feed_forward.w13.weight"}:
        assert config is None or config.get("image_model") != "fibo2"


def test_fused_feed_forward_matches_unfused_math():
    fused = FeedForward(8, 16, True, operations=comfy.ops.disable_weight_init)
    unfused = FeedForward(8, 16, False, operations=comfy.ops.disable_weight_init)
    generator = torch.Generator().manual_seed(7)
    w1, w3 = [torch.randn(16, 8, generator=generator) * 0.1 for _ in range(2)]
    w2 = torch.randn(8, 16, generator=generator) * 0.1
    fused.load_state_dict({"w13.weight": torch.cat((w1, w3)), "w2.weight": w2})
    unfused.load_state_dict({"w1.weight": w1, "w3.weight": w3, "w2.weight": w2})
    x = torch.randn(2, 3, 8, generator=generator)
    torch.testing.assert_close(fused(x), unfused(x))


def test_edit_prompt_wraps_all_images_in_numeric_order(monkeypatch):
    tokenizer = fibo2.Fibo2Tokenizer()
    monkeypatch.setattr(qwen3vl.Qwen3VLTokenizer, "tokenize_with_weights", lambda self, text, *a, **kw: (text, kw))
    images = [torch.zeros(1, 64, 64, 3), torch.ones(1, 64, 64, 4)]
    text, options = tokenizer.tokenize_with_weights("Put <image_1> beside <image_2>", images=images)
    caption = json.loads(text)
    assert caption["image_1"] == caption["image_2"] == fibo2.VISION_MARKER
    assert caption["edit_instruction"] == "Put <image_1> beside <image_2>"
    assert len(options["images"]) == 2
    assert all(image.shape[-1] == 3 for image in options["images"])
    assert torch.count_nonzero(options["images"][0]) == 0
    assert torch.all(options["images"][1] == 1)


def test_null_caption_stays_text_only_even_with_reference_images(monkeypatch):
    tokenizer = fibo2.Fibo2Tokenizer()
    monkeypatch.setattr(qwen3vl.Qwen3VLTokenizer, "tokenize_with_weights", lambda self, text, *a, **kw: (text, kw))
    text, options = tokenizer.tokenize_with_weights("  ", images=[torch.ones(1, 64, 64, 3)])
    assert text == fibo2.NULL_CAPTION
    assert "images" not in options


def test_reference_latents_condition_both_sampler_outputs():
    class Clip:
        def tokenize(self, text, **kwargs):
            return {"text": [[(1, 1.0)]]}

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 2, 12), {}]]

    class Vae:
        def encode(self, image):
            return torch.ones(1, 128, image.shape[1] // 16, image.shape[2] // 16)

    result = TextEncodeFibo2.execute(Clip(), "Make it blue", vae=Vae(), images={"image_1": torch.zeros(1, 64, 80, 3)}, width=80, height=64)
    positive, negative, latent = result.result
    refs = positive[0][1]["reference_latents"]
    assert negative[0][1]["reference_latents"][0] is refs[0]
    assert refs[0].shape == latent["samples"].shape == (1, 128, 4, 5)
