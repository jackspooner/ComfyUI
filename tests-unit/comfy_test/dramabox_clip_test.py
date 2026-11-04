from types import SimpleNamespace

import pytest
import torch

import comfy.sd as sd
import comfy.text_encoders.dramabox as dramabox
from comfy.text_encoders.lt import LTXAVGemmaTokenizer


def _gemma_state_dict():
    return {
        "model.layers.0.self_attn.k_proj.weight": torch.empty(1),
        "model.layers.47.self_attn.q_norm.weight": torch.empty(1),
        "model.norm.weight": torch.empty(1),
        "spiece_model": torch.tensor([1], dtype=torch.uint8),
    }


def _audio_state_dict():
    return {
        "text_embedding_projection.audio_aggregate_embed.weight": torch.empty(1),
    }


def test_split_state_dicts_accepts_either_file_order():
    assert dramabox.split_state_dicts([_gemma_state_dict(), _audio_state_dict()]) == (0, 1)
    assert dramabox.split_state_dicts([_audio_state_dict(), _gemma_state_dict()]) == (1, 0)


def test_split_state_dicts_rejects_an_incomplete_pair():
    with pytest.raises(ValueError, match="one Gemma 3 12B"):
        dramabox.split_state_dicts([_gemma_state_dict()])


def test_shared_loader_builds_dramabox_clip_target(monkeypatch):
    captured = {}

    class FakeCLIP:
        def __init__(self, target, **kwargs):
            captured["target"] = target

    factory = object()

    def capture_factory(**kwargs):
        captured["factory_options"] = kwargs
        return factory

    monkeypatch.setattr(sd, "CLIP", FakeCLIP)
    monkeypatch.setattr(dramabox, "te", capture_factory)

    audio_path = "/models/dramabox-audio-components.safetensors"
    sd.load_text_encoder_state_dicts(
        [_audio_state_dict(), _gemma_state_dict()],
        clip_type=sd.CLIPType.DRAMABOX,
        model_options={"dramabox_audio_components_path": audio_path},
    )

    assert captured["target"].clip is factory
    assert captured["target"].tokenizer is LTXAVGemmaTokenizer
    assert captured["factory_options"]["audio_components_path"] == audio_path


def test_path_loader_passes_the_matching_audio_components_path(monkeypatch):
    captured = {}
    state_dicts = {
        "/models/gemma.safetensors": _gemma_state_dict(),
        "/models/audio.safetensors": _audio_state_dict(),
    }

    class FakePatcher:
        cached_patcher_init = None

    class FakeCLIP:
        patcher = FakePatcher()

    monkeypatch.setattr(
        sd.comfy.utils,
        "load_torch_file",
        lambda path, **kwargs: (state_dicts[path], {}),
    )
    monkeypatch.setattr(
        sd.comfy.utils,
        "convert_old_quants",
        lambda state_dict, **kwargs: (state_dict, kwargs["metadata"]),
    )

    def capture_loader(clip_data, **kwargs):
        captured.update(kwargs)
        captured["clip"] = FakeCLIP()
        return captured["clip"]

    monkeypatch.setattr(sd, "load_text_encoder_state_dicts", capture_loader)

    sd._load_clip(
        ["/models/audio.safetensors", "/models/gemma.safetensors"],
        clip_type=sd.CLIPType.DRAMABOX,
    )

    assert captured["model_options"]["dramabox_audio_components_path"] == (
        "/models/audio.safetensors"
    )
    clip = captured["clip"]
    assert clip.dramabox_model_path == "/models/gemma.safetensors"
    assert clip.dramabox_model_name == "gemma.safetensors"
    assert clip.dramabox_audio_components_path == "/models/audio.safetensors"


def test_audio_only_embeddings_preserves_audio_and_mask():
    audio = torch.tensor([[[1.0, 2.0]]])
    mask = torch.tensor([[[0.0, 0.0, -2.0]]])

    class AudioConnector:
        def __call__(self, features, additive_attention_mask):
            return features + 1, None

    embeddings_processor = SimpleNamespace(audio_connector=AudioConnector())
    video, encoded_audio, binary_mask = dramabox._audio_only_create_embeddings(
        embeddings_processor,
        torch.zeros_like(audio),
        audio,
        mask,
    )

    assert torch.equal(video, torch.zeros_like(audio))
    assert torch.equal(encoded_audio, audio + 1)
    assert torch.equal(binary_mask, torch.tensor([[1, 1, 0]]))
