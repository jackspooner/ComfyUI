from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.sd import CLIP
from comfy.model_patcher import ModelPatcher
from comfy.text_encoders.qwen3vl import Qwen3VL, Qwen3VLClipModel
from comfy.text_encoders.vision_features import VisionFeatures


class Visual(torch.nn.Module):
    def __init__(self, random=False):
        super().__init__()
        self.random = random
        self.calls = 0
        self.eval()

    def forward(self, image, grid):
        self.calls += 1
        merged = image.reshape(-1, 3).mean(dim=0, keepdim=True)
        if self.random:
            merged = merged + torch.rand_like(merged)
        return merged, [merged + index for index in range(3)]


def make_clip():
    transformer = Qwen3VL.__new__(Qwen3VL)
    torch.nn.Module.__init__(transformer)
    transformer.visual = Visual()
    transformer.dtype = torch.float32
    transformer.vision_feature_cache = None
    clip_model = Qwen3VLClipModel.__new__(Qwen3VLClipModel)
    torch.nn.Module.__init__(clip_model)
    clip_model.transformer = transformer
    clip_model.options_default = ("last", None, True)
    clip_model.layer, clip_model.layer_idx = "last", None
    clip_model.return_projected_pooled, clip_model.execution_device = True, "cpu"
    text_calls = []
    def encode(tokens):
        text_calls.append(tokens["text"])
        merged, extra = transformer.preprocess_embed({"type": "image", "data": tokens["image"]}, "cpu")
        return merged + tokens["text"], None, extra
    clip = CLIP(no_init=True)
    clip.cond_stage_model = SimpleNamespace(reset_clip_options=clip_model.reset_clip_options,
                                           set_clip_options=clip_model.set_clip_options, encode_token_weights=encode)
    clip.patcher = SimpleNamespace(model=transformer, patches_uuid="one", load_device=torch.device("cpu"), forced_hooks=None,
                                   current_hooks=None, patches={}, object_patches={"manual_cast_dtype": torch.float32}, weight_wrapper_patches={},
                                   injections={}, model_options={"transformer_options": {}}, callbacks={}, wrappers={})
    clip.use_clip_schedule, clip.layer_idx, clip.apply_hooks_to_conds = False, None, None
    clip.load_model = lambda tokens: None
    return clip, transformer, text_calls


@torch.inference_mode()
def test_scoped_positive_negative_keep_text_passes_and_private_deepstack():
    clip, transformer, texts = make_clip()
    image = torch.linspace(0, 1, 32 * 32 * 3).reshape(1, 32, 32, 3)
    baseline = [clip.encode_from_tokens({"text": value, "image": image}, return_dict=True) for value in (1, 2)]
    assert transformer.visual.calls == 2
    with clip.vision_encode_scope():
        positive = clip.encode_from_tokens({"text": 1, "image": image}, return_dict=True)
        negative = clip.encode_from_tokens({"text": 2, "image": image.clone()}, return_dict=True)
        torch.testing.assert_close(positive["cond"], baseline[0]["cond"], rtol=0, atol=0)
        torch.testing.assert_close(negative["cond"], baseline[1]["cond"], rtol=0, atol=0)
        positive["deepstack"][0].add_(20)
        positive["grid"][0] = 9
        third = clip.encode_from_tokens({"text": 2, "image": image}, return_dict=True)
        torch.testing.assert_close(third["deepstack"][0], baseline[1]["deepstack"][0], rtol=0, atol=0)
        assert torch.equal(third["grid"], baseline[1]["grid"])
        features = clip._vision_features
    assert transformer.visual.calls == 3
    assert texts == [1, 2, 1, 2, 2]
    assert features.entries == [] and features.bytes == 0
    assert transformer.vision_feature_cache is None and clip._vision_features is None


@torch.inference_mode()
def test_exact_content_geometry_dtype_and_model_keys():
    features = VisionFeatures(16, 1024**2)
    visual = Visual()
    image = torch.zeros(1, 3)
    for pixels, grid, dtype in [(image, [1, 1, 1], torch.float32), (image.clone(), [1, 1, 1], torch.float32),
                                (image + 1, [1, 1, 1], torch.float32), (image, [1, 1, 2], torch.float32),
                                (image, [1, 1, 1], torch.float16)]:
        features.encode(visual, pixels, grid, "cpu", dtype)
    assert visual.calls == 4
    features.encode(visual, -image, [1, 1, 1], "cpu", torch.float32)
    assert visual.calls == 5  # Signed zero differs at the processed-byte boundary.
    other = Visual()
    features.encode(other, image, [1, 1, 1], "cpu", torch.float32)
    assert other.calls == 1


@pytest.mark.parametrize("field,value", [("forced_hooks", object()), ("current_hooks", object()),
                                        ("patches", {"x": 1}), ("object_patches", {"x": 1}),
                                        ("injections", {"x": 1}), ("wrappers", {"x": {"y": [1]}}),
                                        ("callbacks", {"x": {"y": [1]}}),
                                        ("model_options", {"transformer_options": {"patched": True}})])
@torch.inference_mode()
def test_custom_patchers_do_not_reuse(field, value):
    clip, transformer, _ = make_clip()
    setattr(clip.patcher, field, value)
    with clip.vision_encode_scope():
        for _ in range(2):
            clip.encode_from_tokens({"text": 1, "image": torch.zeros(1, 32, 32, 3)})
    assert transformer.visual.calls == 2


@torch.inference_mode()
def test_patch_generation_schedule_exception_and_nested_scope():
    clip, transformer, _ = make_clip()
    tokens = {"text": 1, "image": torch.zeros(1, 32, 32, 3)}
    with pytest.raises(RuntimeError, match="failure"), clip.vision_encode_scope():
        clip.encode_from_tokens(tokens)
        clip.patcher.patches_uuid = "two"
        clip.encode_from_tokens(tokens)
        with clip.vision_encode_scope():
            clip.encode_from_tokens(tokens)
        clip.encode_from_tokens(tokens)
        clip.use_clip_schedule = True
        clip.encode_from_tokens(tokens)
        clip.encode_from_tokens(tokens)
        assert transformer.visual.calls == 5
        raise RuntimeError("failure")
    assert transformer.vision_feature_cache is None and clip._vision_features is None


@torch.inference_mode()
def test_rng_state_parity_and_hooks_bypass():
    image, grid = torch.zeros(1, 3), [1, 1, 1]
    visual, features = Visual(random=True), VisionFeatures(16, 1024**2)
    torch.manual_seed(42)
    baseline = [visual(image, grid) for _ in range(2)]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    actual = [features.encode(visual, image, grid, "cpu", torch.float32) for _ in range(2)]
    assert torch.equal(expected_rng, torch.get_rng_state()) and not features.entries
    for a, b in zip(actual, baseline):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    visual = Visual()
    visual.register_forward_hook(lambda *args: None)
    features.encode(visual, image, grid, "cpu", torch.float32)
    features.encode(visual, image, grid, "cpu", torch.float32)
    assert visual.calls == 2 and not features.entries


@torch.inference_mode()
def test_entry_and_byte_limits():
    visual, image, grid = Visual(), torch.zeros(1, 3), [1, 1, 1]
    for features in (VisionFeatures(1, 1024**2), VisionFeatures(16, 1)):
        for value in range(3):
            features.encode(visual, image + value, grid, "cpu", torch.float32)
        assert len(features.entries) <= features.max_entries and features.bytes <= features.max_bytes


def test_native_patcher_default_cast_is_admitted_and_changes_invalidate():
    patcher = ModelPatcher(torch.nn.Linear(2, 2), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patcher.set_model_compute_dtype(torch.float32)
    clip = SimpleNamespace(patcher=patcher, use_clip_schedule=False)
    features = VisionFeatures(16, 1024)
    assert features.accepts(clip)
    features.entries.append("fixture")
    patcher.set_model_compute_dtype(torch.float16)
    assert features.accepts(clip) and features.entries == []
