import pytest
import torch

from comfy.cli_args import args
args.cpu = True

import comfy.sd as sd
from comfy.component_leases import ComponentLeasePool, active_pool
from comfy.model_patcher import ModelPatcher
from comfy.text_encoders import qwen3vl, krea2, fibo2, flux, boogu, ideogram4, qwen_image21


class Language(torch.nn.Module):
    constructions = 0

    def __init__(self, config, **kwargs):
        super().__init__()
        Language.constructions += 1
        self.config = config
        self.norm = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, tokens, **kwargs):
        return tokens * self.weight


class Visual(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

    def forward(self, image, grid):
        value = image.reshape(-1, 3).mean(dim=0, keepdim=True)
        return value, [value + i for i in range(3)]


class Tokenizer:
    def __init__(self, **kwargs):
        self.options = {}


@pytest.fixture
def source(monkeypatch):
    Language.constructions = 0
    monkeypatch.setattr(qwen3vl, "Llama2_", Language)
    monkeypatch.setattr(qwen3vl, "Qwen3VLVisionModel", Visual)
    for module, name in [(krea2, "Krea2Tokenizer"), (fibo2, "Fibo2Tokenizer"), (flux, "Flux3Tokenizer"),
                         (boogu, "BooguTokenizer"), (ideogram4, "Ideogram4Qwen3VLTokenizer"),
                         (qwen_image21, "QwenImage21Tokenizer")]:
        monkeypatch.setattr(module, name, Tokenizer)
    monkeypatch.setattr(qwen3vl, "tokenizer", lambda **kwargs: Tokenizer)
    wrapper = krea2.Krea2TEModel(dtype=torch.float32)
    result = sd.CLIP(no_init=True)
    result.cond_stage_model = wrapper
    result.patcher = ModelPatcher(wrapper, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    result.tokenizer = Tokenizer()
    result.layer_idx = None
    result.tokenizer_options = {}
    result.use_clip_schedule = False
    result.apply_hooks_to_conds = None
    result.generation_missing_keys = set()
    result.generation_load_scope = False
    return result


def adapter(source, kind):
    result, reason = sd._component_clip_adapter(source, sd.CLIPType.KREA2, kind, None, {})
    assert reason is None
    return result


def test_cross_wrapper_shares_native_owner_without_constructing_backbone(source):
    fibo = adapter(source, sd.CLIPType.FIBO2)
    flux_clip = adapter(source, sd.CLIPType.FLUX3)
    generic = adapter(source, sd.CLIPType.STABLE_DIFFUSION)
    original = source.cond_stage_model.qwen3vl_4b
    for other in [fibo, flux_clip, generic]:
        child = other.cond_stage_model.qwen3vl_4b
        assert child is not original and child.transformer is not original.transformer
        assert child.transformer.model is original.transformer.model
        assert child.transformer.visual is original.transformer.visual
        assert other.patcher.model is source.patcher.model
        assert other.patcher.is_clone(source.patcher)
        assert other.tokenizer is not source.tokenizer
    assert Language.constructions == 1
    assert original.layer == krea2.KREA2_TAP_LAYERS
    assert fibo.cond_stage_model.qwen3vl_4b.layer == fibo2.TAP_LAYERS
    assert flux_clip.cond_stage_model.qwen3vl_4b.layer == [4, 8, 12, 16, 20, 24, 28, 32]
    assert isinstance(flux_clip.cond_stage_model.qwen3vl_4b.transformer, flux.Flux3Qwen3VL)
    assert generic.cond_stage_model.qwen3vl_4b.layer == "hidden"


def test_private_options_restored_after_success_error_and_cancellation(source, monkeypatch):
    fibo = adapter(source, sd.CLIPType.FIBO2)
    child = fibo.cond_stage_model.qwen3vl_4b
    before = (child.layer, child.layer_idx, child.execution_device, child.return_projected_pooled)
    monkeypatch.setattr(fibo, "load_model", lambda *a, **kw: None)
    def encode(tokens):
        assert child.execution_device == torch.device("cpu")
        child.layer = "mutated"
        if tokens:
            raise tokens
        return torch.tensor([1.0]), None
    monkeypatch.setattr(fibo.cond_stage_model, "encode_token_weights", encode)
    assert fibo.encode_from_tokens(None).item() == 1
    for error in [RuntimeError("failed"), KeyboardInterrupt()]:
        with pytest.raises(type(error)):
            fibo.encode_from_tokens(error, return_pooled="unprojected")
        assert (child.layer, child.layer_idx, child.execution_device, child.return_projected_pooled) == before
    assert source.cond_stage_model.qwen3vl_4b.layer == krea2.KREA2_TAP_LAYERS


@pytest.mark.parametrize("missing", [None, {"model.layers.35.weight"}, {"model.lm_head.weight"}])
def test_partial_backbone_never_becomes_generator_or_other_wrapper(source, missing):
    source.generation_missing_keys = missing
    result, reason = sd._component_clip_adapter(source, sd.CLIPType.KREA2, sd.CLIPType.FIBO2, None, {})
    assert result is None and reason == "partial_or_conditioning_only_backbone"
    with pytest.raises(ValueError, match="completely loaded"):
        source.generate({})


def test_truncated_layers_rejected_and_other_architecture_splits(source):
    source.cond_stage_model.qwen3vl_4b.transformer.model.config.num_hidden_layers -= 1
    with pytest.raises(ValueError, match="completely loaded"):
        source.generate({})
    result, reason = sd._component_clip_adapter(source, sd.CLIPType.KREA2, sd.CLIPType.FLUX2, None, {})
    assert result is None and reason == "partial_or_conditioning_only_backbone"


def test_generation_has_private_calls_and_restores_options(source, monkeypatch):
    generic = adapter(source, sd.CLIPType.STABLE_DIFFUSION)
    child = generic.cond_stage_model.qwen3vl_4b
    monkeypatch.setattr(generic, "load_model", lambda *a, **kw: None)
    monkeypatch.setattr(sd.model_management, "get_free_memory", lambda device: 10**20)
    monkeypatch.setattr(generic.cond_stage_model, "generation_memory_estimation_function", lambda *a, **kw: 0, raising=False)
    seen = []
    def generate(tokens, **options):
        seen.append((tokens, options["seed"], options.get("constraint")))
        child.layer_idx = 99
        if options["seed"] == 2:
            raise RuntimeError("cancelled")
        return [options["seed"]]
    monkeypatch.setattr(generic.cond_stage_model, "generate", generate)
    original = child.layer_idx
    assert generic.generate({"private": 1}, seed=1) == [1]
    with pytest.raises(RuntimeError, match="cancelled"):
        generic.generate({"private": 2}, seed=2)
    assert generic.generate({"private": 3}, seed=3) == [3]
    assert child.layer_idx == original and child.execution_device is None
    assert seen == [({"private": n}, n, None) for n in (1, 2, 3)]


def test_pool_scope_counters_identity_bounds_and_close(source, monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(sd, "_load_clip", lambda *args: source)
    pool = ComponentLeasePool(max_entries=1)
    assert active_pool.get() is None
    with pool.activate():
        first = sd.load_clip([path], clip_type=sd.CLIPType.KREA2)
        second = sd.load_clip([path], clip_type=sd.CLIPType.FIBO2)
        third = sd.load_clip([path], clip_type=sd.CLIPType.KREA2)
    assert first.patcher is not source.patcher
    assert second.cond_stage_model is not third.cond_stage_model
    assert pool.snapshot()["constructions"] == 1
    assert pool.snapshot()["backbone_reuses"] == 1
    assert pool.snapshot()["exact_reuses"] == 1
    assert active_pool.get() is None
    with pool.activate():
        sd.load_clip([path], clip_type=sd.CLIPType.KREA2, model_options={"dtype": torch.float16})
    assert pool.snapshot()["constructions"] == 2
    assert pool.snapshot()["evictions"] == 1
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(KeyboardInterrupt), pool.activate():
        sd.load_clip([path], clip_type=sd.CLIPType.KREA2)
        raise KeyboardInterrupt()
    assert active_pool.get() is None
    assert pool.snapshot()["constructions"] == 3
    pool.close()
    assert pool.snapshot()["retained"] == 0
    with pytest.raises(RuntimeError, match="closed"), pool.activate():
        pass


def test_opaque_options_bypass_reuse(source, monkeypatch, tmp_path):
    path = tmp_path / "model"
    path.touch()
    monkeypatch.setattr(sd, "_load_clip", lambda *args: source)
    pool = ComponentLeasePool()
    with pool.activate():
        for _ in range(2):
            sd.load_clip([path], model_options={"custom_operations": object()})
    assert pool.snapshot()["constructions"] == 2
    assert pool.snapshot()["retained"] == 0
    assert pool.snapshot()["splits"] == {"opaque_runtime_options": 2}


def test_mismatched_backbone_constructor_fails_closed(source):
    child = source.cond_stage_model.qwen3vl_4b
    with qwen3vl.share_backbone(child.transformer, child.operations):
        with pytest.raises(ValueError, match="Incompatible"):
            qwen3vl.Qwen3VL({}, torch.float32, "cpu", child.operations)
    assert qwen3vl._shared_backbone.get() is None


def test_8b_adapters_keep_normalization_taps_and_private_image_spans(source):
    source.cond_stage_model = qwen3vl.Qwen3VLTEModel(dtype=torch.float32)
    source.patcher = ModelPatcher(source.cond_stage_model, torch.device("cpu"), torch.device("cpu"))
    outputs = []
    for kind in [sd.CLIPType.BOOGU, sd.CLIPType.IDEOGRAM4, sd.CLIPType.QWEN_IMAGE]:
        result, reason = sd._component_clip_adapter(source, sd.CLIPType.STABLE_DIFFUSION, kind, None, {})
        assert reason is None
        outputs.append(result.cond_stage_model.qwen3vl_8b)
        assert result.patcher.is_clone(source.patcher)
        assert outputs[-1].transformer.model is source.cond_stage_model.qwen3vl_8b.transformer.model
    assert outputs[0].layer_norm_hidden_state is True
    assert outputs[1].layer == ideogram4.IDEOGRAM4_TAP_LAYERS
    assert outputs[2].layer_norm_hidden_state is False
    assert outputs[2].image_spans == []
    assert Language.constructions == 2  # fixture 4B plus this 8B, no adapter construction


@torch.inference_mode()
def test_pool_vision_reuse_across_nodes_and_close_cannot_refill(source, monkeypatch, tmp_path):
    path = tmp_path / "model"
    path.touch()
    monkeypatch.setattr(sd, "_load_clip", lambda *args: source)
    pool = ComponentLeasePool(max_feature_entries=2, max_feature_bytes=128 * 1024)
    clips = []
    with pool.activate():
        for kind in [sd.CLIPType.KREA2, sd.CLIPType.FIBO2, sd.CLIPType.KREA2]:
            clips.append(sd.load_clip([path], clip_type=kind))
    for clip in clips:
        monkeypatch.setattr(clip, "load_model", lambda *args: None)
        transformer = clip.cond_stage_model.qwen3vl_4b.transformer
        def encode(tokens, transformer=transformer):
            merged, extra = transformer.preprocess_embed({"type": "image", "data": tokens}, "cpu")
            return merged, None, extra
        monkeypatch.setattr(clip.cond_stage_model, "encode_token_weights", encode)
    image = torch.zeros(1, 32, 32, 3)
    for clip in clips:
        with clip.vision_encode_scope():
            result = clip.encode_from_tokens(image.clone(), return_dict=True)
            torch.testing.assert_close(result["cond"], torch.full((1, 3), -1.0))
            result["deepstack"][0].add_(123)  # Returned tensors never mutate the cache.
    assert pool.snapshot()["vision_features"]["encodes"] == 1
    assert pool.snapshot()["vision_features"]["hits"] == 2
    assert pool.snapshot()["vision_features"]["retained"] == 1
    with pytest.raises(RuntimeError), clips[0].vision_encode_scope():
        clips[0].encode_from_tokens(image)
        raise RuntimeError("cancelled")
    assert clips[0]._vision_features is None
    pool.close()
    assert pool.snapshot()["vision_features"]["bytes"] == 0
    with clips[0].vision_encode_scope():
        clips[0].encode_from_tokens(image)
    assert pool.snapshot()["vision_features"]["retained"] == 0


def test_exact_non_qwen_family_and_nested_context(source, monkeypatch, tmp_path):
    path = tmp_path / "model"
    path.touch()
    source.cond_stage_model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(sd, "_load_clip", lambda *args: source)
    outer, inner = ComponentLeasePool(), ComponentLeasePool()
    with outer.activate():
        first = sd.load_clip([path], clip_type=sd.CLIPType.STABLE_DIFFUSION)
        with inner.activate():
            assert active_pool.get() is inner
        second = sd.load_clip([path], clip_type=sd.CLIPType.STABLE_DIFFUSION)
    assert first.patcher is not second.patcher
    assert outer.snapshot()["constructions"] == 1 and outer.snapshot()["exact_reuses"] == 1
    assert inner.snapshot()["constructions"] == 0


def test_known_unsupported_wrapper_records_split(source, monkeypatch, tmp_path):
    path = tmp_path / "model"
    path.touch()
    monkeypatch.setattr(sd, "_load_clip", lambda *args: source)
    pool = ComponentLeasePool()
    with pool.activate():
        sd.load_clip([path], clip_type=sd.CLIPType.KREA2)
        sd.load_clip([path], clip_type=sd.CLIPType.FLUX2)
    assert pool.snapshot()["constructions"] == 2
    assert pool.snapshot()["splits"] == {"unaudited_wrapper_or_architecture": 1}


def test_leased_mutation_and_object_replacement_fail_explicitly(source):
    pool = ComponentLeasePool()
    leased = pool._lease(adapter(source, sd.CLIPType.FIBO2))
    with pytest.raises(ValueError, match="replace shared component weights"):
        leased.load_sd({})
    leased.patcher.object_patches["qwen3vl_4b.transformer"] = object()
    with pytest.raises(ValueError, match="object replacements"):
        leased.load_model()
    assert not source.patcher.object_patches


def test_ordinary_weight_patches_remain_private_and_disable_vision_reuse(source):
    pool = ComponentLeasePool()
    leased = pool._lease(adapter(source, sd.CLIPType.FIBO2))
    leased.add_patches({"qwen3vl_4b.transformer.model.weight": (torch.ones(()),)})
    assert leased.patcher.patches
    assert source.patcher.patches == {}
    assert leased.patcher.patches_uuid != source.patcher.patches_uuid
    assert not pool.vision_features.accepts(leased)
