import copy
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl, DiTBlock
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlBlockPatch, MiniMaxH3FunControlPatch
from comfy_extras import nodes_sparse_attention as sparse


class Model:
    def __init__(self):
        self.diffusion = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(self.diffusion)
        self.diffusion.blocks = [SimpleNamespace(attn=SimpleNamespace(to_gate_compress=None))]
        self.model_options = {"transformer_options": {}}
        self.callbacks = {}

    def clone(self):
        out = copy.copy(self)
        out.model_options = copy.deepcopy(self.model_options)
        out.callbacks = {key: list(values) for key, values in self.callbacks.items()}
        return out

    def get_model_object(self, name):
        return self.diffusion if name == "diffusion_model" else SimpleNamespace(percent_to_sigma=lambda x: 1 - x)

    def add_callback_with_key(self, kind, key, callback):
        self.callbacks.setdefault((kind, key), []).append(callback)

    def remove_callbacks_with_key(self, kind, key):
        self.callbacks.pop((kind, key), None)

    def set_model_patch_replace(self, patch, kind, block, index):
        self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault(kind, {})[(block, index)] = patch


@pytest.mark.parametrize("control_first", [False, True])
@pytest.mark.parametrize("eligible", [False, True])
def test_sparse_and_control_compose_in_both_orders(monkeypatch, control_first, eligible):
    monkeypatch.setattr(sparse, "h3_eligible", lambda *args: eligible)
    control = Mock()
    control.after_block.side_effect = lambda index, args, out: {"img": out["img"] + 7}
    model = Model()
    if control_first:
        model.set_model_patch_replace(MiniMaxH3FunControlBlockPatch(control, 0, None), "dit", "double_block", 0)
    model = sparse.apply_block_sparse_attention(model, tau=1.3, topk_ratio=0, vsa=False,
        start_percent=0, end_percent=1, min_tokens=1, dense_blocks=set(),
        sink_conditioning="off", extra_tokens=0, verbose=False)
    patch = model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    if not control_first:
        patch = MiniMaxH3FunControlBlockPatch(control, 0, patch)
    original = Mock(side_effect=lambda args: {"img": args["img"] * 2})
    args = {"img": torch.tensor([3.0]), "rope_freqs": None, "transformer_options": {}}
    result = patch(args, {"original_block": original})
    torch.testing.assert_close(result["img"], torch.tensor([13.0]))
    assert ("attention" in original.call_args.args[0]) is eligible
    assert "attention" not in args


def test_sparse_wrapper_preserves_control_lifecycle():
    previous = Mock()
    previous.to.return_value = previous
    previous.models.return_value = ["control weights"]
    patch = sparse.H3SparseBlockPatch(None, 0, None, previous)
    assert patch.to(torch.device("cpu")) is patch
    assert patch.models() == ["control weights"]
    patch.cleanup()
    previous.to.assert_called_once_with(torch.device("cpu"))
    previous.cleanup.assert_called_once_with()


def test_control_stream_does_not_inherit_base_attention_override(monkeypatch):
    forward = Mock(side_effect=lambda block, c, *args, **kwargs: c + 1)
    monkeypatch.setattr(DiTBlock, "forward", forward)
    block = SimpleNamespace(after_proj=lambda x: x * 2)
    model = SimpleNamespace(control_blocks=[block])
    override = Mock(side_effect=AssertionError("base sparse override reached control stream"))
    options = {"optimized_attention_override": override, "sigmas": torch.tensor([0.5]), "other_patch": "preserved"}
    c, skip = MiniMaxH3FunControl.step(model, 0, torch.tensor([2.0]), None, None, None, options)
    torch.testing.assert_close(c, torch.tensor([3.0]))
    torch.testing.assert_close(skip, torch.tensor([6.0]))
    received = forward.call_args.kwargs["transformer_options"]
    assert "optimized_attention_override" not in received
    assert received["other_patch"] == "preserved"
    assert options["optimized_attention_override"] is override


def test_repeated_sparse_prepare_is_idempotent_and_releases_replaced_overrides():
    model = Model()
    dense = Mock(return_value="dense")
    model.model_options["transformer_options"]["optimized_attention_override"] = dense
    for _ in range(2):
        model = sparse.apply_block_sparse_attention(model, tau=1.3, topk_ratio=0, vsa=False,
            start_percent=0, end_percent=1, min_tokens=1, dense_blocks=set(),
            sink_conditioning="off", extra_tokens=0, verbose=False)
    prepare = model.callbacks[(sparse.comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "block_sparse_attention")]
    cleanup = model.callbacks[(sparse.comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "block_sparse_attention")]
    assert len(prepare) == 1
    assert len(cleanup) == 2
    options = model.model_options["transformer_options"]
    installed = options["optimized_attention_override"]
    for _ in range(10):
        prepare[0](model, None, model.model_options)
        assert options["optimized_attention_override"] is installed
        for callback in cleanup:
            callback(model)
    old = weakref.ref(installed)
    del installed
    # A later dense selector replaces the old chain; preparation wraps it once.
    options["optimized_attention_override"] = Mock()
    prepare[0](model, None, model.model_options)
    gc.collect()
    assert old() is None
    installed = options["optimized_attention_override"]
    prepare[0](model, None, model.model_options)
    assert options["optimized_attention_override"] is installed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_control_residual_updates_only_target_video(dtype):
    layout = SimpleNamespace(segments=[(0, 2, "cond"), (2, 4, "ref_img"),
        (4, 6, "ref_audio"), (6, 8, "audio"), (8, 12, "video")])
    control = Mock()
    control.injection_layers = (0,)
    skip = torch.arange(24, dtype=dtype).reshape(12, 2)
    control.step.return_value = (torch.zeros_like(skip), skip)
    patch = SimpleNamespace(active=True, model_patch=SimpleNamespace(model=control),
        control_latent=torch.zeros(1), pristine_stream=torch.zeros_like(skip), strength=0.5)
    args = {"layout": layout, "t_emb": torch.zeros(1), "mod_segments": [],
            "rope_freqs": None, "transformer_options": {}}
    original = torch.ones_like(skip)
    result = MiniMaxH3FunControlPatch.after_block(patch, 0, args, {"img": original.clone()})
    torch.testing.assert_close(result["img"][:8], original[:8])
    torch.testing.assert_close(result["img"][8:], original[8:] + skip[8:] * 0.5)
    torch.testing.assert_close(skip, torch.arange(24, dtype=dtype).reshape(12, 2))
