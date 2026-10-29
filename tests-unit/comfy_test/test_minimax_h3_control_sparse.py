import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl, DiTBlock
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlBlockPatch
from comfy_extras import nodes_sparse_attention as sparse


class Model:
    def __init__(self):
        self.diffusion = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(self.diffusion)
        self.diffusion.blocks = [SimpleNamespace(attn=SimpleNamespace(to_gate_compress=None))]
        self.model_options = {"transformer_options": {}}

    def clone(self):
        out = copy.copy(self)
        out.model_options = copy.deepcopy(self.model_options)
        return out

    def get_model_object(self, name):
        return self.diffusion if name == "diffusion_model" else SimpleNamespace(percent_to_sigma=lambda x: 1 - x)

    def add_callback_with_key(self, *args):
        pass

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
