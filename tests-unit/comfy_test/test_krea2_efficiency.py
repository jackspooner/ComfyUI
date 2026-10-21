import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

from comfy.ldm.krea2 import model as krea
from comfy.ldm.modules.attention import attention_pytorch


def tiny_model():
    torch.manual_seed(123)
    model = krea.SingleStreamDiT(features=64, heads=4, kvheads=2, layers=2,
                                 txtdim=32, txtheads=2, txtkvheads=1, txtlayers=3,
                                 channels=4, multiplier=1, operations=nn)
    for param in model.parameters():
        param.detach().uniform_(-0.1, 0.1)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_compact_gqa_matches_expanded_heads(monkeypatch, dtype, masked):
    attn = tiny_model().blocks[0].attn.to(dtype=dtype)
    x = torch.randn(2, 9, 64, dtype=dtype)
    mask = torch.randn(1, 1, 9, 9, dtype=dtype) if masked else None
    shapes = []

    def compact(q, k, v, heads, **kwargs):
        shapes.append((q.peek().shape[1], k.peek().shape[1], v.peek().shape[1]))
        return attention_pytorch(q, k, v, heads, **kwargs)

    monkeypatch.setattr(krea, "optimized_attention_masked", compact)
    actual = attn(x, mask=mask)
    assert shapes == [(4, 2, 2)]

    def expanded(q, k, v, heads, **kwargs):
        q, k, v = q.take(), k.take(), v.take()
        k = k.repeat_interleave(heads // k.shape[1], dim=1)
        v = v.repeat_interleave(heads // v.shape[1], dim=1)
        kwargs["enable_gqa"] = False
        return attention_pytorch(q, k, v, heads, **kwargs)

    monkeypatch.setattr(krea, "optimized_attention_masked", expanded)
    expected = attn(x, mask=mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("references", [0, 1, 2])
def test_output_head_only_sees_target_tokens(temporal, references):
    model = tiny_model()
    x = torch.randn(2, 4, 5, 7)
    if temporal:
        x = x.unsqueeze(2)
    context = torch.randn(2, 5, 96)
    t = torch.tensor([0.7, 0.3])
    refs = [torch.randn(1, 4, 4, 6) for _ in range(references)]
    captured = {}
    model.blocks[-1].register_forward_hook(lambda module, inputs, out: captured.update(combined=out))
    model.last.register_forward_pre_hook(lambda module, inputs: captured.update(target_length=inputs[0].shape[1]))
    actual = model(x, t, context, ref_latents=refs, ref_latents_method="index_timestep_zero")
    assert captured["target_length"] == 12  # ceil(5/2) * ceil(7/2)

    tvec = model.tmlp(krea.timestep_embedding(t, model.tdim).unsqueeze(1))
    full = model.last(captured["combined"], tvec)[:, 5:17]
    expected = full.reshape(2, 3, 4, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(2, 4, 6, 8)[:, :, :5, :7]
    if temporal:
        expected = expected.unsqueeze(2)
    torch.testing.assert_close(actual, expected)
