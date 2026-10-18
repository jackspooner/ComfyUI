import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

from comfy.ldm.krea2 import model as krea


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frames", [None, 1, 2])
@pytest.mark.parametrize("references", [0, 1, 2])
@pytest.mark.parametrize("ref_method", ["index", "index_timestep_zero"])
def test_target_projection_matches_full_projection(dtype, frames, references, ref_method):
    torch.manual_seed(123)
    model = krea.SingleStreamDiT(features=64, heads=4, kvheads=2, layers=2,
        txtdim=32, txtheads=2, txtkvheads=1, txtlayers=3,
        channels=4, multiplier=1, operations=nn).to(dtype=dtype)
    for param in model.parameters():
        param.detach().uniform_(-0.1, 0.1)

    batch = 2
    x = torch.randn(batch, 4, 5, 7, dtype=dtype)
    if frames is not None:
        x = x.unsqueeze(2).repeat(1, 1, frames, 1, 1)
        batch *= frames
    context = torch.randn(batch, 5, 96, dtype=dtype)
    t = torch.linspace(0.3, 0.7, batch, dtype=dtype)
    refs = [torch.randn(1, 4, 3 + i, 5 + i, dtype=dtype) for i in range(references)]
    captured = {}
    patched_lengths = []

    def record_attention(q, k, v, **kwargs):
        patched_lengths.append(q.shape[2])
        return {}

    model.blocks[-1].register_forward_hook(lambda module, inputs, out: captured.update(combined=out))
    model.last.register_forward_pre_hook(lambda module, inputs: captured.update(target_length=inputs[0].shape[1]))
    options = {"patches": {"attn1_patch": [record_attention]}}
    actual = model(x, t, context, ref_latents=refs, ref_latents_method=ref_method,
                   transformer_options=options)
    assert actual.shape == x.shape
    assert captured["target_length"] == 12  # ceil(5/2) * ceil(7/2)
    assert patched_lengths == [captured["combined"].shape[1]] * 2
    assert "block_index" not in options

    tvec = model.tmlp(krea.timestep_embedding(t, model.tdim).unsqueeze(1).to(dtype))
    full = model.last(captured["combined"], tvec)[:, 5:17]
    expected = full.reshape(batch, 3, 4, 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    expected = expected.reshape(batch, 4, 6, 8)[:, :, :5, :7]
    if frames is not None:
        expected = expected.reshape(2, frames, 4, 5, 7).movedim(1, 2)
    torch.testing.assert_close(actual, expected)
