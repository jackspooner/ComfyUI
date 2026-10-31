"""A model's precision opt-out must survive backend and container dispatch."""
import pytest
import torch

from comfy.cli_args import args
args.cpu = True

from comfy.ldm.modules import attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("containers", [False, True])
@pytest.mark.parametrize("skip_reshape", [False, True])
def test_kitchen_precision_opt_out_matches_sdpa(monkeypatch, dtype, containers, skip_reshape):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit precision opt-out reached INT8 attention")

    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention", forbidden)
    monkeypatch.setattr(attention.comfy_kitchen, "prequantize_int8_attention", forbidden)
    generator = torch.Generator().manual_seed(7)
    tensors = [torch.randn(1, 2, 5, 8, generator=generator).to(dtype) for _ in range(3)]
    if not skip_reshape:
        tensors = [t.transpose(1, 2).reshape(1, 5, 16) for t in tensors]
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    options = dict(mask=mask, skip_reshape=skip_reshape,
                   skip_output_reshape=skip_reshape, scale=0.2,
                   low_precision_attention=False)
    expected = attention.attention_pytorch(*tensors, 2, **options)
    inputs = [attention.AttentionTensorContainer(t) for t in tensors] if containers else tensors
    actual = attention.attention_comfy_kitchen_int8(*inputs, 2, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    if containers:
        assert all(c.tensor is None for c in inputs)


@pytest.mark.parametrize("containers", [False, True])
@pytest.mark.parametrize("options", [{}, {"low_precision_attention": True}])
def test_kitchen_default_keeps_int8_dispatch(monkeypatch, containers, options):
    calls = []
    q = torch.ones(1, 2, 5, 8, dtype=torch.bfloat16)

    def int8(*args, **kwargs):
        calls.append("int8")
        return q

    def prequantize(*args, **kwargs):
        calls.append("prequantize")
        return "quantized"

    def from_prequantized(value):
        assert value == "quantized"
        calls.append("from_prequantized")
        return q

    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention", int8)
    monkeypatch.setattr(attention.comfy_kitchen, "prequantize_int8_attention", prequantize)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_from_prequantized", from_prequantized)
    inputs = [attention.AttentionTensorContainer(q) for _ in range(3)] if containers else [q, q, q]
    actual = attention.attention_comfy_kitchen_int8(*inputs, 2, skip_reshape=True,
                                                  skip_output_reshape=True, **options)
    assert actual is q
    assert calls == (["prequantize", "from_prequantized"] if containers else ["int8"])
