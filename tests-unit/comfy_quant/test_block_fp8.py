import pytest
import torch
from torch.nn import functional as F

from comfy.quant_ops import BlockWiseFP8E4M3Layout, QuantizedTensor


@pytest.mark.parametrize('shape', [(256, 384), (131, 257), (2, 256, 384)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_block_fp8_decode_preserves_independent_block_scales(shape, dtype):
    generator = torch.Generator().manual_seed(419)
    data = torch.randint(-20, 21, shape, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand(*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128, generator=generator).bfloat16()
    expected_scale = scales.float().repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :shape[-2], :shape[-1]]
    expected = (data.float() * expected_scale).to(dtype)
    params = BlockWiseFP8E4M3Layout.Params(scale=scales, orig_dtype=dtype, orig_shape=shape)
    weight = QuantizedTensor(data, 'BlockWiseFP8E4M3Layout', params)
    torch.testing.assert_close(weight.dequantize(), expected, atol=0, rtol=0)
    if len(shape) == 2:
        inputs = torch.randn(2, shape[-1], generator=generator).to(dtype).requires_grad_()
        actual = F.linear(inputs, weight)
        torch.testing.assert_close(actual, F.linear(inputs, expected), atol=0, rtol=0)
        actual.sum().backward()
        torch.testing.assert_close(inputs.grad, expected.sum(0).expand_as(inputs), atol=0, rtol=0)


def test_block_fp8_requantization_handles_zero_and_unaligned_blocks():
    weights = torch.zeros(131, 257, dtype=torch.bfloat16)
    weights[:128, :128] = 1
    data, params = BlockWiseFP8E4M3Layout.quantize(weights)
    assert data.shape == weights.shape and params.scale.shape == (2, 3)
    assert torch.isfinite(params.scale).all() and (params.scale > 0).all()
    actual = BlockWiseFP8E4M3Layout.dequantize(data, params)
    torch.testing.assert_close(actual, weights, atol=0, rtol=0)
