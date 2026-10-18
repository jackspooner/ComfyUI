import json
import os
from unittest import mock

import pytest
import torch
from torch.nn import functional as F
from comfy_kitchen.registry import registry

from comfy.cli_args import args
args.cpu = True
from comfy import ops
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.fixture
def cpu_matmul(monkeypatch):
    # CPU oracle for the hardware-only matmul; the CUDA test uses the real backend.
    def linear(a, b, tensor_scale_a, tensor_scale_b, block_scale_a, block_scale_b, bias=None, out_dtype=None):
        left = ops.quant_ops.ck.dequantize_nvfp4(a, tensor_scale_a, block_scale_a, output_type=torch.float32)
        right = ops.quant_ops.ck.dequantize_nvfp4(b, tensor_scale_b, block_scale_b, output_type=torch.float32)
        return F.linear(left, right, bias.float() if bias is not None else None).to(out_dtype)
    monkeypatch.setattr(ops.quant_ops.ck, 'scaled_mm_nvfp4', linear)


def quantized_fixture(device='cpu'):
    generator = torch.Generator(device=device).manual_seed(51)
    original = torch.randn(32, 64, device=device, dtype=torch.bfloat16, generator=generator)
    quantized = QuantizedTensor.from_float(original, 'TensorCoreNVFP4Layout')
    data, scale, blocks = TensorCoreNVFP4Layout.get_plain_tensors(quantized)
    expected = ops.quant_ops.ck.dequantize_nvfp4(data, scale, blocks, output_type=torch.float32)[:32, :64]
    state = {'weight': data, 'weight_scale': blocks, 'weight_scale_2': scale,
             'comfy_quant': torch.tensor(list(json.dumps({'format': 'nvfp4'}).encode()), dtype=torch.uint8)}
    x = torch.randn(6, 64, device=device, dtype=torch.bfloat16, generator=generator) * .1
    x[0, 0] = 15
    return state, expected, x


@pytest.mark.parametrize('rank', [2, 3])
def test_linear_residual_accuracy_bias_shape_and_packed_weights(rank, cpu_matmul):
    with registry.use_backend('eager'):
        state, weight, x = quantized_fixture()
        bias = torch.linspace(.2, .9, 32, dtype=torch.bfloat16)
        state['bias'] = bias
        packed = state['weight'].clone()
        target = F.linear(x.float(), weight, bias.float()).bfloat16()
        errors = []
        for passes in (1, 2, 3):
            module = ops.mixed_precision_ops(nvfp4_input_passes=passes).Linear(64, 32)
            module.load_state_dict(state.copy(), strict=True)
            actual = module(x if rank == 2 else x.view(2, 3, 64))
            assert actual.dtype == torch.bfloat16
            assert actual.shape == ((6, 32) if rank == 2 else (2, 3, 32))
            errors.append((actual.reshape(6, 32).float() - target.float()).square().mean().sqrt())
            assert torch.equal(module.weight._qdata, packed)
        assert errors[2] < errors[0] * .3
        assert errors[1] < errors[0] * .3
        module._full_precision_mm = True
        inputs = x if rank == 2 else x.view(2, 3, 64)
        with mock.patch.object(ops, '_nvfp4_residual_linear', side_effect=AssertionError('bypassed full-precision owner')):
            actual = module(inputs)
        expected = F.linear(inputs, module.weight.dequantize(), bias)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_expert_bank_residual_and_full_precision_owner(cpu_matmul):
    with registry.use_backend('eager'):
        state, weight, x = quantized_fixture()
        bank_state = {key: torch.stack([value, value]) for key, value in state.items() if key != 'comfy_quant'}
        bank_state['comfy_quant'] = torch.tensor(list(json.dumps({'format': 'nvfp4', 'num_experts': 2}).encode()), dtype=torch.uint8)
        bank = ops.mixed_precision_ops(nvfp4_input_passes=3).MoEExperts(2, 64, 32, bias=False)
        bank.load_state_dict(bank_state, strict=True)
        with mock.patch.object(TensorCoreNVFP4Layout, 'supports_fast_matmul', return_value=True):
            actual = bank.expert_linear(x, 1)
        target = F.linear(x.float(), weight).bfloat16()
        torch.testing.assert_close(actual, target, atol=.04, rtol=.04)
        bank._full_precision_mm = True
        with mock.patch.object(ops, '_nvfp4_residual_linear', side_effect=AssertionError('bypassed full-precision owner')):
            actual = bank.expert_linear(x, 0)
        expected = F.linear(x, bank._expert_qt_from(bank.weight, 0).dequantize())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_residual_configuration_rejects_invalid_pass_counts():
    for value in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            ops.mixed_precision_ops(nvfp4_input_passes=value)


@pytest.mark.skipif(os.environ.get('COMFY_TEST_NVFP4_CUDA') != '1', reason='Requires an owned idle CUDA validation window')
def test_cuda_residual_zero_and_bias_are_exact():
    with registry.use_backend('cuda'):
        state, _, _ = quantized_fixture('cuda')
        bias = torch.linspace(.2, .9, 32, device='cuda', dtype=torch.bfloat16)
        state['bias'] = bias
        module = ops.mixed_precision_ops(nvfp4_input_passes=3).Linear(64, 32, device='cuda')
        module.load_state_dict(state, strict=True)
        for rows in (1, 16, 128):
            actual = module(torch.zeros(rows, 64, device='cuda', dtype=torch.bfloat16))
            assert torch.equal(actual, bias.expand(rows, -1))
