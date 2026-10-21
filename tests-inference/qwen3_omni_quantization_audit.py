"""Offline package-layout and residual precision audit; CUDA needs an owned GPU window."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import struct
import sys
import urllib.request
from unittest import mock

import torch
from torch.nn import functional as F
from safetensors import safe_open
from comfy_kitchen.registry import registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.cpu = True
from comfy import ops, sd1_clip
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout
from comfy.text_encoders import qwen3_omni
from scripts.benchmark_qwen3_omni_quant import warm_timing


def package_layout(path):
    with path.open('rb') as stream:
        encoded = stream.read(struct.unpack('<Q', stream.read(8))[0])
        header = json.loads(encoded)
    dtypes = {'U8': torch.uint8, 'BF16': torch.bfloat16, 'F16': torch.float16,
              'F32': torch.float32, 'F8_E4M3': torch.float8_e4m3fn}
    state = {key: torch.empty(value['shape'], dtype=dtypes[value['dtype']], device='meta')
             for key, value in header.items() if key != '__metadata__'}
    with safe_open(path, framework='pt', device='cpu') as tensors:
        for key in state:
            if key.endswith('.comfy_quant') or key in qwen3_omni.ASSET_KEYS:
                state[key] = tensors.get_tensor(key)
    config = json.loads(qwen3_omni.asset_bytes(state['qwen3_omni_config_json']))
    config['processor_config'] = json.loads(qwen3_omni.asset_bytes(state['qwen3_omni_processor_json']))
    model = qwen3_omni.te(config, dtype_llama=torch.bfloat16)(device='meta').qwen3_omni
    # Validate actual package headers without allocating/loading the full weights.
    with mock.patch.object(sd1_clip.SDClipModel, 'load_sd', return_value='validated'):
        assert model.load_sd(state) == 'validated'
    return {'header_layout_valid': True, 'quantized_modules': sum(key.endswith('.comfy_quant') for key in state),
            'header_sha256': hashlib.sha256(encoded).hexdigest(), 'artifact_bytes': path.stat().st_size,
            'payload_integrity_checked': False, 'model_executed': False}


def difference(actual, expected):
    delta = actual.float() - expected.float()
    return {'finite': bool(actual.isfinite().all()), 'max_absolute_error': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-30)).item()}


def check_gpu_window(options):
    if options.device != 'cuda':
        return
    if not (options.runtime_host_url and options.hold_id and options.owner_id and os.environ.get('RUNTIME_HOST_API_KEY')):
        raise RuntimeError('CUDA audit requires a live RuntimeHost hold, its URL/IDs and RUNTIME_HOST_API_KEY.')
    with urllib.request.urlopen(options.comfy_url.rstrip('/') + '/queue', timeout=10) as response:
        queue = json.load(response)
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('ComfyUI work is active or pending; CUDA audit stopped.')
    request = urllib.request.Request(options.runtime_host_url.rstrip('/') + '/v1/runtime/gpu-holds',
                                     headers={'Authorization': 'Bearer ' + os.environ['RUNTIME_HOST_API_KEY']})
    with urllib.request.urlopen(request, timeout=10) as response:
        status = json.load(response)
    hold = status['hold']
    if (status['coordinationVersion'] != 1 or status['inFlight'] or hold is None or hold['state'] != 'held'
            or hold['holdId'] != options.hold_id or hold['ownerId'] != options.owner_id
            or status['worker']['state'] != 'stopped' or status['worker']['pid'] is not None):
        raise RuntimeError('The requested RuntimeHost GPU hold is not exclusively held with its worker stopped.')


def numerical_audit(options):
    device = options.device
    results = []
    shapes = [('text_attention', 2048, 4096), ('expert_gate_up', 2048, 1536), ('expert_down', 768, 2048),
              ('audio_fc1', 1280, 5120), ('vision_fc2_padded', 4320, 1152), ('vision_merger', 4608, 4608)]
    for name, columns, rows in shapes:
        generator = torch.Generator(device=device).manual_seed(71)
        original = torch.randn(rows, columns, dtype=torch.bfloat16, device=device, generator=generator) * .02
        weight = QuantizedTensor.from_float(original, 'TensorCoreNVFP4Layout')
        data, scale, blocks = TensorCoreNVFP4Layout.get_plain_tensors(weight)
        decoded = ops.quant_ops.ck.dequantize_nvfp4(data, scale, blocks, output_type=torch.float32)[:rows, :columns]

        def cpu_oracle(a, b, sa, sb, ba, bb, out_dtype):
            left = ops.quant_ops.ck.dequantize_nvfp4(a, sa, ba, output_type=torch.float32)
            return F.linear(left, decoded).to(out_dtype)

        for tokens in (1, 16, 128):
            check_gpu_window(options)
            inputs = torch.randn(tokens, columns, dtype=torch.bfloat16, device=device, generator=generator)
            inputs[0, 0] = 20
            if name == 'vision_fc2_padded':
                inputs[:, 4304:] = 0
            expected = F.linear(inputs.float(), decoded).bfloat16()
            baseline = F.linear(inputs, decoded.bfloat16())
            record = {'shape': name, 'tokens': tokens, 'columns': columns, 'rows': rows,
                      'bf16_decoded_matmul': difference(baseline, expected), 'residual': []}
            if device == 'cuda':
                bf16_weight = decoded.bfloat16()
                record['resident_bf16_matmul_timing'] = warm_timing(lambda: F.linear(inputs, bf16_weight))
                record['decode_and_bf16_matmul_timing'] = warm_timing(lambda: F.linear(inputs, weight.dequantize()))
            for passes in (1, 2, 3):
                with mock.patch.object(ops.quant_ops.ck, 'scaled_mm_nvfp4', cpu_oracle) if device == 'cpu' else registry.use_backend('cuda'):
                    actual = ops._nvfp4_residual_linear(inputs, weight, None, passes)
                    measurement = {'passes': passes, **difference(actual, expected)}
                    if device == 'cuda':
                        measurement['timing'] = warm_timing(lambda: ops._nvfp4_residual_linear(inputs, weight, None, passes))
                record['residual'].append(measurement)
            results.append(record)
    return results


def range_probe(device):
    weight = QuantizedTensor.from_float(torch.ones(32, 64, device=device, dtype=torch.bfloat16), 'TensorCoreNVFP4Layout')
    inputs = torch.full((1, 64), 2048, dtype=torch.bfloat16, device=device)

    def cpu_oracle(a, b, sa, sb, ba, bb, out_dtype):
        left = ops.quant_ops.ck.dequantize_nvfp4(a, sa, ba, output_type=torch.float32)
        right = ops.quant_ops.ck.dequantize_nvfp4(b, sb, bb, output_type=torch.float32)
        return F.linear(left, right).to(out_dtype)

    with mock.patch.object(ops.quant_ops.ck, 'scaled_mm_nvfp4', cpu_oracle) if device == 'cpu' else registry.use_backend('cuda'):
        actual = ops._nvfp4_residual_linear(inputs, weight, None, 3)
    return {'input': 'ones weight [32,64], constant 2048 input [1,64]',
            'bf16_decoded_value': F.linear(inputs, weight.dequantize())[0, 0].item(),
            'residual_finite': bool(actual.isfinite().all()), 'known_fp16_partial_range_limit': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--runtime-host-url')
    parser.add_argument('--hold-id')
    parser.add_argument('--owner-id')
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    check_gpu_window(options)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    root = Path(__file__).resolve().parents[1]
    report = {'versions': {name: importlib.metadata.version(name) for name in ('torch', 'comfy-kitchen', 'transformers')},
              'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ('comfy/ops.py', 'comfy/quant_ops.py', 'comfy/text_encoders/qwen3_omni.py',
                                             'scripts/benchmark_qwen3_omni_quant.py',
                                             'tests-inference/qwen3_omni_quantization_audit.py')},
              'device': options.device, 'production_acceptance': False,
              'matmul': 'CPU FP32 decoded oracle with FP16 partial rounding' if options.device == 'cpu' else 'Comfy Kitchen CUDA NVFP4',
              'package': package_layout(options.artifact), 'numerical': numerical_audit(options),
              'range_probe': range_probe(options.device)}
    if options.device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name()
    report['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    check_gpu_window(options)
    with options.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
