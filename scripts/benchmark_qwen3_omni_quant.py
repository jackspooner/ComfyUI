"""Compare native NVFP4 storage/operations with offline ModelOpt component evidence."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import statistics

import comfy_kitchen as kitchen
from comfy_kitchen.registry import registry
import torch
from torch.nn import functional as F

from scripts.check_qwen3_omni_quant import load_tensor, native_storage
from scripts.qwen3_omni_checkpoint import PROJECTION_SAMPLES, SOURCE_REVISION, inspect_source, sha256_file


def difference(actual, expected):
    delta = actual.float() - expected.float()
    reference_rms = expected.float().square().mean().sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    return {'max_absolute_error': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean().sqrt() / reference_rms).item()}


def warm_timing(operation, repetitions=20):
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {'median_milliseconds': statistics.median(times),
            'temporary_peak_bytes': torch.cuda.max_memory_allocated() - baseline,
            'warmup_runs': 5, 'timed_runs': repetitions}


def benchmark_projection(weight, blocks, scale, reference, captured=None):
    packed, blocked = native_storage(weight, blocks)
    packed, blocked, scale = packed.cuda(), blocked.cuda(), scale.float().cuda()
    reference = F.pad(reference.cuda(), (0, packed.shape[1] * 2 - reference.shape[1]))
    results = []
    for tokens in (1, 16, 128):
        generator = torch.Generator(device='cuda').manual_seed(17)
        inputs = torch.randn(tokens, reference.shape[1], device='cuda', dtype=torch.bfloat16, generator=generator)
        original_width = weight.shape[1] * 2
        inputs[:, original_width:] = 0
        inputs[0, 0] = 20
        input_kind = 'fixed_random_with_outlier'
        if captured is not None:
            if captured.ndim != 2 or captured.shape[1] != original_width or not captured.shape[0]:
                raise ValueError('Captured activation shape differs from source projection')
            inputs = captured.repeat((tokens + captured.shape[0] - 1) // captured.shape[0], 1)[:tokens].cuda()
            inputs = F.pad(inputs, (0, reference.shape[1] - original_width))
            input_kind = 'captured_reference_rows_repeated_to_shape'
        # Match TensorCoreNVFP4Layout's scale computation, including BF16 rounding.
        tensor_scale = (inputs.abs().amax() / (448 * 6)).float()
        qx, sx = kitchen.quantize_nvfp4(inputs, tensor_scale, pad_16x=tokens % 16 != 0)
        expected = F.linear(inputs, reference)
        baseline = warm_timing(lambda: F.linear(inputs, reference))
        for backend, description in registry.list_backends().items():
            if not description['available'] or description['disabled'] or 'scaled_mm_nvfp4' not in description['capabilities']:
                continue
            arguments = dict(a=qx, b=packed, tensor_scale_a=tensor_scale, tensor_scale_b=scale,
                             block_scale_a=sx, block_scale_b=blocked, out_dtype=torch.bfloat16,
                             alpha=tensor_scale * scale)
            eligible = registry.validate_backend_for_call(backend, 'scaled_mm_nvfp4', arguments)
            if not eligible.success:
                results.append({'tokens': tokens, 'backend': backend, 'eligible': False,
                                'reason': eligible.failure_reason})
                continue
            with registry.use_backend(backend):
                def operation():
                    return kitchen.scaled_mm_nvfp4(**arguments)[:tokens]
                actual = operation()
                timing = warm_timing(operation)
                def quantized_operation():
                    current_scale = (inputs.abs().amax() / (448 * 6)).float()
                    current_q, current_blocks = kitchen.quantize_nvfp4(inputs, current_scale, pad_16x=tokens % 16 != 0)
                    return kitchen.scaled_mm_nvfp4(current_q, packed, current_scale, scale,
                                                   current_blocks, blocked, out_dtype=torch.bfloat16,
                                                   alpha=current_scale * scale)[:tokens]
                complete_timing = warm_timing(quantized_operation)
                zero = torch.zeros_like(inputs)
                zero_scale = (zero.abs().amax() / (448 * 6)).float()
                zero_q, zero_blocks = kitchen.quantize_nvfp4(zero, zero_scale, pad_16x=tokens % 16 != 0)
                zero_output = kitchen.scaled_mm_nvfp4(zero_q, packed, zero_scale, scale,
                                                      zero_blocks, blocked, out_dtype=torch.bfloat16)[:tokens]
            results.append({'tokens': tokens, 'backend': backend, 'eligible': True,
                            'inputs': input_kind, 'zero_input_exact': bool(torch.equal(zero_output, torch.zeros_like(expected))),
                            'against_bf16_reference': difference(actual, expected),
                            'matmul': timing, 'quantize_and_matmul': complete_timing, 'bf16_matmul': baseline,
                            'native_input_columns': packed.shape[1] * 2})
    return results


def compare(source, components, device, activation_results=None):
    original = json.loads((components / 'components.json').read_text())
    if original['source']['source_revision'] != SOURCE_REVISION or not original['source']['all_shapes_and_dtypes_match']:
        raise ValueError('Component evidence does not identify the verified source')
    if original['versions']['nvidia-modelopt'] != '0.42.0' or original['versions']['transformers'] != '5.20.0':
        raise ValueError('Component reference dependency pins differ')
    projections = {item['prefix']: item for item in original['projections']}
    if set(projections) != set(PROJECTION_SAMPLES):
        raise ValueError('Component evidence does not cover the fixed projection suite')
    records, _ = inspect_source(source)
    report = {'source_revision': SOURCE_REVISION, 'reference': 'Actual ModelOpt 0.42.0 component decode',
              'reference_components_sha256': sha256_file(components / 'components.json'),
              'device': device, 'accelerated_execution_accepted': False,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'comfy-kitchen')}, 'projections': []}
    activations = {}
    if activation_results is not None:
        evidence = json.loads(activation_results.read_text())
        if evidence['source']['source_revision'] != SOURCE_REVISION:
            raise ValueError('Activation evidence source revision differs')
        item = evidence['activations']
        path = activation_results.parent / item['file']
        if path.resolve().parent != activation_results.resolve().parent or sha256_file(path) != item['sha256']:
            raise ValueError('Unsafe or changed activation evidence')
        activations = torch.load(path, map_location='cpu', weights_only=True)
        report['activation_results_sha256'] = sha256_file(activation_results)
        report['missing_activation_probes'] = sorted(set(PROJECTION_SAMPLES) - set(activations))
    if device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name()
    for prefix in PROJECTION_SAMPLES:
        item = projections[prefix]
        path = components / item['file']
        if path.resolve().parent != components.resolve() or sha256_file(path) != item['sha256']:
            raise ValueError(f'Unsafe or changed component evidence: {item["file"]}')
        reference = torch.load(path, map_location='cpu', weights_only=True)
        weight, blocks, scale = [load_tensor(records, prefix + suffix) for suffix in
                                 ('.weight', '.weight_scale', '.weight_scale_2')]
        packed, blocked = native_storage(weight, blocks)
        width = weight.shape[1] * 2
        actual = kitchen.dequantize_nvfp4(packed, scale.float(), blocked, output_type=torch.float32)[:, :width]
        if not torch.equal(actual, reference['fp32']):
            raise ValueError(f'Native FP32 decode differs from the ModelOpt decoder: {prefix}')
        bf16 = kitchen.dequantize_nvfp4(packed, scale.float(), blocked, output_type=torch.bfloat16)[:, :width]
        result = {'prefix': prefix, 'fp32_decode_equal': True,
                  'cpu_bf16_decode': difference(bf16, reference['bf16']),
                  'source_shape': list(reference['fp32'].shape), 'native_shape': [packed.shape[0], packed.shape[1] * 2]}
        if device == 'cuda':
            result['benchmarks'] = benchmark_projection(weight, blocks, scale, reference['bf16'])
            if prefix in activations:
                result['captured_benchmarks'] = benchmark_projection(weight, blocks, scale, reference['bf16'], activations[prefix])
        report['projections'].append(result)
        print(prefix, 'equal', flush=True)
    report['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='CUDA requires an independently established idle GPU window and ownership')
    parser.add_argument('--activation-results', type=Path, help='Reference results.json with checksummed captured activations')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    result = compare(args.source, args.components, args.device, args.activation_results)
    with args.report.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
