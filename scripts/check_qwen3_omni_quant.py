"""Offline storage gate for the Qwen3-Omni ModelOpt checkpoint; never loads a model."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import struct

import comfy_kitchen as kitchen
from comfy_kitchen.float_utils import from_blocked, swap_nibbles, to_blocked
from safetensors import safe_open
import torch

from scripts.convert_moss_audio import read_json, unique_object


def inspect_source(source):
    # Diagnostics retain a stale total_size as a failed gate while still examining
    # independently valid shards. The production converter must not silently repair it.
    index = read_json(source / 'model.safetensors.index.json')
    records = {}
    for filename in sorted(set(index['weight_map'].values())):
        path = source / filename
        if path.resolve().parent != source.resolve():
            raise ValueError(f'Shard escapes source directory: {filename}')
        with safe_open(path, framework='pt', device='cpu') as tensors:
            keys = set(tensors.keys())
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError(f'Oversized shard header: {filename}')
            header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
        if keys != set(header) - {'__metadata__'}:
            raise ValueError(f'Invalid shard inventory: {filename}')
        for key in keys:
            if key in records or index['weight_map'].get(key) != filename:
                raise ValueError(f'Duplicate or misassigned tensor: {key}')
            records[key] = (path, header_size + 8, header[key])
    if set(records) != set(index['weight_map']):
        raise ValueError('Shard inventory differs from index')
    actual = sum(r[2]['data_offsets'][1] - r[2]['data_offsets'][0] for r in records.values())
    declared = index['metadata']['total_size']
    return records, {'declared_payload_bytes': declared, 'actual_payload_bytes': actual,
                     'size_matches': actual == declared}


def source_decode(weight, block_scale, scale):
    # ModelOpt 0.42.0 NVFP4QTensor: low nibble first, row-major groups of 16.
    # This is an independent format decoder, not a full-model reference runtime.
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    values = levels[codes].reshape(*block_scale.shape, 16)
    return (values * (block_scale.float() * scale.float()).unsqueeze(-1)).flatten(-2)


def native_storage(weight, block_scale):
    return swap_nibbles(weight), to_blocked(block_scale, flatten=False)


def check_projection(weight, block_scale, scale):
    packed, blocked = native_storage(weight, block_scale)
    if not torch.equal(swap_nibbles(packed), weight):
        raise ValueError('Packed-weight round trip changed bytes')
    restored = from_blocked(blocked, *block_scale.shape)
    if not torch.equal(restored.view(torch.uint8), block_scale.view(torch.uint8)):
        raise ValueError('Block-scale round trip changed bytes')
    reference = source_decode(weight, block_scale, scale)
    actual = kitchen.dequantize_nvfp4(packed, scale.float(), blocked, output_type=torch.float32)
    if not torch.equal(reference, actual):
        raise ValueError(f'FP32 format decode differs: {(reference - actual).abs().max().item()}')
    if not torch.isfinite(actual).all():
        raise ValueError('Decoded weights contain non-finite values')
    return {'logical_shape': list(actual.shape), 'packed_roundtrip': True,
            'scale_roundtrip': True, 'fp32_decode_max_abs_error': 0.0}


def check_expert(records, prefix):
    reference_weights = []
    native_weights = []
    for projection in ('gate', 'up', 'down'):
        name = prefix + '.' + projection + '_proj'
        weight, blocks, scale = [load_tensor(records, name + suffix) for suffix in
                                 ('.weight', '.weight_scale', '.weight_scale_2')]
        reference_weights.append(source_decode(weight, blocks, scale))
        packed, blocked = native_storage(weight, blocks)
        native_weights.append(kitchen.dequantize_nvfp4(packed, scale.float(), blocked, output_type=torch.float32))
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(16, reference_weights[0].shape[1], generator=generator)
    inputs[0].zero_()
    inputs[1, 0] = 20
    outputs = []
    for gate, up, down in (reference_weights, native_weights):
        outputs.append((torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T)
    if not torch.equal(*outputs) or not torch.isfinite(outputs[0]).all():
        raise ValueError(f'FP32 expert path differs or is non-finite: {prefix}')
    return {'tokens': 16, 'seed': 17, 'fp32_output_max_abs_error': 0.0,
            'execution': 'CPU decoded-weight diagnostic, not accelerated inference'}


def load_tensor(records, key):
    path, _, _ = records[key]
    with safe_open(path, framework='pt', device='cpu') as tensors:
        return tensors.get_tensor(key).clone()


def check_inventory(records):
    counts = Counter()
    sizes = Counter()
    quantized = []
    for key, (_, _, entry) in records.items():
        component = '.'.join(key.split('.')[:2])
        counts[component + ':' + entry['dtype']] += 1
        start, end = entry['data_offsets']
        sizes[key.split('.')[0]] += end - start
        if entry['dtype'] != 'U8':
            continue
        if not key.endswith('.weight') or len(entry['shape']) != 2:
            raise ValueError(f'Unrecognized packed tensor: {key}')
        prefix = key.removesuffix('.weight')
        rows, packed_cols = entry['shape']
        if packed_cols % 8:
            raise ValueError(f'Incomplete block of 16: {key}')
        for suffix, dtype, shape in [('.weight_scale', 'F8_E4M3', [rows, packed_cols // 8]),
                                      ('.weight_scale_2', 'BF16', [])]:
            scale_key = prefix + suffix
            if scale_key not in records:
                raise ValueError(f'Missing quantization scale: {scale_key}')
            scale = records[scale_key][2]
            if scale['dtype'] != dtype or scale['shape'] != shape:
                raise ValueError(f'Unexpected scale layout: {scale_key}: {scale}')
        quantized.append(prefix)
    return {'tensor_count': len(records), 'component_dtypes': dict(sorted(counts.items())),
            'payload_bytes_by_stage': dict(sizes), 'quantized_projections': len(quantized)}


def check_hashes(records, manifest_path):
    manifest = read_json(manifest_path)
    expected = {f['name']: f for f in manifest['files']}
    results = []
    for path in sorted({record[0] for record in records.values()}):
        item = expected[path.name]
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Shard size differs from download receipt: {path.name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != item['sha256'] or actual != item['expected_sha256']:
            raise ValueError(f'Shard hash differs from publisher: {path.name}')
        results.append({'name': path.name, 'bytes': item['bytes'], 'sha256': actual})
    return results


def audit(source, revision, manifest_path):
    manifest = read_json(manifest_path)
    if manifest['revision'] != revision:
        raise ValueError('Source revision differs from download receipt')
    records, integrity = inspect_source(source)
    result = {'source_revision': revision, 'reference_kind': 'ModelOpt 0.42.0 format equation only',
              'full_model_parity': False, 'accelerated_execution_accepted': False,
              'versions': {name: importlib.metadata.version(name) for name in
                           ('torch', 'transformers', 'comfy-kitchen', 'safetensors')},
              'source_integrity': integrity, 'inventory': check_inventory(records),
              'shards': check_hashes(records, manifest_path)}
    prefixes = ['thinker.model.layers.0.self_attn.q_proj',
                'thinker.audio_tower.conv_out', 'thinker.audio_tower.layers.0.self_attn.k_proj',
                'thinker.visual.blocks.0.attn.qkv', 'thinker.visual.merger.mlp.0']
    for layer in (0, 23, 47):
        for expert in (0, 63, 127):
            prefixes.extend(f'thinker.model.layers.{layer}.mlp.experts.{expert}.{proj}_proj'
                            for proj in ('gate', 'up', 'down'))
    result['projections'] = {}
    for prefix in prefixes:
        tensors = [load_tensor(records, prefix + suffix) for suffix in ('.weight', '.weight_scale', '.weight_scale_2')]
        result['projections'][prefix] = check_projection(*tensors)
        del tensors
    result['expert_paths'] = {f'{layer}:{expert}': check_expert(records, f'thinker.model.layers.{layer}.mlp.experts.{expert}')
                              for layer in (0, 23, 47) for expert in (0, 63, 127)}
    # Check every expert before assuming fused gate/up storage is exact.
    config = read_json(source / 'config.json')['thinker_config']['text_config']
    global_scales = {}
    for path in sorted({record[0] for record in records.values()}):
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                if key.startswith('thinker.model.') and key.endswith(('.gate_proj.weight_scale_2', '.up_proj.weight_scale_2')):
                    global_scales[key] = tensors.get_tensor(key).item()
    unequal = []
    for layer in range(config['num_hidden_layers']):
        for expert in range(config['num_experts']):
            prefix = f'thinker.model.layers.{layer}.mlp.experts.{expert}'
            gate = global_scales[prefix + '.gate_proj.weight_scale_2']
            up = global_scales[prefix + '.up_proj.weight_scale_2']
            if gate != up:
                unequal.append(prefix)
    result['gate_up_unequal_global_scales'] = unequal
    result['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--download-manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(2)
    result = audit(args.source, args.source_revision, args.download_manifest)
    # A failed run publishes no success report; exclusive creation protects older evidence.
    with args.report.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(f'Checked {len(result["projections"])} projections; full-model and accelerated acceptance remain pending.')
    if not result['source_integrity']['size_matches']:
        raise SystemExit('Source integrity gate failed: index total_size differs from shard payloads; see report.')


if __name__ == '__main__':
    main()
