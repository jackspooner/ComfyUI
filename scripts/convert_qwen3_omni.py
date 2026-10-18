"""Offline, lossless ModelOpt NVFP4 packaging for the pinned Qwen3-Omni Thinker."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import subprocess
import sys
import tempfile

from comfy_kitchen.float_utils import from_blocked, swap_nibbles, to_blocked
from safetensors import safe_open
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_moss_audio import copy_payload, read_json
from scripts.qwen3_omni_checkpoint import SOURCE_MANIFEST, SourceTensors, sha256_file, validate_source


ASSETS = {'tokenizer_json': 'tokenizer.json', 'qwen3_omni_config_json': 'config.json',
          'qwen3_omni_processor_json': 'processor_config.json',
          'qwen3_omni_tokenizer_config_json': 'tokenizer_config.json',
          'qwen3_omni_chat_template': 'chat_template.jinja',
          'qwen3_omni_quant_config_json': 'hf_quant_config.json'}
EXPERT = re.compile(r'^(thinker\.model\.layers\.\d+\.mlp\.experts)\.\d+\.')


def mapped_name(key):
    if not key.startswith('thinker.'):
        raise ValueError(f'Not a Thinker tensor: {key}')
    name = key.removeprefix('thinker.')
    if name.startswith('lm_head.'):
        return 'model.' + name
    if name.startswith('visual.'):
        name = name.replace('visual.merger_list.', 'visual.deepstack_merger_list.')
        if name.startswith(('visual.merger.', 'visual.deepstack_merger_list.')):
            name = name.replace('.ln_q.', '.norm.').replace('.mlp.0.', '.linear_fc1.').replace('.mlp.2.', '.linear_fc2.')
    return name


def packed_weight(weight):
    packed = F.pad(swap_nibbles(weight), (0, -weight.shape[1] % 16))
    if not torch.equal(swap_nibbles(packed)[:, :weight.shape[1]], weight):
        raise ValueError('NVFP4 packing changed source bytes')
    return packed


def blocked_scales(scales):
    if not torch.isfinite(scales.float()).all():
        raise ValueError('NVFP4 block scales contain non-finite values')
    padded = F.pad(scales.view(torch.uint8), (0, -scales.shape[1] % 2)).view(scales.dtype)
    blocked = to_blocked(padded, flatten=False)
    restored = from_blocked(blocked, *padded.shape)[:, :scales.shape[1]]
    if not torch.equal(restored.view(torch.uint8), scales.view(torch.uint8)):
        raise ValueError('NVFP4 scale permutation changed source bytes')
    return blocked


def make_plan(records, assets, num_experts):
    plan, consumed, excluded, expert_prefixes = {}, {}, {}, set()

    def add(name, kind, dtype, shape, groups=(), data=None):
        if name in plan:
            raise ValueError(f'Converted tensor collision: {name}')
        plan[name] = dict(kind=kind, dtype=dtype, shape=shape, groups=groups, data=data)
        for group in groups:
            for key in group:
                if key in consumed:
                    raise ValueError(f'Source tensor consumed twice: {key}')
                consumed[key] = name

    def quantized(prefixes, target):
        rows = sum(records[prefix + '.weight'][2]['shape'][0] for prefix in prefixes[0])
        packed_cols = records[prefixes[0][0] + '.weight'][2]['shape'][1]
        native_cols = (packed_cols + 15) // 16 * 16
        leading = [len(prefixes)] if len(prefixes) > 1 else []
        add(target + '.weight', 'packed', 'U8', [*leading, rows, native_cols],
            [[prefix + '.weight' for prefix in group] for group in prefixes])
        add(target + '.weight_scale', 'blocked', 'F8_E4M3', [*leading, (rows + 127) // 128 * 128, (native_cols // 8 + 3) // 4 * 4],
            [[prefix + '.weight_scale' for prefix in group] for group in prefixes])
        add(target + '.weight_scale_2', 'global', 'BF16', leading,
            [[prefix + '.weight_scale_2' for prefix in group] for group in prefixes])
        descriptor = {'format': 'nvfp4'}
        if leading:
            descriptor['num_experts'] = len(prefixes)
        data = json.dumps(descriptor, separators=(',', ':')).encode()
        add(target + '.comfy_quant', 'embedded', 'U8', [len(data)], data=data)

    for key, (_, _, entry) in sorted(records.items()):
        if not key.startswith('thinker.'):
            if not key.startswith(('talker.', 'code2wav.')):
                raise ValueError(f'Unrecognized source stage: {key}')
            excluded[key] = {'dtype': entry['dtype'], 'shape': entry['shape']}
            continue
        if key in consumed:
            continue
        expert = EXPERT.match(key)
        if expert:
            prefix = expert.group(1)
            if prefix in expert_prefixes:
                raise ValueError(f'Unaccounted expert tensor: {key}')
            expert_prefixes.add(prefix)
            for bank, projections in [('gate_up_proj', ('gate_proj', 'up_proj')), ('down_proj', ('down_proj',))]:
                groups = [[f'{prefix}.{i}.{projection}' for projection in projections] for i in range(num_experts)]
                quantized(groups, mapped_name(prefix + '.' + bank))
        elif entry['dtype'] == 'U8' and key.endswith('.weight'):
            quantized([[key.removesuffix('.weight')]], mapped_name(key.removesuffix('.weight')))
        else:
            add(mapped_name(key), 'copy', entry['dtype'], entry['shape'], [[key]])
    if set(consumed) | set(excluded) != set(records):
        raise ValueError('Source tensor accounting is incomplete')
    for name, data in assets.items():
        add(name, 'embedded', 'U8', [len(data)], data=data)
    return plan, consumed, excluded


def transformed_parts(spec, tensors):
    for group in spec['groups']:
        values = [tensors(key) for key in group]
        if spec['kind'] == 'global':
            if not torch.isfinite(values[0]).all():
                raise ValueError(f'Non-finite global scale: {group[0]}')
            for value in values[1:]:
                if not torch.equal(value.reshape(-1).view(torch.uint8), values[0].reshape(-1).view(torch.uint8)):
                    raise ValueError(f'Gate/up global scales differ: {group}')
            value = values[0]
        else:
            value = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            value = packed_weight(value) if spec['kind'] == 'packed' else blocked_scales(value)
        yield value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def stream_checkpoint(source, records, plan, destination, provenance):
    destination = Path(destination)
    manifest_path = destination.with_suffix('.manifest.json')
    if os.path.lexists(destination) or os.path.lexists(manifest_path):
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {'__metadata__': {'format': 'pt', 'model_type': 'qwen3_omni', 'source_revision': provenance['source_revision'],
                               'weight_precision': 'ModelOpt NVFP4, lossless layout conversion',
                               'activation_precision': 'native NVFP4', 'kv_cache_precision': 'bfloat16'}}
    offset = 0
    for key, spec in sorted(plan.items()):
        length = math.prod(spec['shape']) * {'U8': 1, 'F8_E4M3': 1, 'BF16': 2}[spec['dtype']]
        header[key] = dict(dtype=spec['dtype'], shape=spec['shape'], data_offsets=[offset, offset + length])
        offset += length
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    temp_path = manifest_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name + '.', delete=False) as output, ExitStack() as stack:
            temp_path = Path(output.name)
            tensors = SourceTensors(source, stack)
            output.write(struct.pack('<Q', len(encoded)))
            output.write(encoded)
            hashes = {}
            for key, spec in sorted(plan.items()):
                start = output.tell()
                if spec['kind'] == 'copy':
                    path, base, entry = records[spec['groups'][0][0]]
                    first, last = entry['data_offsets']
                    with path.open('rb') as original:
                        original.seek(base + first)
                        hashes[key] = copy_payload(original, output, last - first)
                else:
                    digest = hashlib.sha256()
                    parts = [spec['data']] if spec['kind'] == 'embedded' else transformed_parts(spec, tensors)
                    for part in parts:
                        output.write(part)
                        digest.update(part)
                    hashes[key] = digest.hexdigest()
                expected = header[key]['data_offsets']
                if output.tell() - start != expected[1] - expected[0]:
                    raise ValueError(f'Converted payload length differs: {key}')
            output.flush()
            os.fsync(output.fileno())
        with safe_open(temp_path, framework='pt', device='cpu') as tensors:
            if set(tensors.keys()) != set(plan):
                raise ValueError('Converted safetensors inventory differs')
        whole = hashlib.sha256()
        with temp_path.open('rb') as output:
            whole.update(output.read(8 + len(encoded)))
            for key in sorted(plan):
                first, last = header[key]['data_offsets']
                digest, remaining = hashlib.sha256(), last - first
                while remaining:
                    data = output.read(min(remaining, 8 * 1024 * 1024))
                    if not data:
                        raise ValueError(f'Truncated converted tensor: {key}')
                    digest.update(data)
                    whole.update(data)
                    remaining -= len(data)
                if digest.hexdigest() != hashes[key]:
                    raise ValueError(f'Converted tensor checksum differs: {key}')
        manifest = {**provenance, 'output_sha256': whole.hexdigest(), 'output_bytes': temp_path.stat().st_size,
                    'output_tensors': {key: {**header[key], 'sha256': hashes[key], 'transformation': plan[key]['kind']} for key in sorted(plan)},
                    'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, prefix='.' + manifest_path.name + '.', delete=False) as stream:
            manifest_temp = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(manifest_temp, manifest_path)
        try:
            os.link(temp_path, destination)
        except BaseException:
            manifest_path.unlink()
            raise
    finally:
        for path in (temp_path, manifest_temp):
            if path is not None:
                path.unlink(missing_ok=True)
    return manifest


def convert(source, destination, revision):
    if destination.suffix != '.safetensors':
        raise ValueError('Destination must have the .safetensors extension')
    if os.path.lexists(destination) or os.path.lexists(destination.with_suffix('.manifest.json')):
        raise FileExistsError(destination)
    records, evidence = validate_source(source, revision)
    assets = {key: (source / filename).read_bytes() for key, filename in ASSETS.items()}
    config = read_json(source / 'config.json')['thinker_config']['text_config']
    plan, mapping, excluded = make_plan(records, assets, config['num_experts'])
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'scripts/convert_qwen3_omni.py',
                            'scripts/qwen3_omni_checkpoint.py', 'scripts/qwen3_omni_source.json'], cwd=root, text=True, capture_output=True)
    provenance = {**evidence, 'source_files': read_json(SOURCE_MANIFEST)['files'],
                  'converter_git_revision': git.stdout.strip() if git.returncode == 0 else None,
                  'converter_sources_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                  'converter_source_sha256': sha256_file(Path(__file__)),
                  'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'comfy-kitchen', 'safetensors')},
                  'source_to_target': mapping, 'excluded_speech_tensors': excluded,
                  'expert_order': list(range(config['num_experts'])), 'gate_up_order': ['gate_proj', 'up_proj'],
                  'alignment': 'Zero-extend input columns to multiples of 32; preserve original packed values and scales',
                  'precision': {'weights': 'NVFP4', 'activations': 'NVFP4', 'cache': 'BF16', 'publisher_w4a8_parity': False}}
    return stream_checkpoint(source, records, plan, destination, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = convert(args.source.resolve(), args.destination.absolute(), args.source_revision)
    print(f'Published {args.destination}: {result["output_bytes"]} bytes, sha256 {result["output_sha256"]}')


if __name__ == '__main__':
    main()
