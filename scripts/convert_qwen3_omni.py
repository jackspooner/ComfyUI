"""Offline source-verified NVFP4/block FP8 packaging or original-weight FP8 conversion for Qwen3-Omni."""
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
import shutil
import struct
import subprocess
import sys
import tempfile

from comfy_kitchen.float_utils import from_blocked, swap_nibbles, to_blocked
from safetensors import safe_open
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3OmniMoeConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_moss_audio import copy_payload, read_json
from scripts.qwen3_omni_checkpoint import FP8_MANIFEST, FP8_REVISION, ORIGINAL_MANIFEST, ORIGINAL_REVISION, SOURCE_MANIFEST, SOURCE_REVISION, SourceTensors, sha256_file, validate_source
from comfy.quant_ops import TensorCoreFP8E4M3Layout


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
            name = name.replace('.ln_q.', '.norm.')
            name = re.sub(r'\.mlp\.([02])(?=\.|$)', lambda match: '.linear_fc1' if match[1] == '0' else '.linear_fc2', name)
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


def make_plan(records, assets, num_experts, quantization='nvfp4'):
    if quantization not in ('nvfp4', 'fp8', 'fp8-block128'):
        raise ValueError('Choose nvfp4 preservation, original fp8 conversion or published fp8-block128 preservation')
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
        leading = [len(prefixes)] if len(prefixes) > 1 else []
        if quantization == 'fp8-block128':
            if len(prefixes[0]) > 1 and any(records[p + '.weight'][2]['shape'][0] % 128 for p in prefixes[0]):
                raise ValueError(f'FP8 gate/up fusion requires block-aligned rows: {target}')
            add(target + '.weight', 'fp8_block_weight', 'F8_E4M3', [*leading, rows, packed_cols],
                [[p + '.weight' for p in group] for group in prefixes])
            add(target + '.weight_scale', 'fp8_block_scale', 'F32', [*leading, (rows + 127) // 128, (packed_cols + 127) // 128],
                [[p + '.weight_scale_inv' for p in group] for group in prefixes])
            descriptor = {'format': 'float8_e4m3fn_block128', 'full_precision_matrix_mult': True}
            if leading:
                descriptor['num_experts'] = len(prefixes)
            data = json.dumps(descriptor, separators=(',', ':')).encode()
            add(target + '.comfy_quant', 'embedded', 'U8', [len(data)], data=data)
            return
        if quantization == 'fp8':
            columns = (packed_cols + 31) // 32 * 32
            groups = [[prefix + '.weight' for prefix in group] for group in prefixes]
            add(target + '.weight', 'fp8', 'F8_E4M3', [*leading, rows, columns], groups)
            # A source group owns both its quantized weight and scale.
            plan[target + '.weight']['columns'] = columns
            add(target + '.weight_scale', 'fp8_scale', 'F32', leading)
            plan[target + '.weight_scale']['groups'] = groups
            descriptor = {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True}
            if leading:
                descriptor['num_experts'] = len(prefixes)
            data = json.dumps(descriptor, separators=(',', ':')).encode()
            add(target + '.comfy_quant', 'embedded', 'U8', [len(data)], data=data)
            return
        native_cols = (packed_cols + 15) // 16 * 16
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
        elif key.endswith('.weight') and (entry['dtype'] in ('U8', 'F8_E4M3') or
                quantization == 'fp8' and len(entry['shape']) == 2 and not key.endswith(('embed_tokens.weight', 'pos_embed.weight', 'lm_head.weight', 'mlp.gate.weight'))):
            quantized([[key.removesuffix('.weight')]], mapped_name(key.removesuffix('.weight')))
        elif quantization == 'fp8-block128' and key.endswith('mlp.linear_fc2.weight'):
            rows, columns = entry['shape']
            add(mapped_name(key), 'pad_dense', entry['dtype'], [rows, (columns + 31) // 32 * 32], [[key]])
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
        if spec['kind'] in ('fp8', 'fp8_scale'):
            value = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError(f'Original projection must contain finite BF16 weights: {group}')
            scale = (value.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max).clamp_min(torch.finfo(torch.float32).tiny)
            if spec['kind'] == 'fp8_scale':
                value = scale
            else:
                value = F.pad(value, (0, spec['columns'] - value.shape[1]))
                value, _ = TensorCoreFP8E4M3Layout.quantize(value.float(), scale=scale)
        elif spec['kind'] in ('fp8_block_weight', 'fp8_block_scale', 'pad_dense'):
            value = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            if spec['kind'] != 'pad_dense' and not torch.isfinite(value.float()).all():
                raise ValueError(f'Non-finite published FP8 data: {group}')
            if spec['kind'] == 'fp8_block_scale' and not (value > 0).all():
                raise ValueError(f'Non-positive published FP8 scale: {group}')
            if spec['kind'] == 'fp8_block_scale':
                # This pinned publisher stores quantization multipliers, despite
                # the standard HF dequantization-scale name weight_scale_inv.
                value = value.float().reciprocal()
            if spec['kind'] == 'pad_dense':
                value = F.pad(value, (0, spec['shape'][-1] - value.shape[-1]))
        elif spec['kind'] == 'global':
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
    precision = provenance.get('precision', {'weights': 'ModelOpt NVFP4, lossless layout conversion', 'activations': 'native NVFP4', 'cache': 'bfloat16'})
    header = {'__metadata__': {'format': 'pt', 'model_type': 'qwen3_omni', 'source_revision': provenance['source_revision'],
                               'weight_precision': precision['weights'],
                               'activation_precision': precision['activations'], 'kv_cache_precision': precision['cache']}}
    offset = 0
    for key, spec in sorted(plan.items()):
        length = math.prod(spec['shape']) * {'U8': 1, 'F8_E4M3': 1, 'BF16': 2, 'F32': 4}[spec['dtype']]
        header[key] = dict(dtype=spec['dtype'], shape=spec['shape'], data_offsets=[offset, offset + length])
        offset += length
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    required_bytes = 8 + len(encoded) + offset
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < required_bytes:
        raise OSError(f'Insufficient destination space: weights require {required_bytes} bytes, only {free_bytes} bytes are free. Reserve additional space for the provenance manifest.')
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


def convert(source, destination, revision, quantization='nvfp4'):
    if destination.suffix != '.safetensors':
        raise ValueError('Destination must have the .safetensors extension')
    if os.path.lexists(destination) or os.path.lexists(destination.with_suffix('.manifest.json')):
        raise FileExistsError(destination)
    if quantization not in ('nvfp4', 'fp8', 'fp8-block128'):
        raise ValueError('Choose nvfp4 preservation, original fp8 conversion or published fp8-block128 preservation')
    if revision != {'nvfp4': SOURCE_REVISION, 'fp8': ORIGINAL_REVISION, 'fp8-block128': FP8_REVISION}[quantization]:
        raise ValueError('Use the pinned source revision for the selected mode: ModelOpt NVFP4, original Qwen FP8 conversion or published block FP8.')
    records, evidence = validate_source(source, revision)
    if quantization in ('fp8', 'fp8-block128'):
        processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        model_config = Qwen3OmniMoeConfig.from_pretrained(source, local_files_only=True).to_dict()
        assets = {'tokenizer_json': processor.tokenizer.backend_tokenizer.to_str().encode(),
                  'qwen3_omni_config_json': json.dumps(model_config).encode(),
                  'qwen3_omni_processor_json': json.dumps({'image_processor': processor.image_processor.to_dict(),
                                                         'feature_extractor': processor.feature_extractor.to_dict()}).encode(),
                  'qwen3_omni_tokenizer_config_json': (source / 'tokenizer_config.json').read_bytes(),
                  'qwen3_omni_chat_template': processor.chat_template.encode(),
                  'qwen3_omni_quant_config_json': json.dumps({'quantization': 'float8_e4m3fn_block128' if quantization == 'fp8-block128' else 'float8_e4m3fn', 'activations': 'BF16', 'cache': 'BF16'}).encode()}
    else:
        assets = {key: (source / filename).read_bytes() for key, filename in ASSETS.items()}
        model_config = read_json(source / 'config.json')
    config = model_config['thinker_config']['text_config']
    plan, mapping, excluded = make_plan(records, assets, config['num_experts'], quantization)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'scripts/convert_qwen3_omni.py',
                            'scripts/qwen3_omni_checkpoint.py', 'scripts/qwen3_omni_source.json', 'scripts/qwen3_omni_original_source.json', 'scripts/qwen3_omni_fp8_source.json'], cwd=root, text=True, capture_output=True)
    source_manifest = {'nvfp4': SOURCE_MANIFEST, 'fp8': ORIGINAL_MANIFEST, 'fp8-block128': FP8_MANIFEST}[quantization]
    provenance = {**evidence, 'source_files': read_json(source_manifest)['files'],
                  'converter_git_revision': git.stdout.strip() if git.returncode == 0 else None,
                  'converter_sources_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                  'converter_source_sha256': sha256_file(Path(__file__)),
                  'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'comfy-kitchen', 'safetensors')},
                  'source_to_target': mapping, 'excluded_speech_tensors': excluded,
                  'expert_order': list(range(config['num_experts'])), 'gate_up_order': ['gate_proj', 'up_proj'],
                  'alignment': 'Zero-extend input columns to multiples of 32',
                  'precision': {'weights': 'FP8 E4M3 block128, original weight bytes with normalized inverse scales' if quantization == 'fp8-block128' else 'FP8 E4M3' if quantization == 'fp8' else 'NVFP4',
                                'activations': 'NVFP4' if quantization == 'nvfp4' else 'BF16',
                                'cache': 'BF16', 'publisher_w4a8_parity': False}}
    if quantization == 'fp8-block128':
        provenance['source_scale_convention'] = 'Decode weight / BF16 weight_scale_inv; native weight_scale stores its FP32 reciprocal.'
    return stream_checkpoint(source, records, plan, destination, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--quantization', choices=('nvfp4', 'fp8', 'fp8-block128'), default='nvfp4', help='Preserve pinned NVFP4 or block FP8, or quantize original BF16 projections to FP8; FP8 uses BF16 compute/cache')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = convert(args.source.resolve(), args.destination.absolute(), args.source_revision, args.quantization)
    print(f'Published {args.destination}: {result["output_bytes"]} bytes, sha256 {result["output_sha256"]}')


if __name__ == '__main__':
    main()
