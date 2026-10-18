"""Pinned source inventory shared by offline Omni validation and conversion."""
import hashlib
import json
import struct
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration

from scripts.convert_moss_audio import read_json, unique_object


SOURCE_MANIFEST = Path(__file__).with_name('qwen3_omni_source.json')
SOURCE_REVISION = '0ed3aa2b00eeedbb0d38b6d4738f506af759213b'
ORIGINAL_MANIFEST = Path(__file__).with_name('qwen3_omni_original_source.json')
ORIGINAL_REVISION = '26291f793822fb6be9555850f06dfe95f2d7e695'
FP8_MANIFEST = Path(__file__).with_name('qwen3_omni_fp8_source.json')
FP8_REVISION = '042db39a102171d5994b72ab717fade92e80061a'
PROJECTION_SAMPLES = ['thinker.model.layers.0.self_attn.q_proj',
                      'thinker.audio_tower.conv_out', 'thinker.audio_tower.layers.0.self_attn.k_proj',
                      'thinker.visual.blocks.0.attn.qkv', 'thinker.visual.merger.mlp.0',
                      'thinker.visual.blocks.0.mlp.linear_fc2'] + [
    f'thinker.model.layers.{layer}.mlp.experts.{expert}.{projection}_proj'
    for layer in (0, 23, 47) for expert in (0, 63, 127) for projection in ('gate', 'up', 'down')]


class SourceTensors:
    def __init__(self, source, stack):
        index = json.loads((source / 'model.safetensors.index.json').read_text())
        self.weight_map = index['weight_map']
        self.shards = {name: stack.enter_context(safe_open(source / name, framework='pt', device='cpu'))
                       for name in sorted(set(self.weight_map.values()))}

    def contains(self, key):
        return key in self.weight_map

    def __call__(self, key):
        return self.shards[self.weight_map[key]].get_tensor(key)


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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_layout(source, quantized=True):
    config = Qwen3OmniMoeConfig.from_pretrained(source, local_files_only=True)
    with torch.device('meta'):
        model = Qwen3OmniMoeForConditionalGeneration(config)
    shapes = {}
    quantized_names = set()
    for key, tensor in model.state_dict().items():
        shape = list(tensor.shape)
        if '.experts.' in key and key.endswith(('gate_up_proj', 'down_proj')):
            prefix, kind = key.rsplit('.', 1)
            projections = ('gate_proj', 'up_proj') if kind == 'gate_up_proj' else ('down_proj',)
            rows = shape[1] // len(projections)
            for expert in range(shape[0]):
                for projection in projections:
                    name = f'{prefix}.{expert}.{projection}.weight'
                    shapes[name] = [rows, shape[2]]
                    if key.startswith('thinker.'):
                        quantized_names.add(name)
        else:
            shapes[key] = shape
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith('thinker.') and name != 'thinker.lm_head':
            quantized_names.add(name + '.weight')
    layout = {key: ('BF16', shape) for key, shape in shapes.items()}
    if not quantized:
        return layout
    for key in quantized_names:
        rows, cols = shapes[key]
        prefix = key.removesuffix('.weight')
        layout[key] = ('U8', [rows, cols // 2])
        layout[prefix + '.weight_scale'] = ('F8_E4M3', [rows, cols // 16])
        layout[prefix + '.weight_scale_2'] = ('BF16', [])
    return layout


def validate_layout(records, expected):
    missing, extra = set(expected) - set(records), set(records) - set(expected)
    if missing or extra:
        raise ValueError(f'Source tensor inventory differs: missing {sorted(missing)}, extra {sorted(extra)}')
    for key, (dtype, shape) in expected.items():
        entry = records[key][2]
        if entry['dtype'] != dtype or entry['shape'] != shape:
            raise ValueError(f'Unexpected source shape/dtype for {key}: {entry["shape"]} {entry["dtype"]}; expected {shape} {dtype}')


def expected_fp8_layout(source):
    layout = expected_layout(source, quantized=False)
    index = read_json(source / 'model.safetensors.index.json')
    for key in index['weight_map']:
        if not key.endswith('.weight_scale_inv'):
            continue
        weight = key.removesuffix('_scale_inv')
        if weight not in layout or len(layout[weight][1]) != 2:
            raise ValueError(f'FP8 scale has no architecture projection: {key}')
        shape = layout[weight][1]
        layout[weight] = ('F8_E4M3', shape)
        layout[key] = ('BF16', [(size + 127) // 128 for size in shape])
    return layout


def validate_source(source, revision):
    manifests = {SOURCE_REVISION: SOURCE_MANIFEST, ORIGINAL_REVISION: ORIGINAL_MANIFEST, FP8_REVISION: FP8_MANIFEST}
    if revision not in manifests:
        raise ValueError(f'Only the verified source revisions are supported: {list(manifests)}')
    source = source.resolve(strict=True)
    manifest = read_json(manifests[revision])
    for entry in manifest['files']:
        path = source / entry['name']
        if path.resolve().parent != source or not path.is_file():
            raise ValueError(f'Source file must be directly inside the source directory: {entry["name"]}')
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError(f'Source size/hash differs from pinned revision: {entry["name"]}')
    records, totals = inspect_source(source)
    validate_layout(records, expected_fp8_layout(source) if revision == FP8_REVISION else expected_layout(source, quantized=revision == SOURCE_REVISION))
    if totals['declared_payload_bytes'] != manifest['declared_payload_bytes'] or totals['actual_payload_bytes'] != manifest['actual_payload_bytes']:
        raise ValueError('Unexpected source payload total; the pinned metadata reconciliation does not apply')
    evidence = {'source_revision': revision, 'verified_files': len(manifest['files']),
                'verified_tensors': len(records), 'all_shapes_and_dtypes_match': True,
                'metadata_reconciliation': {**totals, 'correction_bytes': totals['actual_payload_bytes'] - totals['declared_payload_bytes'],
                                            'reason': 'Pinned publisher index aggregate is stale; every source file hash and every expected tensor matches.' if revision != ORIGINAL_REVISION else 'Original index payload total matches all verified tensors.'}}
    return records, evidence
