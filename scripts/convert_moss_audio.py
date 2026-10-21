#!/usr/bin/env python3
"""Prepare a local MOSS-Audio checkpoint without loading its tensor payloads into RAM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from safetensors import safe_open
from transformers import Qwen2TokenizerFast

MODEL_REVISION = "aacbb8036eb06dd968c105f91d2663af8fbee720"
IMPLEMENTATION_REVISION = "66326e6e0db34f036c86a76ba005efa4830c69dd"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def mapped_name(name):
    if name.startswith("language_model."):
        return "model." + name.removeprefix("language_model.")
    if name.startswith("lm_head."):
        return "model." + name
    return name


def expected_shapes():
    shapes = {}
    def linear(name, out_size, in_size, bias=False):
        shapes[name + '.weight'] = [out_size, in_size]
        if bias:
            shapes[name + '.bias'] = [out_size]
    for i, channels in enumerate((1, 480, 480), 1):
        shapes[f'audio_encoder.conv{i}.weight'] = [480, channels, 3, 3]
        shapes[f'audio_encoder.conv{i}.bias'] = [480]
    linear('audio_encoder.stem_proj', 1280, 7680, True)
    for i in range(32):
        prefix = f'audio_encoder.layers.{i}'
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            linear(f'{prefix}.self_attn.{name}', 1280, 1280, name != 'k_proj')
        linear(prefix + '.fc1', 5120, 1280, True)
        linear(prefix + '.fc2', 1280, 5120, True)
        for norm in ('self_attn_layer_norm', 'final_layer_norm'):
            for param in ('weight', 'bias'):
                shapes[f'{prefix}.{norm}.{param}'] = [1280]
    for param in ('weight', 'bias'):
        shapes['audio_encoder.layer_norm.' + param] = [1280]
    for prefix in ['audio_adapter'] + [f'deepstack_audio_merger_list.{i}' for i in range(3)]:
        linear(prefix + '.gate_proj', 8192, 1280)
        linear(prefix + '.up_proj', 8192, 1280)
        linear(prefix + '.down_proj', 4096, 8192)
    linear('language_model.embed_tokens', 151936, 4096)
    linear('lm_head', 151936, 4096)
    shapes['language_model.norm.weight'] = [4096]
    for i in range(36):
        prefix = f'language_model.layers.{i}'
        for name, size in [('q_proj',4096), ('k_proj',1024), ('v_proj',1024), ('o_proj',4096)]:
            linear(f'{prefix}.self_attn.{name}', size, 4096)
        for name in ('q_norm', 'k_norm'):
            shapes[f'{prefix}.self_attn.{name}.weight'] = [128]
        for name in ('gate_proj', 'up_proj'):
            linear(f'{prefix}.mlp.{name}', 12288, 4096)
        linear(prefix + '.mlp.down_proj', 4096, 12288)
        for name in ('input_layernorm', 'post_attention_layernorm'):
            shapes[f'{prefix}.{name}.weight'] = [4096]
    return shapes


def inspect_shards(source, expected=None):
    index_path = source / 'model.safetensors.index.json'
    index = read_json(index_path) if index_path.exists() else None
    weight_map = index['weight_map'] if index is not None else None
    if expected is not None and weight_map is not None and set(weight_map) != set(expected):
        raise ValueError(f"Unexpected tensor inventory: missing {sorted(set(expected) - set(weight_map))}, extra {sorted(set(weight_map) - set(expected))}")
    records = {}
    filenames = sorted(set(weight_map.values())) if weight_map is not None else ['model.safetensors']
    for filename in filenames:
        path = source / filename
        if path.resolve().parent != source.resolve():
            raise ValueError(f"Shard must be directly inside the source directory: {filename}")
        if not path.is_file():
            raise ValueError(f"Missing shard: {path}")
        with path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Incomplete shard: {path}")
            size = struct.unpack('<Q', raw)[0]
            if size > 100_000_000 or size + 8 > path.stat().st_size:
                raise ValueError(f"Invalid safetensors header (possibly a Git LFS pointer): {path}")
            header = json.loads(stream.read(size), object_pairs_hook=unique_object)
        with safe_open(path, framework='pt', device='cpu') as tensors:
            keys = set(tensors.keys())
        if keys != set(header) - {'__metadata__'}:
            raise ValueError(f"Invalid shard inventory: {filename}")
        for key in keys:
            if key in records:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            if weight_map is not None and weight_map.get(key) != filename:
                raise ValueError(f"Tensor not assigned to this shard: {key}")
            entry = header[key]
            if expected is not None and key not in expected:
                raise ValueError(f"Unexpected tensor: {key}")
            if expected is not None and (entry['shape'] != expected[key] or entry['dtype'] != 'BF16'):
                raise ValueError(f"Unexpected shape/dtype for {key}: {entry['shape']} {entry['dtype']}")
            records[key] = (path, size + 8, entry)
    if weight_map is not None and set(records) != set(weight_map):
        raise ValueError(f"Missing tensors: {sorted(set(weight_map) - set(records))}")
    if expected is not None and set(records) != set(expected):
        raise ValueError(f"Missing tensors: {sorted(set(expected) - set(records))}")
    total = sum(entry['data_offsets'][1] - entry['data_offsets'][0] for _, _, entry in records.values())
    if index is not None and total != index['metadata']['total_size']:
        raise ValueError('Payload size does not match the shard index')
    return records


def copy_payload(source, destination, count):
    digest = hashlib.sha256()
    while count:
        chunk = source.read(min(count, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError('Source payload ended during conversion')
        destination.write(chunk)
        digest.update(chunk)
        count -= len(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def stream_checkpoint(records, tokenizer_json, destination, metadata, *, source_stats=None, name_mapping=None):
    destination = Path(destination)
    manifest_path = destination.with_suffix('.manifest.json')
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"Destination or manifest already exists: {destination}")
    if source_stats is None:
        source_stats = {path: file_identity(path) for path, _, _ in records.values()}
    for path, identity in source_stats.items():
        if file_identity(path) != identity:
            raise ValueError(f'Source changed before conversion: {path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {'__metadata__': metadata}
    names = {key: mapped_name(key) for key in records} if name_mapping is None else name_mapping
    offset = 0
    for key, (_, _, entry) in sorted(records.items()):
        name = names[key]
        if name in header or name == 'tokenizer_json':
            raise ValueError(f"Mapped tensor collision: {name}")
        length = entry['data_offsets'][1] - entry['data_offsets'][0]
        header[name] = dict(dtype=entry['dtype'], shape=entry['shape'], data_offsets=[offset, offset + length])
        offset += length
    if tokenizer_json is not None:
        header['tokenizer_json'] = dict(dtype='U8', shape=[len(tokenizer_json)], data_offsets=[offset, offset + len(tokenizer_json)])
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    temp_path = None
    manifest_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name + '.', delete=False) as output:
            temp_path = Path(output.name)
            output.write(struct.pack('<Q', len(encoded)))
            output.write(encoded)
            hashes = {}
            for key, (path, base, entry) in sorted(records.items()):
                start, end = entry['data_offsets']
                with path.open('rb') as source:
                    source.seek(base + start)
                    hashes[key] = copy_payload(source, output, end - start)
            if tokenizer_json is not None:
                output.write(tokenizer_json)
            output.flush()
            os.fsync(output.fileno())
        with safe_open(temp_path, framework='pt', device='cpu') as tensors:
            if set(tensors.keys()) != set(header) - {'__metadata__'}:
                raise ValueError('Converted tensor inventory mismatch')
            if tokenizer_json is not None and tensors.get_tensor('tokenizer_json').numpy().tobytes() != tokenizer_json:
                raise ValueError('Embedded tokenizer was changed')
        # Re-read every completed output payload, independently of the copy operation.
        with temp_path.open('rb') as output:
            for key in records:
                start, end = header[names[key]]['data_offsets']
                output.seek(8 + len(encoded) + start)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    chunk = output.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError('Truncated converted payload')
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != hashes[key]:
                    raise ValueError(f'Converted payload differs: {key}')
        for path, identity in source_stats.items():
            if file_identity(path) != identity:
                raise ValueError(f'Source changed during conversion: {path}')
        manifest = dict(metadata=metadata, source_tensors=len(records), source_bytes=offset,
                        tensors={key: dict(destination=names[key], sha256=hashes[key],
                                           source_file=records[key][0].name, shape=records[key][2]['shape'], dtype=records[key][2]['dtype']) for key in sorted(records)})
        if tokenizer_json is not None:
            manifest['tokenizer_sha256'] = hashlib.sha256(tokenizer_json).hexdigest()
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, prefix='.' + manifest_path.name + '.', delete=False) as stream:
            manifest_temp = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Hard links publish atomically with no overwrite, including concurrent converters.
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


def convert(source, destination, model_id, model_revision):
    source = Path(source).resolve()
    config = read_json(source / 'config.json')
    language = config['language_config']
    audio = config['audio_config']
    required_language = dict(vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                             num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=1000000,
                             rms_norm_eps=1e-6, hidden_act='silu', attention_bias=False, max_position_embeddings=40960)
    required_audio = dict(num_mel_bins=128, d_model=1280, output_dim=1280, encoder_layers=32,
                         encoder_attention_heads=20, encoder_ffn_dim=5120, downsample_hidden_size=480,
                         layer_norm_eps=1e-5, activation_function='gelu', deepstack_encoder_layer_indexes=[8,16,24])
    if config['model_type'] != 'moss_audio' or config['adapter_hidden_size'] != 8192 or config.get('deepstack_num_inject_layers', 3) not in (None, 3):
        raise ValueError('Unsupported MOSS-Audio configuration')
    for section, required in ((language, required_language), (audio, required_audio)):
        for key, value in required.items():
            if section.get(key) != value:
                raise ValueError(f'Unsupported MOSS-Audio configuration: {key}={section.get(key)!r}; expected {value!r}')
    if audio.get('n_window', 200) != 200 or audio.get('conv_chunksize', 64) != 64 or language.get('rope_scaling') is not None or language.get('use_sliding_window', False):
        raise ValueError('Unsupported MOSS-Audio chunking or rotary configuration')
    records = inspect_shards(source, expected_shapes())
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    for digit in range(10):
        if tokenizer.encode(str(digit), add_special_tokens=False) != [15 + digit]:
            raise ValueError('Tokenizer digit IDs do not match MOSS timestamps')
    tokenizer_json = tokenizer.backend_tokenizer.to_str().encode('utf-8')
    metadata = dict(format='pt', architecture='moss_audio', conversion_version='1', model_id=model_id,
                    model_revision=model_revision, implementation_revision=IMPLEMENTATION_REVISION,
                    source_config_sha256=hashlib.sha256((source / 'config.json').read_bytes()).hexdigest())
    return stream_checkpoint(records, tokenizer_json, destination, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--model-id', default='OpenMOSS-Team/MOSS-Audio-8B-Thinking')
    parser.add_argument('--model-revision', default=MODEL_REVISION)
    args = parser.parse_args()
    result = convert(args.source, args.destination, args.model_id, args.model_revision)
    print(f"Verified {result['source_tensors']} tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
