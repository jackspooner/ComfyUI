"""Offline FP8 -> native NVFP4 Drama 3 packaging; no downloads or inference."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.cpu = True
from comfy.quant_ops import TensorCoreNVFP4Layout
from comfy.fish_audio.config import FishOmniConfig
from comfy.fish_audio.runtime import GenerationModel
from comfy.fish_audio.codec import Codec
import comfy.ops
from scripts.convert_moss_audio import read_json

SOURCE_REVISION = 'b2b5ee998034b2975720906c815e7fb0e9f007bd'
UPSTREAM_REVISION = '541566932abbaebcb0a40e562a1f846738383f38'
class InventoryOperations(comfy.ops.manual_cast):
    # MixedPrecision.Linear allocates weights only at load time; use ordinary
    # meta Linear placeholders to validate every non-expert checkpoint tensor.
    MoEExperts = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).MoEExperts


EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$')
TOWER_PREFIXES = {'audio_decoder.codebook_embeddings.': 'tower.codebook_embeddings.',
                  'audio_decoder_mtp_shared.layers.': 'tower.shared_layers.',
                  'audio_decoder_mtp.layers.': 'tower.residual_layers.',
                  'audio_decoder_mtp.heads.': 'tower.codebook_heads.'}
DTYPES = {torch.bfloat16: 'BF16', torch.float32: 'F32', torch.uint8: 'U8', torch.float8_e4m3fn: 'F8_E4M3'}
NOTICES = Path(__file__).resolve().parents[1] / 'comfy/fish_audio'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def mapped_name(key):
    if key.startswith('model.language_model.'):
        return 'backbone.' + key.removeprefix('model.language_model.')
    if key == 'lm_head.weight':
        return key
    for old, new in TOWER_PREFIXES.items():
        if key.startswith(old):
            return new + key[len(old):]
    raise ValueError(f'Unknown Drama 3 tensor: {key}')


def decode_fp8(weight, scales):
    rows, cols = weight.shape
    if weight.dtype != torch.float8_e4m3fn or rows % 128 or cols % 128 or scales.shape != (rows // 128, cols // 128) or not torch.isfinite(scales).all():
        raise ValueError('Invalid block-FP8 expert weight or scales')
    out = (weight.view(rows // 128, 128, cols // 128, 128).float() * scales[:, None, :, None]).reshape(rows, cols)
    if not torch.isfinite(out).all():
        raise ValueError('Non-finite decoded expert')
    return out.to(torch.bfloat16)


def verify_source(directory, codec_source):
    lock = read_json(Path(__file__).with_name('drama3-source.json'))
    if lock['revision'] != SOURCE_REVISION:
        raise ValueError('Source pin and converter revision differ')
    receipts = {}
    for name, record in lock['files'].items():
        path = Path(codec_source) if name == 'codec.pth' else Path(directory) / name
        if path.stat().st_size != record['size']:
            raise ValueError(f'Incomplete source file: {name}')
        digest = sha256(path)
        if 'lfs' in record:
            matches = digest == record['lfs']['sha256']
        else:
            blob = b'blob ' + str(record['size']).encode() + b'\0' + path.read_bytes()
            matches = hashlib.sha1(blob).hexdigest() == record['blobId']
        if not matches:
            raise ValueError(f'Published source hash mismatch: {name}')
        receipts[name] = digest
        print('Verified source', name, flush=True)
    return receipts


class Source:
    def __init__(self, directory, stack):
        self.directory = Path(directory)
        index = read_json(self.directory / 'model.safetensors.index.json')
        self.mapping = index['weight_map']
        self.files, self.specs = {}, {}
        for name in sorted(set(self.mapping.values())):
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError('Source shard escapes checkpoint directory')
            saved = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
            self.files[name] = saved
            with path.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                if length > 100 * 1024 ** 2:
                    raise ValueError('Invalid shard header')
                header = json.loads(f.read(length))
            for key in saved.keys():
                if self.mapping.get(key) != name or key in self.specs:
                    raise ValueError(f'Shard/index tensor mismatch: {key}')
                self.specs[key] = header[key]
        if set(self.specs) != set(self.mapping):
            raise ValueError('Incomplete source shard inventory')
        declared_size = index.get('metadata', {}).get('total_size')
        actual_size = sum(s['data_offsets'][1] - s['data_offsets'][0] for s in self.specs.values())
        if declared_size is not None and actual_size != declared_size:
            raise ValueError('Shard payload size differs from the checkpoint index')

    def get(self, key):
        return self.files[self.mapping[key]].get_tensor(key)


class Writer:
    def __init__(self, path, specs, metadata):
        self.path = Path(path)
        if self.path.exists() or self.path.with_suffix('.manifest.json').exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = {'__metadata__': metadata}
        offset = 0
        for name, (shape, dtype) in specs.items():
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            self.header[name] = dict(shape=list(shape), dtype=DTYPES[dtype], data_offsets=[offset, offset + size])
            offset += size
        encoded = json.dumps(self.header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        self.base = 8 + len(encoded)
        self.file = tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.' + self.path.name + '.', delete=False)
        self.temporary = Path(self.file.name)
        self.file.write(struct.pack('<Q', len(encoded)) + encoded)
        self.file.truncate(self.base + offset)
        self.written = {}
        self.digest = {}

    def put(self, name, tensor, index=None):
        spec = self.header[name]
        data = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        start, end = spec['data_offsets']
        if index is not None:
            width = (end - start) // spec['shape'][0]
            if len(data) != width or not 0 <= index < spec['shape'][0]:
                raise ValueError(f'Wrong expert payload: {name}')
            start += index * width
        elif len(data) != end - start:
            raise ValueError(f'Wrong tensor payload: {name}')
        identity = (name, index)
        if identity in self.written:
            raise ValueError(f'Duplicate output payload: {identity}')
        self.file.seek(self.base + start)
        self.file.write(data)
        self.written[identity] = (start, len(data), hashlib.sha256(data).hexdigest())

    def finish(self, manifest):
        for name, spec in self.header.items():
            if name == '__metadata__':
                continue
            required = spec['data_offsets'][1] - spec['data_offsets'][0]
            if sum(size for (n, _), (_, size, _) in self.written.items() if n == name) != required:
                raise ValueError(f'Output not completely written: {name}')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        with self.temporary.open('rb') as f:
            for (name, index), (start, size, digest) in self.written.items():
                f.seek(self.base + start)
                if hashlib.sha256(f.read(size)).hexdigest() != digest:
                    raise ValueError(f'Output payload verification failed: {name}/{index}')
        with safe_open(str(self.temporary), framework='pt', device='cpu') as saved:
            if set(saved.keys()) != set(self.header) - {'__metadata__'}:
                raise ValueError('Output inventory mismatch')
        manifest['tensor_names'] = sorted(set(self.header) - {'__metadata__'})
        manifest['output_sha256'] = sha256(self.temporary)
        manifest['output_size'] = self.temporary.stat().st_size
        receipt = tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.' + self.path.name + '.manifest.', mode='w', delete=False)
        try:
            with receipt as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.link(self.temporary, self.path)
            os.link(receipt.name, self.path.with_suffix('.manifest.json'))
            self.temporary.unlink()
        finally:
            Path(receipt.name).unlink(missing_ok=True)
        return manifest

    def abort(self):
        if not self.file.closed:
            self.file.close()
        self.temporary.unlink(missing_ok=True)


def codec_weights(source):
    state = torch.load(str(source), map_location='cpu', mmap=True, weights_only=True)
    state = state.get('state_dict', state)
    if any(k.startswith('generator.') for k in state):
        state = {k.removeprefix('generator.'): v for k, v in state.items() if k.startswith('generator.')}
    result, disposition = {}, {}
    for key, value in state.items():
        if key.endswith(('.freqs_cis', '.causal_mask')):
            disposition[key] = 'recomputed per execution'
        elif key.endswith(('.weight_g', '.parametrizations.weight.original0')):
            stem = key.removesuffix('.weight_g').removesuffix('.parametrizations.weight.original0')
            vkey = stem + ('.weight_v' if key.endswith('.weight_g') else '.parametrizations.weight.original1')
            result[stem + '.weight'] = torch._weight_norm(state[vkey].float(), value.float(), 0).to(torch.bfloat16)
            disposition[key] = disposition[vkey] = stem + '.weight'
        elif key.endswith(('.weight_v', '.parametrizations.weight.original1')):
            continue
        else:
            result[key] = value.to(torch.bfloat16).contiguous()
            disposition[key] = key
    with torch.device('meta'):
        expected = Codec(torch.bfloat16, torch.device('meta'), comfy.ops.manual_cast).state_dict()
    if set(result) != set(expected):
        raise ValueError(f'Codec tensor mismatch: missing {set(expected)-set(result)}, extra {set(result)-set(expected)}')
    for key, value in result.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid codec tensor: {key}')
    return result, disposition


def convert_codec(source, destination):
    state, disposition = codec_weights(source)
    source_hash = sha256(source)
    metadata = dict(format='fish_dac', package_version='1', source_hash=source_hash, source_revision=SOURCE_REVISION,
                    license=(NOTICES / 'LICENSE').read_text(), notice=(NOTICES / 'NOTICE').read_text(), dac_license=(NOTICES / 'DAC-LICENSE').read_text())
    writer = Writer(destination, {k: (v.shape, v.dtype) for k, v in state.items()}, metadata)
    try:
        for key, value in state.items():
            writer.put(key, value)
        writer.finish(dict(source_hash=source_hash, disposition=disposition, metadata=metadata))
    finally:
        writer.abort()
    return source_hash


def convert(source, destination, codec_hash, verified_hashes=None):
    source = Path(source)
    raw_config = read_json(source / 'config.json')
    config = FishOmniConfig.from_dict(raw_config)
    tokenizer = (source / 'tokenizer.json').read_text()
    with torch.device('meta'):
        model = GenerationModel(config, torch.bfloat16, torch.device('meta'), InventoryOperations)
    expected = {k: tuple(v.shape) for k, v in model.named_parameters()}
    specs, copies, consumed = {}, {}, {}
    banks = []
    with ExitStack() as stack:
        tensors = Source(source, stack)
        for key, spec in tensors.specs.items():
            if EXPERT.match(key):
                continue
            name = mapped_name(key)
            if name not in expected or tuple(spec['shape']) != expected[name] or spec['dtype'] != 'BF16':
                raise ValueError(f'Unexpected dense source tensor: {key}')
            copies[name] = key
            specs[name] = (spec['shape'], torch.bfloat16)
            consumed[key] = name
        if set(copies) != set(expected):
            raise ValueError(f'Missing dense weights: {set(expected) - set(copies)}')
        c = config.text
        for layer in range(c.num_hidden_layers):
            for bank, projections, rows, cols in [('gate_up_proj', ('gate_proj', 'up_proj'), 2 * c.moe_intermediate_size, c.hidden_size),
                                                   ('down_proj', ('down_proj',), c.hidden_size, c.moe_intermediate_size)]:
                prefix = f'backbone.layers.{layer}.mlp.experts.{bank}'
                specs[prefix + '.weight'] = ((c.num_experts, rows, cols // 2), torch.uint8)
                specs[prefix + '.weight_scale'] = ((c.num_experts, (rows + 127) // 128 * 128, (cols // 16 + 3) // 4 * 4), torch.float8_e4m3fn)
                specs[prefix + '.weight_scale_2'] = ((c.num_experts,), torch.float32)
                descriptor = torch.tensor(list(json.dumps(dict(format='nvfp4', num_experts=c.num_experts)).encode()), dtype=torch.uint8)
                specs[prefix + '.comfy_quant'] = (descriptor.shape, torch.uint8)
                banks.append((prefix, layer, projections, descriptor))
                for expert in range(c.num_experts):
                    for projection in projections:
                        base = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}'
                        for suffix in ('weight', 'weight_scale_inv'):
                            key = base + '.' + suffix
                            if key not in tensors.specs:
                                raise ValueError(f'Missing expert tensor: {key}')
                            consumed[key] = prefix
        if set(consumed) != set(tensors.specs):
            raise ValueError('Source tensor accounting is incomplete')
        metadata = dict(format='drama3_nvfp4', package_version='1', source_revision=SOURCE_REVISION, upstream_revision=UPSTREAM_REVISION,
                        config=json.dumps(raw_config), tokenizer=tokenizer, codec_hash=codec_hash,
                        license=(source / 'LICENSE').read_text(), notice=(NOTICES / 'NOTICE').read_text())
        writer = Writer(destination, specs, metadata)
        errors = {}
        try:
            for name, key in copies.items():
                value = tensors.get(key)
                if not torch.isfinite(value).all():
                    raise ValueError(f'Non-finite source tensor: {key}')
                writer.put(name, value)
            for prefix, layer, projections, descriptor in banks:
                writer.put(prefix + '.comfy_quant', descriptor)
                squared_error = squared_signal = 0.
                peak = 0.
                for expert in range(c.num_experts):
                    values = []
                    for projection in projections:
                        base = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}'
                        values.append(decode_fp8(tensors.get(base + '.weight'), tensors.get(base + '.weight_scale_inv')))
                    value = torch.cat(values) if len(values) > 1 else values[0]
                    packed, params = TensorCoreNVFP4Layout.quantize(value)
                    restored = TensorCoreNVFP4Layout.dequantize(packed, params)
                    difference = restored.float() - value.float()
                    squared_error += difference.square().sum().item()
                    squared_signal += value.float().square().sum().item()
                    peak = max(peak, difference.abs().max().item())
                    writer.put(prefix + '.weight', packed, expert)
                    writer.put(prefix + '.weight_scale', params.block_scale, expert)
                    writer.put(prefix + '.weight_scale_2', params.scale.float().reshape(1), expert)
                errors[prefix] = dict(relative_rmse=math.sqrt(squared_error / max(squared_signal, 1e-30)), max_abs=peak)
                print(prefix, errors[prefix], flush=True)
            return writer.finish(dict(source_revision=SOURCE_REVISION, upstream_revision=UPSTREAM_REVISION,
                source_hashes=verified_hashes if verified_hashes is not None else {name: sha256(source / name) for name in tensors.files}, mapping=consumed, quantization_errors=errors,
                codec_hash=codec_hash, precision='NVFP4 routed experts; BF16 otherwise', package_version=1,
                quantization=dict(layout='TensorCoreNVFP4Layout', block_size=16, stochastic_rounding=0,
                    scale='per-expert absmax/(448*6)', block_scale_dtype='E4M3FN', scale_dtype='FP32',
                    source_block_size=[128, 128], source_dequant_dtype='FP32 then BF16', input_passes=1)))
        finally:
            writer.abort()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--codec-output', type=Path, required=True)
    p.add_argument('--codec-source', type=Path)
    options = p.parse_args()
    if options.model_output.exists() or options.model_output.with_suffix('.manifest.json').exists():
        raise FileExistsError(options.model_output)
    codec_source = options.codec_source or options.source / 'codec.pth'
    verified_hashes = verify_source(options.source, codec_source)
    # Completed codec packages are reusable after verifying their source identity.
    if options.codec_output.exists():
        with safe_open(str(options.codec_output), framework='pt', device='cpu') as saved:
            meta = saved.metadata() or {}
        manifest = read_json(options.codec_output.with_suffix('.manifest.json'))
        codec_hash = sha256(codec_source)
        if meta.get('source_hash') != codec_hash or manifest.get('output_sha256') != sha256(options.codec_output):
            raise ValueError('Existing codec package failed recovery verification')
    else:
        codec_hash = convert_codec(codec_source, options.codec_output)
    convert(options.source, options.model_output, codec_hash, verified_hashes)


if __name__ == '__main__':
    main()
