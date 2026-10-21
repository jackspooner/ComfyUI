#!/usr/bin/env python3
"""Prepare pinned MOSS SoundEffect v2 weights offline without executing repository code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.convert_moss_audio import file_identity, inspect_shards, read_json, stream_checkpoint
from scripts.convert_moss_speech import verify_sources


def transformer_key(key):
    replacements = {
        'condition_embedder.text_embedder.linear_1': 'text_embedding.0',
        'condition_embedder.text_embedder.linear_2': 'text_embedding.2',
        'condition_embedder.time_embedder.linear_1': 'time_embedding.0',
        'condition_embedder.time_embedder.linear_2': 'time_embedding.2',
        'condition_embedder.time_proj': 'time_projection.1',
        'proj_out': 'head.head',
        'scale_shift_table': 'head.modulation',
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + '.'):
            return target + key[len(source):]
    if key.startswith('blocks.'):
        for source, target in (('.attn1.', '.self_attn.'), ('.attn2.', '.cross_attn.'),
                               ('.to_q.', '.q.'), ('.to_k.', '.k.'), ('.to_v.', '.v.'),
                               ('.to_out.0.', '.o.'), ('.ffn.net.0.proj.', '.ffn.0.'),
                               ('.ffn.net.2.', '.ffn.2.'), ('.norm2.', '.norm3.'),
                               ('.scale_shift_table', '.modulation')):
            key = key.replace(source, target)
    return key


def fold_dac_weights(state):
    folded, transforms = {}, {}
    for key, tensor in state.items():
        if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
            raise ValueError(f'Expected dense FP32 DAC tensor: {key}')
        if key.endswith('.weight_g'):
            direction_key = key.removesuffix('.weight_g') + '.weight_v'
            direction = state[direction_key]
            destination = key.removesuffix('.weight_g') + '.weight'
            folded[destination] = torch._weight_norm(direction, tensor, 0)
            transforms[destination] = dict(operation='torch._weight_norm(direction, magnitude, dim=0)', sources=[
                dict(key=name, shape=list(value.shape), sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
                for name, value in ((key, tensor), (direction_key, direction))])
        elif key.endswith('.weight_v'):
            if key.removesuffix('.weight_v') + '.weight_g' not in state:
                raise ValueError(f'Missing weight magnitude: {key}')
        else:
            folded[key] = tensor
    return folded, transforms


def convert_vae(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name('moss_soundeffect_v2.json'))
    vae_provenance = dict(provenance, files={name.removeprefix('vae/'): entry for name, entry in provenance['files'].items() if name.startswith('vae/')})
    snapshots = verify_sources(source / 'vae', vae_provenance)
    checkpoint = torch.load(source / 'vae/vae_128d_48k.pth', map_location='cpu', weights_only=True, mmap=True)
    config = checkpoint['metadata']['kwargs']
    expected = dict(encoder_dim=128, encoder_rates=[2, 3, 4, 5, 8], latent_dim=128, decoder_dim=2048,
                    decoder_rates=[8, 5, 4, 3, 2], sample_rate=48000, continuous=True)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported SoundEffect DAC configuration')
    state = checkpoint['state_dict']
    folded, transforms = fold_dac_weights(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.moss-soundeffect-') as temporary:
        temporary = Path(temporary)
        staged = temporary / 'model.safetensors'
        save_file(folded, staged)
        snapshots[staged] = file_identity(staged)
        records = inspect_shards(temporary)
        metadata = dict(format='pt', architecture='moss_soundeffect_vae', conversion_version='1',
            model_id=provenance['model_id'], model_revision=provenance['revision'],
            source_manifest=json.dumps(vae_provenance, separators=(',', ':')),
            source_config=json.dumps(config, separators=(',', ':')), source_tensor_count=str(len(state)),
            weight_norm_transforms=json.dumps(transforms, separators=(',', ':')))
        return stream_checkpoint(records, None, destination, metadata, source_stats=snapshots,
                                 name_mapping={key: key for key in records})


def convert_component(source, destination, component):
    if component == 'vae':
        return convert_vae(source, destination)
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name('moss_soundeffect_v2.json'))
    components = [component, 'tokenizer'] if component == 'text_encoder' else [component]
    snapshots = {}
    for folder in components:
        manifest = dict(provenance, files={name.removeprefix(folder + '/'): entry for name, entry in provenance['files'].items() if name.startswith(folder + '/')})
        snapshots.update(verify_sources(source / folder, manifest))
    tokenizer_json = None
    if component == 'text_encoder':
        records = inspect_shards(source / component)
        tokenizer_json = (source / 'tokenizer/tokenizer.json').read_bytes()
        expected_dtype = 'BF16'
    else:
        path = source / 'transformer/diffusion_pytorch_model.safetensors'
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        with safe_open(path, framework='pt', device='cpu') as weights:
            if set(weights.keys()) != set(header) - {'__metadata__'}:
                raise ValueError('Invalid SoundEffect transformer inventory')
        records = {key: (path, size + 8, entry) for key, entry in header.items() if key != '__metadata__'}
        expected_dtype = 'F32'
    if any(entry['dtype'] != expected_dtype for _, _, entry in records.values()):
        raise ValueError(f'Unexpected SoundEffect {component} weight dtype')
    metadata = dict(format='pt', architecture='moss_soundeffect_' + component, conversion_version='1',
                    model_id=provenance['model_id'], model_revision=provenance['revision'],
                    source_manifest=json.dumps(provenance, separators=(',', ':')),
                    source_config=(source / component / 'config.json').read_text())
    if component == 'text_encoder':
        metadata['unused_tensors'] = 'lm_head.weight'
    return stream_checkpoint(records, tokenizer_json, destination, metadata, source_stats=snapshots,
                             name_mapping={key: transformer_key(key) if component == 'transformer' else key for key in records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--component', choices=('vae', 'transformer', 'text_encoder'), required=True)
    args = parser.parse_args()
    result = convert_component(args.source, args.destination, args.component)
    print(f"Verified {result['source_tensors']} inference tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
