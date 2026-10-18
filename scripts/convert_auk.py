#!/usr/bin/env python3
"""Convert complete local AuK-Flash/Qwen snapshots to native ComfyUI components."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comfy.cli_args import args as comfy_args

if __name__ == '__main__':
    comfy_args.cpu = True

from scripts.convert_moss_audio import inspect_shards, read_json
from comfy.ldm.auk.model import AuK
from comfy.ldm.auk.vae import AuKVAE
from comfy.text_encoders.auk import AuKThinker
import comfy.ops

SOURCE_REVISION = '9481474f107ee4d8dec959b3aed05b0627cebbf8'
AUK_REVISION = '575b92f0895f75180bf2cbd35f2e176c5732b8ed'
QWEN_REVISION = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
# Pinned payload SHA256 values; small files verified against revision Git blob IDs.
SOURCE_HASHES = {
    'auk_flash.safetensors': '9b3ec1400a9ebcc87ee540ebfad0aeeb08e4f210149e190b1469fd8eeb977fe4',
    'vae.safetensors': '0f5857fd0d6b916d161c73fef433e0ccb42484a2dad3aa08e8a51975e29a0645',
    'config.yaml': '65149f38c90003f4b73a6bd5ea74911a67fc1512740e402dad923d27a35e2e17',
    'model-00001-of-00003.safetensors': '349972cebff443030e9e96e1e930d7230979c6961cf046fa29fc580e4bcf49a6',
    'model-00002-of-00003.safetensors': 'b29a76dfefb3aa33a5bd0faa19c681d039a44de57223653ea0e933df728c6d8c',
    'model-00003-of-00003.safetensors': '8f2138170a87ec604b53232491a96d6dcf19e0724aca37e69aace143e508b170',
    'tokenizer.json': '8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99',
    'config.json': '20790f362c37a1718a3e764f597ae33dfc20177399762018dffaabf8321d4dd1',
    'preprocessor_config.json': 'b47055ce61463ce143e9aab741d55c0aa520801a0a5d63be73c5b17cecb6bc69',
    'tokenizer_config.json': '569aa7a9171e36dfff80f0a7550ab0c9c09e46ac840fea5030641417394fb0d2',
    'special_tokens_map.json': 'dc241604d085780a33f760f69c47e6622451104c20eee3cd3a12b6e884b43f44',
    'added_tokens.json': 'e81a2cc3bd867a1217019eed202d1d8a07e1063ece716f22060dda14f6cc07d8',
    'chat_template.json': '1933f7ca08e6503460e981d9037ddc0b851196abfa3aceea52d6ee0c0fdfd9f0',
    'model.safetensors.index.json': '5b7629198e2ef80e37612a491d9bfd71639d2f212632d36d8ab086922e74e129',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'LICENSE': 'ae8ef9d3fb476f735d9b9abeaab73b7f778b121c77427cd918ec3da01eefbc44',
}


def validate_inventory(weights, expected, component):
    missing, extra = expected.keys() - weights.keys(), weights.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f'{component} inventory mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    for key, value in weights.items():
        if value.shape != expected[key].shape or not value.is_floating_point():
            raise ValueError(f'{component} invalid tensor {key}: {tuple(value.shape)}, {value.dtype}')



def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_complete(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8 or struct.unpack('<Q', raw)[0] > 100_000_000:
            raise ValueError(f'Incomplete safetensors or Git LFS pointer: {path}')
    return load_file(path, device='cpu')


def fold_weight_norm(weights):
    result = {}
    actions = {}
    for name, value in weights.items():
        if name.startswith('flow.'):
            actions[name] = {'action': 'excluded', 'reason': 'training flow is unused by upstream encode/decode'}
        elif name.endswith('.weight_g'):
            prefix = name.removesuffix('.weight_g')
            vector_name = prefix + '.weight_v'
            if vector_name not in weights:
                raise ValueError(f'Missing weight norm vector: {vector_name}')
            vector = weights[vector_name]
            result[prefix + '.weight'] = torch._weight_norm(vector, value, 0)
            actions[name] = actions[vector_name] = {'action': 'weight_norm', 'destination': prefix + '.weight'}
        elif name.endswith('.weight_v'):
            if name.removesuffix('.weight_v') + '.weight_g' not in weights:
                raise ValueError(f'Missing weight norm magnitude: {name}')
        else:
            result[name] = value
            actions[name] = {'action': 'retained', 'destination': name}
    return result, actions


def write_component(weights, path, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as stream:
            temporary = Path(stream.name)
        save_file(weights, temporary, metadata=metadata)
        with safe_open(temporary, framework='pt', device='cpu') as output:
            if set(output.keys()) != set(weights):
                raise ValueError(f'Converted inventory differs: {path}')
            for key, value in weights.items():
                actual = output.get_tensor(key)
                if actual.dtype != value.dtype or actual.shape != value.shape or not torch.equal(actual, value):
                    raise ValueError(f'Converted payload differs: {key}')
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size, 'tensors': len(weights)}


def convert(auk, qwen, destination):
    auk, qwen, destination = Path(auk).resolve(), Path(qwen).resolve(), Path(destination).resolve()
    paths = {'model': destination / 'diffusion_models/auk_flash.safetensors',
             'clip': destination / 'text_encoders/auk_qwen25_omni_3b.safetensors',
             'vae': destination / 'vae/auk_audio_vae.safetensors'}
    manifest_path = destination / 'auk_flash.manifest.json'
    for path in [*paths.values(), manifest_path, destination / 'auk_flash_sources']:
        if path.exists():
            raise FileExistsError(f'Preserving existing output: {path}')
    source_paths = [auk / name if name in ('auk_flash.safetensors', 'vae.safetensors', 'config.yaml') else qwen / name for name in SOURCE_HASHES]
    source_metadata = {}
    for path in source_paths:
        digest = sha256(path)
        if digest != SOURCE_HASHES[path.name]:
            raise ValueError(f'Pinned source hash mismatch (incomplete or different revision): {path}')
        source_metadata[str(path)] = {'sha256':digest, 'bytes':path.stat().st_size}
    config = read_json(qwen / 'config.json')
    thinker = config['thinker_config']
    required = {'hidden_size': 2048, 'num_hidden_layers': 36, 'num_attention_heads': 16, 'num_key_value_heads': 2,
                'intermediate_size': 11008, 'vocab_size': 151936, 'rope_theta': 1000000.0, 'rms_norm_eps': 1e-6}
    for key, value in required.items():
        if thinker['text_config'][key] != value:
            raise ValueError(f'Unsupported Qwen text configuration: {key}')
    for key, value in {'d_model':1280, 'encoder_layers':32, 'encoder_attention_heads':20, 'encoder_ffn_dim':5120,
                       'num_mel_bins':128, 'output_dim':2048, 'n_window':100}.items():
        if thinker['audio_config'][key] != value:
            raise ValueError(f'Unsupported Qwen audio configuration: {key}')
    records = inspect_shards(qwen)
    auk_weights = load_complete(auk / 'auk_flash.safetensors')
    vae_weights = load_complete(auk / 'vae.safetensors')
    model, clip, actions = {}, {}, {'auk':{}, 'qwen':{}, 'vae':{}}
    for name, value in auk_weights.items():
        if name.startswith('transformer.'):
            target = name.removeprefix('transformer.')
            model[target] = value
            actions['auk'][name] = {'action':'retained', 'component':'model', 'destination':target}
        elif name in ('layer_weights', 'layer_scale'):
            clip[name] = value
            actions['auk'][name] = {'action':'retained', 'component':'clip', 'destination':name}
        else:
            raise ValueError(f'Unaccounted AuK weight: {name}')
    if clip.keys() != {'layer_weights', 'layer_scale'} or clip['layer_weights'].shape != (36,):
        raise ValueError('Missing or incompatible learned layer fusion')
    for name, (path, _, entry) in records.items():
        if name.startswith('thinker.model.') or (name.startswith('thinker.audio_tower.') and '.audio_bos_eos_token.' not in name):
            target = name.removeprefix('thinker.')
            with safe_open(path, framework='pt', device='cpu') as source:
                clip[target] = source.get_tensor(name)
            actions['qwen'][name] = {'action':'retained', 'destination':target, 'source_file':path.name}
        elif name.startswith(('thinker.visual.', 'thinker.lm_head.', 'talker.', 'token2wav.', 'thinker.audio_tower.audio_bos_eos_token.')):
            actions['qwen'][name] = {'action':'excluded', 'reason':'unused vision, logits, audio boundary embedding or speech-generation component'}
        else:
            raise ValueError(f'Unaccounted Qwen weight: {name}')
    config_files = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.json']
    embedded = {name:read_json(qwen / name) for name in config_files}
    embedded['chat_template'] = embedded['chat_template.json']['chat_template']
    vae, actions['vae'] = fold_weight_norm(vae_weights)
    with torch.device('meta'):
        expected_model = AuK(operations=comfy.ops.disable_weight_init).state_dict()
        expected_clip = AuKThinker({}, torch.float32, 'meta', comfy.ops.disable_weight_init).state_dict()
        expected_vae = AuKVAE().state_dict()
    for kind, weights, expected in [('model', model, expected_model), ('clip', clip, expected_clip), ('vae', vae, expected_vae)]:
        validate_inventory(weights, expected, kind)
    for key, data in [('tokenizer_json', (qwen / 'tokenizer.json').read_bytes()),
                      ('auk_config_json', json.dumps(embedded, ensure_ascii=False).encode())]:
        clip[key] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    metadata = {'format':'pt', 'auk_source_revision':SOURCE_REVISION, 'auk_model_revision':AUK_REVISION,
                'qwen_model_revision':QWEN_REVISION}
    manifest = {'metadata':metadata, 'sources':source_metadata,
                'actions':actions, 'components':{}}
    for kind, weights in [('model',model), ('clip',clip), ('vae',vae)]:
        manifest['components'][kind] = write_component(weights, paths[kind], {**metadata,'auk_component':kind})
    manifest['configuration'] = {name:sha256(qwen / name) for name in config_files + ['tokenizer.json', 'merges.txt', 'vocab.json', 'LICENSE']}
    provenance = destination / 'auk_flash_sources'
    provenance.mkdir()
    for label, root, names in [('auk', auk, ['LICENSE', 'README.md', 'config.yaml']),
                               ('qwen', qwen, config_files + ['LICENSE', 'README.md', 'tokenizer.json', 'merges.txt', 'vocab.json'])]:
        folder = provenance / label
        folder.mkdir()
        for name in names:
            shutil.copyfile(root / name, folder / name)
    with manifest_path.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest['components'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auk', required=True, help='Directory containing complete auk_flash.safetensors and vae.safetensors')
    parser.add_argument('--qwen', required=True, help='Complete original Qwen2.5-Omni-3B directory')
    parser.add_argument('--destination', required=True, help='Output model root; existing outputs are never overwritten')
    args = parser.parse_args()
    convert(args.auk, args.qwen, args.destination)


if __name__ == '__main__':
    main()
