#!/usr/bin/env python3
"""Package the pinned MOSS Local Transformer BF16 checkpoint without executing it."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from scripts.convert_moss_audio import inspect_shards, read_json, stream_checkpoint
from scripts.convert_moss_speech import LANGUAGE, SPEECH_32, SPECIAL_TOKENS, TOKENIZER_FILES, verify_sources


LOCAL = dict(additional_mlp_ffn_hidden_size=2048, local_ffn_hidden_size=8960, local_hidden_size=1536, local_num_layers=4)


def validate_config(config):
    for actual, expected in ((config, dict(SPEECH_32, **LOCAL)), (config['language_config'], LANGUAGE)):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f'Unsupported MOSS Local configuration: {key}={actual.get(key)!r}; expected {value!r}')
    if config['language_config']['layer_types'] != ['full_attention'] * 28:
        raise ValueError('Unsupported MOSS Local attention layer types')


def expected_shapes():
    shapes = {'model.language_model.embed_tokens.weight': [155648, 2048],
              'model.embedding_list.0.weight': [155648, 2048], 'lm_heads.0.weight': [155648, 2048]}
    for q in range(33):
        if q:
            shapes[f'model.embedding_list.{q}.weight'] = [1025, 2048]
            shapes[f'lm_heads.{q}.weight'] = [1025, 2048]
        shapes[f'layer_norm_before_lm_heads.{q}.weight'] = [2048]
        for name in ('gate_proj', 'up_proj'):
            shapes[f'local_to_speech_embedding_mlps.{q}.{name}.weight'] = [2048, 1536]
        shapes[f'local_to_speech_embedding_mlps.{q}.down_proj.weight'] = [2048, 2048]
    for name in ('gate_proj', 'up_proj'):
        shapes[f'speech_embedding_to_local_mlp.{name}.weight'] = [2048, 2048]
    shapes['speech_embedding_to_local_mlp.down_proj.weight'] = [1536, 2048]
    for root, hidden, inner, layers in [('model.language_model', 2048, 6144, 28), ('local_transformer', 1536, 8960, 4)]:
        shapes[f'{root}.norm.weight'] = [hidden]
        for i in range(layers):
            prefix = f'{root}.layers.{i}'
            for name, width in [('q_proj', 2048), ('k_proj', 1024), ('v_proj', 1024)]:
                shapes[f'{prefix}.self_attn.{name}.weight'] = [width, hidden]
            shapes[f'{prefix}.self_attn.o_proj.weight'] = [hidden, 2048]
            for name in ('q_norm', 'k_norm'):
                shapes[f'{prefix}.self_attn.{name}.weight'] = [128]
            for name in ('gate_proj', 'up_proj'):
                shapes[f'{prefix}.mlp.{name}.weight'] = [inner, hidden]
            shapes[f'{prefix}.mlp.down_proj.weight'] = [hidden, inner]
            for name in ('input_layernorm', 'post_attention_layernorm'):
                shapes[f'{prefix}.{name}.weight'] = [hidden]
    return shapes


def mapped_name(key):
    if key == 'model.language_model.embed_tokens.weight':
        return 'unused_global_embedding.weight'
    if key.startswith('model.language_model.'):
        return 'model.' + key.removeprefix('model.language_model.')
    if key.startswith('model.embedding_list.'):
        channel = int(key.split('.')[2])
        return 'model.embed_tokens.weight' if channel == 0 else f'emb_ext.{channel - 1}.weight'
    return key


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name('moss_speech_local.json'))
    snapshots = verify_sources(source, provenance)
    validate_config(read_json(source / 'config.json'))
    records = inspect_shards(source, expected_shapes())
    if {path for path, _, _ in records.values()} != {source / name for name in provenance['files'] if name.endswith('.safetensors')}:
        raise ValueError('Weight files differ from the pinned MOSS Local inventory')
    tokenizer_json = (source / 'tokenizer.json').read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_json.decode())
    for token, token_id in SPECIAL_TOKENS.items():
        if tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f'Unexpected special token ID: {token}')
    assets = {name: (source / name).read_text() for name in TOKENIZER_FILES}
    if 'generation_config.json' in provenance['files']:
        assets['generation_config.json'] = (source / 'generation_config.json').read_text()
    metadata = dict(format='pt', architecture='moss_tts_local', conversion_version='1',
        model_id=provenance['model_id'], model_revision=provenance['revision'],
        source_manifest=json.dumps(provenance, separators=(',', ':')), source_config=(source / 'config.json').read_text(),
        tokenizer_assets=json.dumps(assets, ensure_ascii=False, separators=(',', ':')),
        unused_tensors='unused_global_embedding.weight')
    return stream_checkpoint(records, tokenizer_json, destination, metadata, source_stats=snapshots,
                             name_mapping={key: mapped_name(key) for key in records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.source, args.destination)
    print(f"Verified {result['source_tensors']} tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
