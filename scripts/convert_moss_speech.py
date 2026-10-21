#!/usr/bin/env python3
"""Convert pinned MOSS delay-family BF16 bundles offline, preserving every weight."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from scripts.convert_moss_audio import file_identity, inspect_shards, read_json, stream_checkpoint


LANGUAGE = dict(model_type='qwen3', vocab_size=155648, hidden_size=2048, intermediate_size=6144,
                num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                rms_norm_eps=1e-6, rope_theta=1000000, max_position_embeddings=40960,
                hidden_act='silu', attention_bias=False, use_sliding_window=False, rope_scaling=None)
SPEECH = dict(model_type='moss_tts_delay', n_vq=16, audio_ch0_vocab_size=1024, audio_vocab_size=1024,
              audio_pad_code=1024, sampling_rate=24000, audio_start_token_id=151652,
              audio_end_token_id=151653, audio_user_slot_token_id=151654,
              audio_assistant_gen_slot_token_id=151656, audio_assistant_delay_slot_token_id=151662)
SPECIAL_TOKENS = {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645,
                  '<|audio_start|>': 151652, '<|audio_end|>': 151653,
                  '<|audio_user_slot|>': 151654, '<|audio_assistant_gen_slot|>': 151656,
                  '<|audio_assistant_delay_slot|>': 151662}
TOKENIZER_FILES = ('added_tokens.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'processor_config.json', 'chat_template.jinja')
LANGUAGE_8B = dict(LANGUAGE, hidden_size=4096, intermediate_size=12288,
                   num_hidden_layers=36, num_attention_heads=32)
SPEECH_32 = {key: value for key, value in dict(SPEECH, n_vq=32).items() if key != 'audio_ch0_vocab_size'}
ACTING_LANGUAGE = {key: value for key, value in LANGUAGE_8B.items() if key not in ('rope_theta', 'rope_scaling')}
ACTING_LANGUAGE['rope_parameters'] = {'rope_theta': 1000000, 'rope_type': 'default'}
VARIANTS = {'voicegenerator': (LANGUAGE, SPEECH), 'delay': (LANGUAGE_8B, SPEECH_32),
            'acting': (ACTING_LANGUAGE, SPEECH_32), 'ttsd': (LANGUAGE_8B, dict(SPEECH_32, n_vq=16))}


def validate_config(config, variant='voicegenerator'):
    language, speech = VARIANTS[variant]
    for actual, required in ((config, speech), (config['language_config'], language)):
        for key, value in required.items():
            if key not in actual or actual[key] != value:
                raise ValueError(f'Unsupported MOSS {variant} configuration: {key}={actual.get(key)!r}; expected {value!r}')
    if config['language_config']['layer_types'] != ['full_attention'] * language['num_hidden_layers']:
        raise ValueError(f'Unsupported MOSS {variant} attention layer types')


def expected_shapes(variant='voicegenerator'):
    language, speech = VARIANTS[variant]
    hidden = language['hidden_size']
    inner = language['intermediate_size']
    q_width = language['num_attention_heads'] * language['head_dim']
    kv_width = language['num_key_value_heads'] * language['head_dim']
    shapes = {'language_model.embed_tokens.weight': [language['vocab_size'], hidden],
              'language_model.norm.weight': [hidden], 'lm_heads.0.weight': [language['vocab_size'], hidden]}
    for i in range(speech['n_vq']):
        shapes[f'emb_ext.{i}.weight'] = [speech['audio_vocab_size'] + 1, hidden]
        shapes[f'lm_heads.{i + 1}.weight'] = [speech['audio_vocab_size'] + 1, hidden]
    for i in range(language['num_hidden_layers']):
        prefix = f'language_model.layers.{i}'
        for name, size in [('q_proj', q_width), ('k_proj', kv_width), ('v_proj', kv_width)]:
            shapes[f'{prefix}.self_attn.{name}.weight'] = [size, hidden]
        shapes[f'{prefix}.self_attn.o_proj.weight'] = [hidden, q_width]
        for name in ('q_norm', 'k_norm'):
            shapes[f'{prefix}.self_attn.{name}.weight'] = [language['head_dim']]
        for name in ('gate_proj', 'up_proj'):
            shapes[f'{prefix}.mlp.{name}.weight'] = [inner, hidden]
        shapes[f'{prefix}.mlp.down_proj.weight'] = [hidden, inner]
        for name in ('input_layernorm', 'post_attention_layernorm'):
            shapes[f'{prefix}.{name}.weight'] = [hidden]
    return shapes


def verify_sources(source, manifest):
    paths = {source / name: entry for name, entry in manifest['files'].items()}
    for path in paths:
        if path.resolve().parent != source.resolve():
            raise ValueError(f'Source must be directly inside the bundle: {path.name}')
    snapshots = {path: file_identity(path) for path in paths}
    for path, entry in paths.items():
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Source does not match pinned {manifest["revision"]}: {path.name}')
        if file_identity(path) != snapshots[path]:
            raise ValueError(f'Source changed during verification: {path.name}')
    return snapshots


def convert(source, destination, variant='voicegenerator'):
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name(f'moss_speech_{variant}.json'))
    snapshots = verify_sources(source, provenance)
    config = read_json(source / 'config.json')
    validate_config(config, variant)
    records = inspect_shards(source, expected_shapes(variant))
    if {path for path, _, _ in records.values()} != {source / name for name in provenance['files'] if name.endswith('.safetensors')}:
        raise ValueError(f'Weight files differ from the pinned MOSS {variant} inventory')
    tokenizer_json = (source / 'tokenizer.json').read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_json.decode('utf-8'))
    for token, token_id in SPECIAL_TOKENS.items():
        if tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f'Unexpected special token ID: {token}')
    assets = {name: (source / name).read_text(encoding='utf-8') for name in TOKENIZER_FILES}
    if 'generation_config.json' in provenance['files']:
        assets['generation_config.json'] = (source / 'generation_config.json').read_text(encoding='utf-8')
    metadata = dict(format='pt', architecture='moss_tts_delay', variant=variant, conversion_version='1',
                    model_id=provenance['model_id'], model_revision=provenance['revision'],
                    source_manifest=json.dumps(provenance, separators=(',', ':')),
                    source_config=(source / 'config.json').read_text(encoding='utf-8'),
                    tokenizer_assets=json.dumps(assets, ensure_ascii=False, separators=(',', ':')))
    return stream_checkpoint(records, tokenizer_json, destination, metadata, source_stats=snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='voicegenerator')
    args = parser.parse_args()
    result = convert(args.source, args.destination, args.variant)
    print(f"Verified {result['source_tensors']} tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
