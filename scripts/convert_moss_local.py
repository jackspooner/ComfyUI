#!/usr/bin/env python3
"""Package the pinned MOSS Local Transformer BF16 checkpoint without executing it."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from scripts.convert_moss_audio import inspect_shards, read_json, stream_checkpoint
from scripts.convert_moss_speech import LANGUAGE, SPEECH_32, SPECIAL_TOKENS, TOKENIZER_FILES, verify_sources


LOCAL = dict(additional_mlp_ffn_hidden_size=2048, local_ffn_hidden_size=8960, local_hidden_size=1536, local_num_layers=4)
LANGUAGE_V15 = dict(LANGUAGE, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                    num_attention_heads=32, vocab_size=151936, max_position_embeddings=32768)
SPECIAL_V15 = {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645,
               '<|audio_start|>': 151669, '<|audio_end|>': 151670,
               '<|vision_pad|>': 151654, '<|video_pad|>': 151656}


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


def validate_v15_config(config):
    controls = dict(model_type='moss_tts_local', n_vq=12, audio_vocab_size=1024, audio_codebook_sizes=[1024]*12,
        audio_pad_token_id=1024, audio_pad_code=1024, pad_token_id=151643, im_start_token_id=151644,
        im_end_token_id=151645, audio_start_token_id=151669, audio_end_token_id=151670,
        audio_user_slot_token_id=151654, audio_assistant_slot_token_id=151656,
        audio_assistant_gen_slot_token_id=151656, sampling_rate=48000, local_transformer_layers=1,
        local_text_head_mode='binary', use_static_local_kv_cache=True,
        audio_tokenizer_name_or_path='OpenMOSS-Team/MOSS-Audio-Tokenizer-v2')
    local = dict(n_embd=2560, n_inner=9728, n_head=32, n_layer=1, activation_function='silu',
        layer_norm_epsilon=1e-6, position_embedding_type='rope', rope_base=1000000.,
        scale_attn_weights=True, scale_attn_by_inverse_layer_idx=False,
        attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)
    for actual, expected in ((config, controls), (config['language_config'], LANGUAGE_V15),
            (config['qwen3_config'], LANGUAGE_V15), (config['gpt2_config'], local)):
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                raise ValueError(f'Unsupported MOSS Local v1.5 configuration: {key}={actual.get(key)!r}; expected {value!r}')
    for name in ('language_config', 'qwen3_config'):
        if config[name]['layer_types'] != ['full_attention']*36:
            raise ValueError('Unsupported MOSS Local v1.5 attention layer types')


def expected_v15_shapes():
    hidden, inner = 2560, 9728
    shapes = {'transformer.embed_tokens.weight': [151936, hidden], 'transformer.norm.weight': [hidden],
              'text_lm_head.weight': [151936, hidden], 'local_text_lm_head.weight': [2, hidden]}
    for q in range(12):
        shapes[f'audio_embeddings.{q}.weight'] = [1024, hidden]
        shapes[f'audio_lm_heads.{q}.weight'] = [1024, hidden]
    for i in range(36):
        prefix = f'transformer.layers.{i}'
        for name, width in [('q_proj', 4096), ('k_proj', 1024), ('v_proj', 1024)]:
            shapes[f'{prefix}.self_attn.{name}.weight'] = [width, hidden]
        shapes[f'{prefix}.self_attn.o_proj.weight'] = [hidden, 4096]
        for name in ('q_norm', 'k_norm'):
            shapes[f'{prefix}.self_attn.{name}.weight'] = [128]
        for name in ('gate_proj', 'up_proj'):
            shapes[f'{prefix}.mlp.{name}.weight'] = [inner, hidden]
        shapes[f'{prefix}.mlp.down_proj.weight'] = [hidden, inner]
        for name in ('input_layernorm', 'post_attention_layernorm'):
            shapes[f'{prefix}.{name}.weight'] = [hidden]
    prefix = 'local_transformer.h.0'
    for name in ('ln_1', 'ln_2'):
        shapes[f'{prefix}.{name}.weight'] = [hidden]
        shapes[f'{prefix}.{name}.bias'] = [hidden]
    for name, inputs, outputs in [('attn.c_attn', hidden, 3*hidden), ('attn.c_proj', hidden, hidden),
                                  ('mlp.fc_in', hidden, inner), ('mlp.fc_out', inner, hidden)]:
        shapes[f'{prefix}.{name}.weight'] = [outputs, inputs]
        shapes[f'{prefix}.{name}.bias'] = [outputs]
    shapes['local_transformer.ln_f.weight'] = [hidden]
    shapes['local_transformer.ln_f.bias'] = [hidden]
    return shapes


def mapped_v15_name(key):
    return 'model.'+key.removeprefix('transformer.') if key.startswith('transformer.') else key


def convert(source, destination, variant='original'):
    if variant not in ('original', 'v1.5'):
        raise ValueError('Unsupported MOSS Local variant; select original or v1.5')
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name('moss_speech_local_v15.json' if variant == 'v1.5' else 'moss_speech_local.json'))
    snapshots = verify_sources(source, provenance)
    validator, shapes = (validate_v15_config, expected_v15_shapes) if variant == 'v1.5' else (validate_config, expected_shapes)
    validator(read_json(source / 'config.json'))
    records = inspect_shards(source, shapes())
    if {path for path, _, _ in records.values()} != {source / name for name in provenance['files'] if name.endswith('.safetensors')}:
        raise ValueError('Weight files differ from the pinned MOSS Local inventory')
    tokenizer_json = (source / 'tokenizer.json').read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_json.decode())
    for token, token_id in (SPECIAL_V15 if variant == 'v1.5' else SPECIAL_TOKENS).items():
        if tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f'Unexpected special token ID: {token}')
    assets = {name: (source / name).read_text() for name in TOKENIZER_FILES}
    if 'generation_config.json' in provenance['files']:
        assets['generation_config.json'] = (source / 'generation_config.json').read_text()
    metadata = dict(format='pt', architecture='moss_tts_local_v15' if variant == 'v1.5' else 'moss_tts_local', conversion_version='1',
        model_id=provenance['model_id'], model_revision=provenance['revision'],
        source_manifest=json.dumps(provenance, separators=(',', ':')), source_config=(source / 'config.json').read_text(),
        tokenizer_assets=json.dumps(assets, ensure_ascii=False, separators=(',', ':')))
    if variant == 'original':
        metadata['unused_tensors'] = 'unused_global_embedding.weight'
    else:
        metadata['tied_weights'] = json.dumps({'text_lm_head.weight': 'model.embed_tokens.weight',
            **{f'audio_lm_heads.{q}.weight': f'audio_embeddings.{q}.weight' for q in range(12)}}, separators=(',', ':'))
    mapping = mapped_v15_name if variant == 'v1.5' else mapped_name
    return stream_checkpoint(records, tokenizer_json, destination, metadata, source_stats=snapshots,
                             name_mapping={key: mapping(key) for key in records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variant', choices=('original', 'v1.5'), default='original')
    args = parser.parse_args()
    result = convert(args.source, args.destination, args.variant)
    print(f"Verified {result['source_tensors']} tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
