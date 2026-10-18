#!/usr/bin/env python3
"""Repackage an explicitly supplied local Audio Flamingo 3 HF snapshot for Load CLIP."""
import argparse
import hashlib
from pathlib import Path

from transformers import Qwen2TokenizerFast
from convert_moss_audio import inspect_shards, read_json, stream_checkpoint


def expected_shapes():
    shapes = {}
    def linear(name, rows, columns, bias=False):
        shapes[name + '.weight'] = [rows, columns]
        if bias:
            shapes[name + '.bias'] = [rows]
    for name, channels in [('conv1',128), ('conv2',1280)]:
        shapes[f'audio_tower.{name}.weight'] = [1280, channels, 3]
        shapes[f'audio_tower.{name}.bias'] = [1280]
    shapes['audio_tower.embed_positions.weight'] = [1500,1280]
    for i in range(32):
        prefix = f'audio_tower.layers.{i}'
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            linear(prefix + '.self_attn.' + name,1280,1280,name != 'k_proj')
        linear(prefix + '.fc1',5120,1280,True)
        linear(prefix + '.fc2',1280,5120,True)
        for norm in ('self_attn_layer_norm','final_layer_norm'):
            for param in ('weight','bias'):
                shapes[f'{prefix}.{norm}.{param}'] = [1280]
    for param in ('weight','bias'):
        shapes['audio_tower.layer_norm.' + param] = [1280]
    linear('multi_modal_projector.linear_1',3584,1280,True)
    linear('multi_modal_projector.linear_2',3584,3584,True)
    linear('language_model.model.embed_tokens',151672,3584)
    linear('language_model.lm_head',151672,3584)
    shapes['language_model.model.norm.weight'] = [3584]
    for i in range(28):
        prefix = f'language_model.model.layers.{i}'
        for name, size in [('q_proj',3584),('k_proj',512),('v_proj',512),('o_proj',3584)]:
            linear(prefix + '.self_attn.' + name,size,3584,name != 'o_proj')
        for name in ('gate_proj','up_proj'):
            linear(prefix + '.mlp.' + name,18944,3584)
        linear(prefix + '.mlp.down_proj',3584,18944)
        for norm in ('input_layernorm','post_attention_layernorm'):
            shapes[f'{prefix}.{norm}.weight'] = [3584]
    return shapes


def convert(source, destination, revision):
    config = read_json(source / 'config.json')
    if config.get('model_type') != 'audioflamingo3' or config.get('audio_token_id') != 151669:
        raise ValueError('Expected the Audio Flamingo 3 HF checkpoint.')
    text = dict(hidden_size=3584,intermediate_size=18944,num_hidden_layers=28,num_attention_heads=28,
                num_key_value_heads=4,vocab_size=151672,rms_norm_eps=1e-6,max_position_embeddings=32768,
                hidden_act='silu',use_sliding_window=False,eos_token_id=151645,pad_token_id=151671)
    audio = dict(hidden_size=1280,intermediate_size=5120,num_hidden_layers=32,num_attention_heads=20,
                 num_mel_bins=128,max_source_positions=1500,activation_function='gelu',scale_embedding=False)
    for name, expected in [('text_config',text),('audio_config',audio)]:
        for key, value in expected.items():
            if config[name].get(key) != value:
                raise ValueError(f'Unsupported Audio Flamingo configuration: {name}.{key}')
    if config['text_config'].get('rope_parameters') != {'rope_theta':1000000.0,'rope_type':'default'} or config.get('projector_hidden_act') != 'gelu' or config.get('projector_bias') is not True:
        raise ValueError('Unsupported Audio Flamingo rotary or projector configuration.')
    original = inspect_shards(source, expected_shapes())
    # Normalize the Transformers language wrapper before using the shared streaming writer.
    records = {}
    for name, record in original.items():
        key = name.removeprefix('language_model.')
        if key == 'lm_head.weight':
            key = 'model.lm_head.weight'
        records[key] = record
    tokenizer = Qwen2TokenizerFast.from_pretrained(source,local_files_only=True)
    for text, token in [('<sound>',151669),('<|im_end|>',151645)]:
        if tokenizer.encode(text,add_special_tokens=False) != [token]:
            raise ValueError(f'Unexpected tokenizer ID for {text}')
    metadata = dict(format='pt',architecture='audio_flamingo',conversion_version='1',
                    model_id='nvidia/audio-flamingo-3-hf',model_revision=revision,
                    source_config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest())
    return stream_checkpoint(records,tokenizer.backend_tokenizer.to_str().encode(),destination,metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--revision',required=True)
    args = parser.parse_args()
    manifest = convert(args.source,args.destination,args.revision)
    print(f"Converted {manifest['source_tensors']} tensors; manifest: {args.destination.with_suffix('.manifest.json')}")
