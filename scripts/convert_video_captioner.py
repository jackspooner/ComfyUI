#!/usr/bin/env python3
"""Offline, lossless single-file packaging of pinned video captioners."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file
from transformers import Qwen2TokenizerFast

from convert_moss_audio import inspect_shards, read_json, stream_checkpoint


SOURCES = {
    'internvideo3_8b': {
        'repo': 'yanziang/InternVideo3-8B-Instruct',
        'revision': 'c4602918b65225650d152db2850fe34e01d21fcd',
        'code_revision': 'c4602918b65225650d152db2850fe34e01d21fcd',
        'config_blob': '3030b2f5de4f04f873dd9ef0b1b70ffa96b6117f',
        'weights': {
            'model-language-0001-others-save_rank0.safetensors': '914ec6c9f5ede61c3dd4694c6e8522b4db4e4e71b6b0675a3e49dc19d3754650',
            'model-language-0002-others-save_rank0.safetensors': '36262c85b72728b543eb0c482956ecb2d81fe3c41493531ee8e2f7434cdf0128',
            'model-language-0003-others-save_rank0.safetensors': 'a7b034b33c7704fdad971cbb80e2121857ee05664bdccfa55f39c3e83f7b04a7',
            'model-language-0004-others-save_rank0.safetensors': '4dea96bbcf19ea509c78e1ab521bc4dd9592aff5635f137f4e7ef01d43245187',
            'model-language-0005-others-save_rank0.safetensors': '2a6fdae561d4d1ebd1410a962c7f702d01b83a18a37b91089baf763281d985a7',
            'model-projector-0001-others-save_rank0.safetensors': '123db5dd1a0a768fb4e877bc80b5b8faf6e056aff6df6e4ee8235edaf92f6cc0',
            'model-vision-0001-others-save_rank0.safetensors': '166a8a730fc7b51832200f937a4d79aea2c3a43401154046e9ecb6ee0b6fef1d',
        },
    },
    'salmonn2_pro_8b': {
        'repo': 'tsinghua-ee/video-SALMONN-2-Pro-8B',
        'revision': '89319a9934169266b8d5da3d7d4850e9f714870f',
        'code_revision': '2d28e841c8378885d2f30982be801d0c44f7a5bb',
        'config_blob': '61dbb0e2bdfce75692ab5f914995b38e53ab01c2',
        'weights': {
            'model-00001-of-00004.safetensors': 'e3b39897ea7ea7d564d1a28daa1e58d4af9fe8e3677e17e5eac6b0b2b9c28547',
            'model-00002-of-00004.safetensors': '65ca5e58ecbfc0b40e77ee474cf43cddd4c7525a227c4917acafeb9580535d6e',
            'model-00003-of-00004.safetensors': 'f2f19c309d871fba610419abb04b3e921a8090aa0cdd64b4da686f958323dfaa',
            'model-00004-of-00004.safetensors': 'cbae59c4fbad9b7111f722551705cb1b5dcd80c39ab9bdd97042072cef218248',
        },
    },
    'padcaptioner_3b': {
        'repo': 'wenzhengzeng/PadCaptioner-3B',
        'revision': 'a1eb578dd619be63819d8342d8d49fa2676d20e8',
        'code_revision': 'ae3f3bc11a66e99d18cdeb457e3c27cf15869dd6',
        'config_blob': 'f63143cebda7b62b96ae88150fad4c768a61294d',
        'weights': {'pytorch_model.bin': '7332f5e62e72f578f0310a19a52d1e91505ff9569c08c3d1823cde374045d8ef'},
    },
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mapped_name(name, family):
    if family in ('salmonn2_pro_8b', 'internvideo3_8b'):
        for source, target in [('model.language_model.', 'model.'), ('model.visual.', 'visual.'), ('model.audio.', 'audio.')]:
            if name.startswith(source):
                return target + name[len(source):]
    if name == 'lm_head.weight':
        return 'model.lm_head.weight'
    return name


def internvideo_shapes(config):
    """Validate dense checkpoint layout before publishing any native package."""
    text, vision = config['text_config'], config['vision_config']
    shapes = {}
    def linear(name, output, input_size, bias=False):
        shapes[name + '.weight'] = [output, input_size]
        if bias:
            shapes[name + '.bias'] = [output]
    def norm(name, size, bias=False):
        shapes[name + '.weight'] = [size]
        if bias:
            shapes[name + '.bias'] = [size]
    size, heads = text['hidden_size'], text['num_attention_heads']
    linear('model.language_model.embed_tokens', text['vocab_size'], size)
    linear('lm_head', text['vocab_size'], size)
    norm('model.language_model.norm', size)
    for i, rank in enumerate(text['kv_lora_rank_list']):
        prefix = f'model.language_model.layers.{i}'
        linear(prefix + '.self_attn.q_proj', heads * (text['qk_nope_head_dim'] + text['qk_rope_head_dim']), size, text['attention_bias'])
        linear(prefix + '.self_attn.kv_a_proj_with_mqa', rank + text['qk_rope_head_dim'], size, text['attention_bias'])
        linear(prefix + '.self_attn.kv_b_proj', heads * (text['qk_nope_head_dim'] + text['v_head_dim']), rank)
        linear(prefix + '.self_attn.o_proj', size, heads * text['v_head_dim'])
        for name in ('gate_proj', 'up_proj'):
            linear(prefix + '.mlp.' + name, text['intermediate_size'], size)
        linear(prefix + '.mlp.down_proj', size, text['intermediate_size'])
        for name in ('input_layernorm', 'post_attention_layernorm'):
            norm(prefix + '.' + name, size)
    size = vision['hidden_size']
    shapes['model.visual.patch_embed.proj.weight'] = [size, vision['in_channels'], vision['temporal_patch_size'], vision['patch_size'], vision['patch_size']]
    shapes['model.visual.patch_embed.proj.bias'] = [size]
    linear('model.visual.pos_embed', vision['num_position_embeddings'], size)
    for i in range(vision['depth']):
        prefix = f'model.visual.blocks.{i}'
        for name in ('norm1', 'norm2'):
            norm(prefix + '.' + name, size, True)
        linear(prefix + '.attn.qkv', size * 3, size, True)
        linear(prefix + '.attn.proj', size, size, True)
        linear(prefix + '.mlp.linear_fc1', vision['intermediate_size'], size, True)
        linear(prefix + '.mlp.linear_fc2', size, vision['intermediate_size'], True)
    merged = size * vision['spatial_merge_size'] ** 2
    norm('model.visual.merger.norm', size, True)
    linear('model.visual.merger.linear_fc1', merged, merged, True)
    linear('model.visual.merger.linear_fc2', vision['out_hidden_size'], merged, True)
    return shapes


def convert(source, destination, family):
    spec = SOURCES[family]
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError('Destination or conversion manifest already exists')
    config = read_json(source / 'config.json')
    config_bytes = (source / 'config.json').read_bytes()
    if hashlib.sha1(f'blob {len(config_bytes)}\0'.encode() + config_bytes).hexdigest() != spec['config_blob']:
        raise ValueError('Configuration does not match the supported pinned model revision')
    expected = {'salmonn2_pro_8b': 'qwen3_vl', 'padcaptioner_3b': 'qwen2_5_vl', 'internvideo3_8b': 'internvideo3'}[family]
    if config.get('model_type') != expected or (family != 'internvideo3_8b' and 'audio_config' not in config):
        raise ValueError('Wrong model architecture for the selected family')
    for filename, expected_hash in spec['weights'].items():
        actual = sha256(source / filename)
        if actual != expected_hash:
            raise ValueError(f'Incomplete or unexpected pinned source: {filename}')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    tokens = [('<|video_pad|>', config['video_token_id'])]
    if family != 'internvideo3_8b':
        tokens.append(('<|audio_pad|>', config['audio_token_id']))
    for text, token in tokens:
        if tokenizer.encode(text, add_special_tokens=False) != [token]:
            raise ValueError(f'Tokenizer does not match the checkpoint: {text}')
    assets = {p.name: p.read_text() for p in source.iterdir() if p.suffix in ('.json', '.jinja') and 'index' not in p.name}
    metadata = dict(format='pt', architecture=family, conversion_version='1', model_id=spec['repo'],
                    model_revision=spec['revision'], implementation_revision=spec['code_revision'],
                    captioner_config=json.dumps(config, separators=(',', ':')),
                    captioner_assets=json.dumps(assets, separators=(',', ':')),
                    source_sha256=json.dumps(spec['weights'], sort_keys=True))
    omitted = {}
    with tempfile.TemporaryDirectory(prefix='captioner-conversion-', dir=destination.parent) as staging:
        if family == 'padcaptioner_3b':
            # weights_only excludes arbitrary pickle execution; mmap bounds resident RAM.
            tensors = torch.load(source / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
            if not isinstance(tensors, dict) or any(not isinstance(t, torch.Tensor) for t in tensors.values()):
                raise ValueError('Expected a tensor-only state dictionary')
            if 'lm_head.weight' in tensors:
                if not torch.equal(tensors['lm_head.weight'], tensors['model.embed_tokens.weight']):
                    raise ValueError('PadCaptioner requires tied output and input embeddings')
                del tensors['lm_head.weight']
                omitted['lm_head.weight'] = 'verified identical to model.embed_tokens.weight; tied native output'
            save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(Path(staging) / 'model.safetensors'))
            del tensors
            records = inspect_shards(Path(staging))
        else:
            records = inspect_shards(source, expected=internvideo_shapes(config) if family == 'internvideo3_8b' else None)
        retained = {}
        for name, record in records.items():
            # Query-only BERT inference never reads these persistent position indices.
            if name.endswith('qformer.bert.embeddings.position_ids'):
                omitted[name] = 'unused BERT text-position indices; query-only encoder'
            else:
                retained[name] = record
        metadata['omitted_tensors'] = json.dumps(omitted, sort_keys=True)
        mapping = {name: mapped_name(name, family) for name in retained}
        return stream_checkpoint(retained, tokenizer.backend_tokenizer.to_str().encode(), destination,
                                 metadata, name_mapping=mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--family', choices=SOURCES, required=True)
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    result = convert(args.source, args.destination, args.family)
    print(f"Converted and independently verified {result['source_tensors']} tensors: {args.destination}")
