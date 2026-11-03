#!/usr/bin/env python3
"""Offline, lossless single-file packaging of the two pinned audio-visual captioners."""
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
    if family == 'salmonn2_pro_8b':
        for source, target in [('model.language_model.', 'model.'), ('model.visual.', 'visual.'), ('model.audio.', 'audio.')]:
            if name.startswith(source):
                return target + name[len(source):]
    if name == 'lm_head.weight':
        return 'model.lm_head.weight'
    return name


def convert(source, destination, family):
    spec = SOURCES[family]
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError('Destination or conversion manifest already exists')
    config = read_json(source / 'config.json')
    config_bytes = (source / 'config.json').read_bytes()
    if hashlib.sha1(f'blob {len(config_bytes)}\0'.encode() + config_bytes).hexdigest() != spec['config_blob']:
        raise ValueError('Configuration does not match the supported pinned model revision')
    expected = 'qwen3_vl' if family == 'salmonn2_pro_8b' else 'qwen2_5_vl'
    if config.get('model_type') != expected or 'audio_config' not in config:
        raise ValueError('Wrong model architecture; the audio-visual checkpoint is required')
    for filename, expected_hash in spec['weights'].items():
        actual = sha256(source / filename)
        if actual != expected_hash:
            raise ValueError(f'Incomplete or unexpected pinned source: {filename}')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    for text, token in [('<|audio_pad|>', config['audio_token_id']), ('<|video_pad|>', config['video_token_id'])]:
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
            records = inspect_shards(source)
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
