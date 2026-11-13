#!/usr/bin/env python3
"""Convert a complete local Qwen3-ASR/aligner or HF Whisper snapshot offline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
import torch
from transformers import Qwen2TokenizerFast, WhisperTokenizerFast
from convert_moss_audio import inspect_shards, read_json, stream_checkpoint


def convert(source, destination, revision, korean_dictionary=None):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Supply the full source snapshot commit revision')
    source = Path(source).resolve()
    config = read_json(source / 'config.json')
    kind = config['model_type']
    if kind == 'qwen3_asr':
        from comfy.asr.qwen import networks
        encoder, decoder = networks(config, device='meta')
        mapping = {}
        tokenizer = Qwen2TokenizerFast.from_pretrained(source, local_files_only=True)
        records = inspect_shards(source)
        for name in records:
            if name.startswith('thinker.audio_tower.'):
                mapping[name] = 'encoder.' + name.removeprefix('thinker.audio_tower.')
            elif name.startswith('thinker.model.'):
                mapping[name] = 'decoder.' + name.removeprefix('thinker.model.')
            elif name == 'thinker.lm_head.weight':
                mapping[name] = 'decoder.lm_head.weight'
            else:
                raise ValueError(f'Unexpected Qwen tensor: {name}')
        for text, token in [('<|audio_pad|>', config['thinker_config']['audio_token_id']), ('<|im_end|>', 151645)]:
            if tokenizer.encode(text, add_special_tokens=False) != [token]:
                raise ValueError(f'Unexpected Qwen tokenizer ID: {text}')
        metadata = {}
        if korean_dictionary is not None:
            dictionary = Path(korean_dictionary).read_bytes()
            metadata['korean_scores'] = json.dumps({line.split()[0]: 1.0 for line in dictionary.decode().splitlines() if line.strip()})
            metadata['korean_dictionary_sha256'] = hashlib.sha256(dictionary).hexdigest()
    elif kind == 'whisper':
        from comfy.asr.whisper import networks
        encoder, decoder = networks(config, device='meta')
        tokenizer = WhisperTokenizerFast.from_pretrained(source, local_files_only=True)
        records = inspect_shards(source)
        mapping = {name: name.removeprefix('model.') for name in records}
        if 'proj_out.weight' in records:
            mapping['proj_out.weight'] = 'decoder.proj_out.weight'
        else:
            # HF omits the tied head; preserve the same payload in both native
            # component keys rather than inventing/reinitializing output weights.
            records['proj_out.weight'] = records['model.decoder.embed_tokens.weight']
            mapping['proj_out.weight'] = 'decoder.proj_out.weight'
        metadata = {'generation_config': json.dumps(read_json(source / 'generation_config.json')),
                    'tokenizer_config': json.dumps(read_json(source / 'tokenizer_config.json'))}
    else:
        raise ValueError(f'Unsupported ASR source architecture: {kind}')
    expected = {**{'encoder.' + name: list(value.shape) for name, value in encoder.state_dict().items()},
                **{'decoder.' + name: list(value.shape) for name, value in decoder.state_dict().items()}}
    if len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(expected):
        raise ValueError(f'Incomplete tensor mapping: missing={sorted(set(expected)-set(mapping.values()))}, extra={sorted(set(mapping.values())-set(expected))}')
    for name, (_, _, entry) in records.items():
        if entry['shape'] != expected[mapping[name]] or entry['dtype'] not in ('BF16', 'F16', 'F32'):
            raise ValueError(f'Unsupported tensor shape/dtype: {name}')
    metadata.update(format='pt', architecture=kind, asr_package_version='1', source_revision=revision,
                    reference_implementation='transformers-5.22.2', config=json.dumps(config),
                    preprocessor_config=json.dumps(read_json(source / 'preprocessor_config.json')),
                    source_config_sha256=hashlib.sha256((source / 'config.json').read_bytes()).hexdigest())
    return stream_checkpoint(records, tokenizer.backend_tokenizer.to_str().encode(), destination,
                             metadata, name_mapping=mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--korean-dictionary', type=Path, help='Pinned Qwen ASR korean_dict_jieba.dict; embedded for original Korean segmentation')
    options = parser.parse_args()
    manifest = convert(options.source, options.destination, options.revision, options.korean_dictionary)
    print(f'Converted {manifest["source_tensors"]} tensors; manifest: {options.destination.with_suffix(".manifest.json")}')
