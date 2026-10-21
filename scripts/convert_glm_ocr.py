#!/usr/bin/env python3
"""Verify the pinned official GLM-OCR checkpoint and prepare an offline native text encoder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.glm_ocr import GlmOCR
from scripts.convert_moss_audio import read_json, stream_checkpoint, unique_object

REVISION = '2e85a62840ccac27daa451df36c736c4636b8628'
WEIGHT_SHA256 = 'a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815'
SOURCE_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                'chat_template.jinja', 'processor_config.json', 'preprocessor_config.json', 'README.md')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mapped_name(key):
    for source, target in [('model.language_model.', 'model.'), ('model.visual.', 'visual.'), ('lm_head.', 'model.lm_head.')]:
        if key.startswith(source):
            return target + key.removeprefix(source)
    raise ValueError(f'Unknown GLM-OCR tensor: {key}')


def expected_shapes():
    native = GlmOCR({}, torch.bfloat16, 'meta', manual_cast)
    shapes = {}
    for key, value in native.state_dict().items():
        if key.startswith('model.lm_head.'):
            original = key.removeprefix('model.')
        elif key.startswith('model.'):
            original = 'model.language_model.' + key.removeprefix('model.')
        else:
            original = 'model.' + key
        shapes[original] = list(value.shape)
    # The official reference excludes layer 16. Non-speculative inference deliberately
    # omits this exact verified MTP inventory, preserving hashes/shapes in the manifest.
    prefix = 'model.language_model.layers.16.'
    mtp = {prefix + key.removeprefix('model.language_model.layers.0.'): value for key, value in shapes.items() if key.startswith('model.language_model.layers.0.')}
    mtp.update({prefix + 'eh_proj.weight': [1536, 3072], prefix + 'embed_tokens.weight': [59392, 1536],
                prefix + 'enorm.weight': [1536], prefix + 'hnorm.weight': [1536],
                prefix + 'shared_head.head.weight': [59392, 1536], prefix + 'shared_head.norm.weight': [1536]})
    return shapes, mtp


def inspect_checkpoint(path, expected):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Incomplete GLM-OCR safetensors file')
        size = struct.unpack('<Q', raw)[0]
        if size > 100_000_000 or size + 8 > path.stat().st_size:
            raise ValueError('Invalid GLM-OCR header (possibly a Git LFS pointer)')
        header = json.loads(stream.read(size), object_pairs_hook=unique_object)
    with safe_open(path, framework='pt', device='cpu') as tensors:
        keys = set(tensors.keys())
    if keys != set(expected):
        raise ValueError(f'GLM-OCR tensor inventory mismatch: missing={sorted(set(expected)-keys)}, extra={sorted(keys-set(expected))}')
    offset = 0
    for key, entry in sorted(((k, header[k]) for k in keys), key=lambda item: item[1]['data_offsets'][0]):
        start, end = entry['data_offsets']
        length = 2 * math.prod(expected[key])
        if entry['shape'] != expected[key] or entry['dtype'] != 'BF16' or start != offset or end - start != length:
            raise ValueError(f'Invalid GLM-OCR shape, dtype or payload range: {key}')
        offset = end
    if offset + size + 8 != path.stat().st_size:
        raise ValueError('GLM-OCR payload size mismatch')
    return {key: (path, size + 8, header[key]) for key in keys}


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    originals, mtp = expected_shapes()
    records = inspect_checkpoint(source / 'model.safetensors', {**originals, **mtp})
    digest = sha256(source / 'model.safetensors')
    if digest != WEIGHT_SHA256:
        raise ValueError('GLM-OCR weight SHA-256 differs from the pinned official artifact')
    files = {name: sha256(source / name) for name in SOURCE_FILES}
    config = read_json(source / 'config.json')
    if config['model_type'] != 'glm_ocr' or config['text_config']['num_hidden_layers'] != 16:
        raise ValueError('Wrong GLM-OCR model configuration')
    omitted = {}
    for key in mtp:
        path, base, entry = records[key]
        start, end = entry['data_offsets']
        with path.open('rb') as stream:
            stream.seek(base + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError('Incomplete MTP payload')
                digest.update(chunk)
                remaining -= len(chunk)
        omitted[key] = dict(shape=entry['shape'], sha256=digest.hexdigest())
    metadata = {'format': 'pt', 'model': 'zai-org/GLM-OCR', 'model_revision': REVISION,
                'weight_sha256': WEIGHT_SHA256, 'source_files': json.dumps(files, sort_keys=True),
                'config': json.dumps(config, sort_keys=True), 'mtp_omitted': json.dumps(omitted, sort_keys=True),
                'preprocessing': 'original-pixels-v1: downscale only above 1000000 pixels; white right/bottom pad to 28',
                'mapping': json.dumps({key: mapped_name(key) for key in originals}, sort_keys=True)}
    stream_checkpoint({mapped_name(key): records[key] for key in originals}, (source / 'tokenizer.json').read_bytes(), destination, metadata)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    options = parser.parse_args()
    print(convert(options.source, options.destination))
