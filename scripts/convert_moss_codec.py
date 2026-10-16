#!/usr/bin/env python3
"""Convert the pinned MOSS v1 FP32 codec offline, folding weight normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.convert_moss_audio import file_identity, inspect_shards, read_json, stream_checkpoint
from scripts.convert_moss_speech import verify_sources


def fold_convolutions(records):
    converted = {}
    transforms = {}
    for key, (path, _, _) in records.items():
        if not key.endswith('.parametrizations.weight.original0'):
            continue
        prefix = key.removesuffix('.parametrizations.weight.original0')
        direction_key = prefix + '.parametrizations.weight.original1'
        direction_path = records[direction_key][0]
        with safe_open(path, framework='pt', device='cpu') as weights:
            magnitude = weights.get_tensor(key)
        with safe_open(direction_path, framework='pt', device='cpu') as weights:
            direction = weights.get_tensor(direction_key)
        weight = torch._weight_norm(direction, magnitude, 0)
        destination = prefix + '.weight'
        converted[destination] = weight
        transforms[destination] = dict(operation='torch._weight_norm(direction, magnitude, dim=0)', dtype='F32',
            sources=[dict(key=name, file=records[name][0].name, shape=records[name][2]['shape'],
                          sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
                     for name, value in ((key, magnitude), (direction_key, direction))])
    return converted, transforms


def convert(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Destination or manifest already exists: {destination}')
    provenance = read_json(Path(__file__).with_name('moss_codec_v1.json'))
    snapshots = verify_sources(source, provenance)
    records = inspect_shards(source)
    for key, (_, _, entry) in records.items():
        if entry['dtype'] != 'F32':
            raise ValueError(f'Expected FP32 codec weight: {key}')
    folded, transforms = fold_convolutions(records)
    consumed = {item['key'] for transform in transforms.values() for item in transform['sources']}
    remaining = {key: value for key, value in records.items() if key not in consumed}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.moss-codec-fold-') as temporary:
        temporary = Path(temporary)
        folded_path = temporary / 'model.safetensors'
        save_file(folded, folded_path)
        folded_records = inspect_shards(temporary)
        if remaining.keys() & folded_records.keys():
            raise ValueError('Folded codec weight collides with an original tensor')
        snapshots[folded_path] = file_identity(folded_path)
        metadata = dict(format='pt', architecture='moss_audio_tokenizer', conversion_version='1',
                        model_id=provenance['model_id'], model_revision=provenance['revision'],
                        source_manifest=json.dumps(provenance, separators=(',', ':')),
                        source_config=(source / 'config.json').read_text(encoding='utf-8'),
                        source_tensor_count=str(len(records)),
                        weight_norm_transforms=json.dumps(transforms, separators=(',', ':')))
        return stream_checkpoint(remaining | folded_records, None, destination, metadata, source_stats=snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.source, args.destination)
    print(f"Verified {result['source_tensors']} inference tensors ({result['source_bytes']} bytes): {args.destination}")


if __name__ == '__main__':
    main()
