#!/usr/bin/env python3
"""Split the exact installed LocateAnything release into verified native components."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import Qwen2TokenizerFast

from scripts.convert_moss_audio import file_identity, inspect_shards, read_json, stream_checkpoint

REVISION = 'c32291ca5e996f5a7a485845b4f57a233936bba0'
SOURCE_HASHES = {
    'config.json': '59e6b5104f9d948db6a38f778e29f86d5c01e373f46d02008fc3070377917007',
    'preprocessor_config.json': '34f0dc33b40ee26d280d7ed93614c90b1d41a68d6ccfe5a8341e274e6169f94e',
    'processor_config.json': '1274db3b9504d37e57ea41ba7de547194d381265c5946f6db8560f851a940992',
    'tokenizer_config.json': '930b057de30312f861a22780017b09eccd87e893d966bd004bf4d70eec0e2652',
    'vocab.json': '87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'added_tokens.json': '1a87d2bec4c707c3946235046555d91f9df6986b8dd4a3ac53e6d0b24c36d176',
    'special_tokens_map.json': 'bfadce2f545458bf9d39fc9153cd2ac1077371ed5ab553fa4988061fefe67ac7',
    'modeling_locateanything.py': 'ffe736fb8ded5d597201704ccd85134d18a8e4dea43d309228644737234b7244',
    'modeling_qwen2.py': 'aadb676c0a587a16b7071977c159df16299fad22d88ee8ed9754881ab7f59575',
    'modeling_vit.py': '96479eb121c840f009a32830c78740154171290419108caefcf8778580700373',
    'generate_utils.py': '863187051772549928bf103b58f6176263c9b786fb19ef83fd7b2e76169fa65e',
    'image_processing_locateanything.py': '5109868add766c7e244487ecfff6a6f5a4aa1497b38b385a33a969f12e23b4ec',
    'processing_locateanything.py': '682145ed054b1e912e66273b476e51a25b2d666d4a37b26385af9300b66d40d8',
    'mask_sdpa_utils.py': '7e9d600eb25283963cc1696da060813066b9795da467cf9fb0ca68bfc8de1e1d',
    'model-00001-of-00002.safetensors': '923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58',
    'model-00002-of-00002.safetensors': '3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47',
}


def expected_shapes(config):
    from comfy.locateanything.model import Language, Vision
    with torch.device('meta'):
        vision = Vision(config, torch.bfloat16, 'meta', torch.nn).state_dict()
        language = Language(config, torch.bfloat16, 'meta', torch.nn).state_dict()
    return {**{key: list(value.shape) for key, value in vision.items()},
            **{'language_model.'+key: list(value.shape) for key, value in language.items()}}


def convert(source, destination):
    from comfy.cli_args import args
    args.cpu = True
    targets = [destination/'clip_vision/locateanything_vision.safetensors',
               destination/'text_encoders/locateanything_language.safetensors']
    if any(path.exists() or path.with_suffix('.manifest.json').exists() for path in targets):
        raise FileExistsError('LocateAnything component or manifest exists; preserve it and select a fresh destination')
    snapshots = {source/name: file_identity(source/name) for name in SOURCE_HASHES}
    for path in snapshots:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != SOURCE_HASHES[path.name]:
                raise ValueError('LocateAnything pinned source mismatch: '+str(path))
    config, processor = read_json(source/'config.json'), read_json(source/'preprocessor_config.json')
    records = inspect_shards(source, expected_shapes(config))
    if len(records) != 770:
        raise ValueError('LocateAnything requires the complete 770-tensor release')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    if len(tokenizer) != 152681 or tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False) != [151665]:
        raise ValueError('LocateAnything tokenizer identity mismatch')
    tokenizer_json = tokenizer.backend_tokenizer.to_str().encode()
    for kind, target in zip(('vision', 'language'), targets):
        selected = {key: value for key, value in records.items() if key.startswith('language_model.') == (kind == 'language')}
        metadata = dict(format='locateanything_'+kind, source_revision=REVISION,
            config=json.dumps(config), processor=json.dumps(processor), sources=json.dumps(SOURCE_HASHES))
        stream_checkpoint(selected, tokenizer_json if kind == 'language' else None, target, metadata,
            source_stats=snapshots, name_mapping={key: key.removeprefix('language_model.') if kind == 'language' else key for key in selected})
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    arguments = parser.parse_args()
    for path in convert(arguments.source, arguments.destination):
        print(path)
