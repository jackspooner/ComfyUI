#!/usr/bin/env python3
"""Prepare the pinned ViiTorVoice-NAR release offline, preserving every tensor."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.convert_moss_audio import file_identity, inspect_shards, stream_checkpoint

REVISION = 'c225ea3e999c5bb0092c41f01e1ab0f2576fcf3e'
IMPLEMENTATION = '40359993534de0980eefff2d5a25bddc30f6bf58'
HASHES = {
    'assets/dualcodec_silence_2s.pt': '449b686e31a2bc7a941236f3379450ce76160cb33869fbcf4da3502d5bb66df5',
    'llm/0p6_emotion/config.json': '87030f7df8a116d5bac6525221189a3f67f80e6a0b224edeebdaddcb20cf0308',
    'llm/0p6_emotion/tokenizer.json': 'ea41cd36295bf9a1f061b633bff46e6f4ec1c94f45c9ae295563601e0d65b3b2',
    'llm/0p6_emotion/tokenizer_config.json': '93b20c9e2f7a3cbea041c10b9340dc0e156750a8c213d816dc54929cd83ca663',
    'llm/0p6_emotion/model.safetensors': '73ee77f45b95d2579381c9d93b3dd1d29469644e9ce9c5c695743563956f9365',
    'dualcodec/w2v-bert-2.0/config.json': 'f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675',
    'dualcodec/w2v-bert-2.0/preprocessor_config.json': '8e6281aad64f97e40534135a59dcc5d33571efae376f2a25adf5551951897ab4',
    'dualcodec/w2v-bert-2.0/model.safetensors': 'eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043',
    'dualcodec/dualcodec_ckpts/dualcodec_25hz_16384_1024.safetensors': '9288d481d789fb846a195a1794665315ac268825c2c9e5ee98e46da38bf1d175',
    'dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia.pt': 'c9c176c2b8850ab2e3ba828bbfa969deaf4566ce55db5f2687b8430b87526ad2',
}


def shapes(model):
    return {name: list(value.shape) for name, value in model.state_dict().items()}


def component_records(path, model=None):
    # The shared scanner accepts the standard model.safetensors filename. Alias
    # the exact codec file privately, without changing the original directory.
    with tempfile.TemporaryDirectory(prefix='viitor-inventory-') as temporary:
        temporary = Path(temporary)
        (temporary/'model.safetensors').symlink_to(path.resolve())
        records = {key: (path, base, entry) for key, (_, base, entry) in
                   inspect_shards(temporary, allow_linked_weights=True).items()}
    if model is not None:
        expected = model.state_dict()
        if records.keys() != expected.keys() or any(entry['shape'] != list(expected[key].shape) or
                entry['dtype'] != ('I64' if expected[key].dtype == torch.int64 else 'F32')
                for key, (_, _, entry) in records.items()):
            raise ValueError('Pinned ViiTor shape/dtype inventory mismatch: '+str(path))
    return records


def convert(source, destination):
    from comfy.cli_args import args
    args.cpu = True
    from comfy.audio_encoders.wav2vec2_bert import Wav2Vec2Bert
    from comfy.ldm.audio.dualcodec import DualCodec
    from comfy.viitorvoice.model import ViiTorVoice
    targets = {kind: destination/category/name for kind, category, name in (
        ('language', 'text_encoders', 'viitorvoice.safetensors'),
        ('semantic', 'audio_encoders', 'viitor_w2v_bert.safetensors'),
        ('codec', 'vae', 'viitor_dualcodec.safetensors'))}
    if any(path.exists() or path.with_suffix('.manifest.json').exists() for path in targets.values()):
        raise FileExistsError('ViiTor component or manifest exists; preserve it and choose a fresh destination')
    snapshots = {source/name: file_identity(source/name) for name in HASHES}
    for path in snapshots:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != HASHES[str(path.relative_to(source))]:
                raise ValueError('Pinned ViiTor source mismatch: '+str(path))
    llm = json.loads((source/'llm/0p6_emotion/config.json').read_text())
    w2v = json.loads((source/'dualcodec/w2v-bert-2.0/config.json').read_text())
    processor = json.loads((source/'dualcodec/w2v-bert-2.0/preprocessor_config.json').read_text())
    models = dict(language=ViiTorVoice(llm, torch.float32, 'meta', torch.nn),
        semantic=Wav2Vec2Bert(w2v, torch.float32, 'meta', torch.nn),
        codec=DualCodec(dtype=torch.float32, device='meta', operations=torch.nn))
    common = dict(source_revision=REVISION, implementation_revision=IMPLEMENTATION, conversion_version='1',
                  sources=json.dumps(HASHES, sort_keys=True))
    results = {}
    language = component_records(source/'llm/0p6_emotion/model.safetensors', models['language'])
    results['language'] = stream_checkpoint(language, (source/'llm/0p6_emotion/tokenizer.json').read_bytes(),
        targets['language'], dict(common, format='viitorvoice_language', config=json.dumps(llm)),
        source_stats=snapshots, name_mapping={key: key for key in language})
    semantic = component_records(source/'dualcodec/w2v-bert-2.0/model.safetensors', models['semantic'])
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix='.viitor-fold-') as temporary:
        temporary = Path(temporary)
        stats = torch.load(source/'dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia.pt', weights_only=True, map_location='cpu')
        if set(stats) != {'mean', 'var'} or any(value.shape != (1024,) for value in stats.values()):
            raise ValueError('Unexpected DualCodec semantic normalization statistics')
        extra_path = temporary/'model.safetensors'
        save_file(dict(semantic_mean=stats['mean'].float(), semantic_std=stats['var'].sqrt().float()), extra_path)
        extra = inspect_shards(temporary)
        extra_stats = snapshots | {extra_path: file_identity(extra_path)}
        combined = semantic | extra
        results['semantic'] = stream_checkpoint(combined, None, targets['semantic'],
            dict(common, format='viitorvoice_semantic', config=json.dumps(w2v), processor=json.dumps(processor)),
            source_stats=extra_stats, name_mapping={key: key for key in combined})
        extra_path.unlink()
        records = component_records(source/'dualcodec/dualcodec_ckpts/dualcodec_25hz_16384_1024.safetensors')
        folded, transforms = {}, {}
        with safe_open(source/'dualcodec/dualcodec_ckpts/dualcodec_25hz_16384_1024.safetensors', framework='pt') as saved:
            for key in records:
                if key.endswith('.weight_g'):
                    direction = key.removesuffix('_g')+'_v'
                    target = key.removesuffix('_g')
                    folded[target] = torch._weight_norm(saved.get_tensor(direction), saved.get_tensor(key), 0)
                    transforms[target] = dict(operation='torch._weight_norm(direction, magnitude, dim=0)', sources=[direction, key])
        silence = torch.load(source/'assets/dualcodec_silence_2s.pt', weights_only=True, map_location='cpu')
        if silence['tokens'].shape != (12, 49) or silence['model_id'] != '25hz_v1':
            raise ValueError('Unexpected pinned DualCodec silence asset')
        folded['silence_codes'] = silence['tokens'].long()
        save_file(folded, extra_path)
        folded_records = inspect_shards(temporary)
        combined = {key: value for key, value in records.items() if not key.endswith(('.weight_g', '.weight_v'))} | folded_records
        expected = shapes(models['codec']) | {'silence_codes': [12, 49]}
        if combined.keys() != expected.keys() or any(value[2]['shape'] != expected[key] for key, value in combined.items()):
            raise ValueError('Incomplete folded DualCodec tensor inventory')
        results['codec'] = stream_checkpoint(combined, None, targets['codec'],
            dict(common, format='viitorvoice_codec', architecture='dualcodec_25hz_v1',
                weight_norm_transforms=json.dumps(transforms), source_tensor_count=str(len(records)),
                semantic_component='../audio_encoders/viitor_w2v_bert.safetensors'),
            source_stats=snapshots | {extra_path: file_identity(extra_path)}, name_mapping={key: key for key in combined})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    values = parser.parse_args()
    for name, manifest in convert(values.source, values.destination).items():
        print(name, manifest['source_tensors'], manifest['source_bytes'])
