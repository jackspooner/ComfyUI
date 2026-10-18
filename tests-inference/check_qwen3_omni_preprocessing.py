"""Audit embedded-tokenizer/media preprocessing against frozen upstream input captures on CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from unittest import mock

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.cpu = True
import comfy.sd
from comfy.text_encoders import qwen3_omni as omni
from scripts.qwen3_omni_checkpoint import SOURCE_REVISION, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    manifest = json.loads(options.model.with_suffix('.manifest.json').read_text())
    digest = sha256_file(options.model)
    if manifest['source_revision'] != SOURCE_REVISION or manifest['output_sha256'] != digest:
        raise ValueError('Artifact hash/revision differs from its manifest')
    reference = json.loads((options.reference / 'results.json').read_text())
    if (reference['source']['source_revision'] != SOURCE_REVISION or reference['versions']['transformers'] != '5.20.0'
            or reference['versions']['nvidia-modelopt'] != '0.42.0'):
        raise ValueError('Reference source or dependency pins differ')
    cases = json.loads(options.cases.read_text())
    if (len({case['id'] for case in cases}) != len(cases)
            or any(re.fullmatch(r'[A-Za-z0-9_-]+', case['id']) is None for case in cases)):
        raise ValueError('Fixture IDs must be unique safe capture filenames')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    clip = comfy.sd.load_clip([str(options.model)], clip_type=comfy.sd.CLIPType.QWEN3_OMNI, disable_dynamic=True)
    stage = clip.cond_stage_model.qwen3_omni
    model = stage.transformer
    pixels, mel = [], []
    processor_factory, audio_preprocess = omni.Qwen2VLImageProcessor.from_dict, omni.audio_features

    def image_processor(config):
        processor = processor_factory(config)
        def process(**kwargs):
            result = processor(**kwargs)
            pixels.append(result.pixel_values.clone())
            return result
        return process

    def audio_features(audio, config):
        features = audio_preprocess(audio, config)
        mel.append(features.clone())
        return features

    def image_forward(values, grids):
        count = sum(t * h * w // 4 for t, h, w in grids)
        features = torch.zeros(count, model.model.config.hidden_size, dtype=model.dtype)
        return features, [features] * len(model.visual.deepstack_merger_list)

    def audio_forward(values):
        count = omni.audio_output_length(values.shape[-1], model.audio_tower.chunk_size)
        return torch.zeros(count, model.model.config.hidden_size, dtype=model.dtype)

    report = {'artifact_sha256': digest, 'source_revision': SOURCE_REVISION,
              'harness_sha256': sha256_file(__file__), 'cases_sha256': sha256_file(options.cases),
              'reference_sha256': sha256_file(options.reference / 'results.json'),
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'comfy-kitchen')},
              'scope': 'Native tokenization, actual preprocessing and modality expansion; encoder forwards replaced with shape fixtures, not full-model parity',
              'cases': []}
    with mock.patch.object(omni.Qwen2VLImageProcessor, 'from_dict', image_processor), \
            mock.patch.object(omni, 'audio_features', audio_features), \
            mock.patch.object(model.visual, 'forward', image_forward), \
            mock.patch.object(model.audio_tower, 'forward', audio_forward):
        for case in cases:
            pixels.clear()
            mel.clear()
            image = torch.from_numpy(np.load(case['image_tensor'], allow_pickle=False)) if case.get('image_tensor') else None
            audio = None
            if case.get('audio'):
                waveform, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
                audio = {'waveform': torch.from_numpy(waveform.T.copy()).unsqueeze(0), 'sample_rate': rate}
            tokens = clip.tokenize(case['prompt'], image=image, audio=audio, thinking=False, system_prompt=case.get('system_prompt', ''))
            row = [item[0] for item in tokens['qwen3_omni'][0]]
            embeds, _, _, info = stage.process_tokens([row], torch.device('cpu'))
            ids, media = [], iter(info)
            for token in row:
                if isinstance(token, dict):
                    entry = next(media)
                    ids.extend([omni.IMAGE_TOKEN if token['type'] == 'image' else omni.AUDIO_TOKEN] * entry['size'])
                else:
                    ids.append(token)
            expected = torch.load(options.reference / (case['id'] + '.pt'), map_location='cpu', weights_only=True)
            result = {'id': case['id'], 'input_ids_equal': ids == expected['input_ids'][0].tolist(),
                      'positions_equal': torch.equal(omni.media_positions(info, embeds.shape[1], torch.device('cpu')), expected['position_ids'][:, 0]),
                      'images_consumed': len(pixels), 'audio_clips_consumed': len(mel)}
            assert len(pixels) == (0 if image is None else image.shape[0]), case['id']
            assert len(mel) == int(audio is not None), case['id']
            if pixels:
                result['pixels_fp32_equal'] = torch.equal(torch.cat(pixels), expected['pixel_values'])
            if mel:
                result['mel_fp32_equal'] = torch.equal(mel[0], expected['input_features'][0])
            if not all(value for key, value in result.items() if key.endswith('_equal')):
                raise AssertionError(result)
            report['cases'].append(result)
            print(case['id'], 'equal', flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
