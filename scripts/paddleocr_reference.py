#!/usr/bin/env python3
"""Offline CPU reference comparison. Never downloads models or starts a service."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, PPOCRV6SmallRecForTextRecognition

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
from comfy.paddleocr import (PaddleOCRModel, decode_ctc, detection_regions, prepare_detector,
                            prepare_recognizer, rectify)


def edit_counts(expected, actual):
    expected = ' '.join(unicodedata.normalize('NFC', expected).split())
    actual = ' '.join(unicodedata.normalize('NFC', actual).split())
    table = [[(0, 0, i) for i in range(len(actual) + 1)]]
    for i, a in enumerate(expected, 1):
        row = [(0, i, 0)]
        for j, b in enumerate(actual, 1):
            sub, delete, insert = table[i - 1][j - 1], table[i - 1][j], row[j - 1]
            choices = [(sub[0] + (a != b), sub[1], sub[2]), (delete[0], delete[1] + 1, delete[2]), (insert[0], insert[1], insert[2] + 1)]
            row.append(min(choices, key=sum))
        table.append(row)
    substitutions, deletions, insertions = table[-1][-1]
    return dict(substitutions=substitutions, deletions=deletions, insertions=insertions,
                expected_characters=len(expected), cer=sum(table[-1][-1]) / len(expected) if expected else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', type=Path, required=True, help='Local official model directory')
    parser.add_argument('--recognizer', type=Path, required=True, help='Local official model directory')
    parser.add_argument('--cases', type=Path, required=True, help='JSON list: id, image path, optional crop xyxy, operation, expected text')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; existing results are preserved')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    native = PaddleOCRModel(options.detector / 'model.safetensors', options.recognizer / 'model.safetensors')
    references = {'detector': AutoModelForObjectDetection.from_pretrained(options.detector, local_files_only=True).eval(),
                  'recognizer': PPOCRV6SmallRecForTextRecognition.from_pretrained(options.recognizer, local_files_only=True).eval()}
    report = {'models': native.identity, 'device': 'cpu', 'dtype': 'float32', 'rtol': 1e-4, 'atol': 1e-5,
              'normalization': 'CER uses NFC and collapsed whitespace; case and punctuation remain significant.', 'cases': []}
    failures = []
    for index, case in enumerate(json.loads(options.cases.read_text())):
        source = Path(case['path'])
        image = Image.open(source).convert('RGB')
        if case.get('crop') is not None:
            image = image.crop(case['crop'])
        array = np.array(image)
        pixels = torch.from_numpy(array).movedim(-1, 0)
        for mode in ('no_enlargement', 'official'):
            start = time.perf_counter()
            arrays, comparisons = {}, []

            def compare(inputs, kind, label):
                actual = getattr(native, kind).model(inputs).detach()
                expected = references[kind](inputs).last_hidden_state.detach()
                arrays[label + '_native'] = actual.numpy()
                arrays[label + '_reference'] = expected.numpy()
                entry = {'stage': label, 'max_absolute_error': (actual - expected).abs().max().item()}
                try:
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    if kind == 'recognizer':
                        assert torch.equal(actual.argmax(-1), expected.argmax(-1)), 'CTC token sequence differs'
                    entry['passed'] = True
                except AssertionError as error:
                    entry.update(passed=False, error=str(error))
                    failures.append([case['id'], mode, label])
                comparisons.append(entry)
                return actual[0]

            if case.get('operation', 'detect_and_recognize') == 'recognize_lines':
                detector_transform = None
                regions = [{'corners': [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]], 'detection_score': None}]
            else:
                inputs, detector_transform = prepare_detector(pixels, mode)
                prediction = compare(inputs, 'detector', 'detector')[0].numpy()
                regions = detection_regions(prediction, detector_transform, mode)
            for region_index, region in enumerate(regions):
                if detector_transform is None:
                    crop, rectification = pixels, None
                else:
                    crop, rectification = rectify(array, region['corners'])
                inputs, transform = prepare_recognizer(crop, mode)
                probabilities = compare(inputs, 'recognizer', f'region_{region_index}')
                text, score = decode_ctc(probabilities, native.vocabulary)
                region.update(text=text, recognition_score=score, rectification=rectification, recognition_transform=transform)
            actual_text = '\n'.join(region['text'] for region in regions)
            record = {'id': case['id'], 'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'crop': case.get('crop'), 'size': list(image.size), 'preprocessing': mode, 'text': actual_text,
                      'detector_transform': detector_transform, 'regions': regions, 'comparisons': comparisons,
                      'elapsed_seconds': time.perf_counter() - start, 'expected': case.get('expected')}
            if case.get('expected') is not None:
                record['accuracy'] = edit_counts(case['expected'], actual_text)
            if case.get('known_strings'):
                record['partial_transcription'] = case['known_strings']
            np.savez_compressed(options.output / f'{index:03d}-{mode}.npz', **arrays)
            report['cases'].append(record)
            (options.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            print(case['id'], mode, repr(actual_text), flush=True)
    report['parity_failures'] = failures
    report['accuracy_by_mode'] = {}
    for mode in ('no_enlargement', 'official'):
        scored = [case['accuracy'] for case in report['cases'] if case['preprocessing'] == mode and 'accuracy' in case]
        totals = {key: sum(case[key] for case in scored) for key in ('substitutions', 'deletions', 'insertions', 'expected_characters')}
        totals['cer'] = sum(totals[key] for key in ('substitutions', 'deletions', 'insertions')) / totals['expected_characters'] if totals['expected_characters'] else None
        report['accuracy_by_mode'][mode] = totals
    (options.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
