"""Compare native Omni captures with a reconstructed reference, retaining incomplete coverage."""
import argparse
import json
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen3_omni_checkpoint import FP8_REVISION, SOURCE_REVISION, sha256_file


def difference(actual, expected):
    if actual.shape != expected.shape:
        return {'shape_mismatch': [list(actual.shape), list(expected.shape)]}
    result = {'equal': torch.equal(actual, expected)}
    if not actual.is_floating_point():
        result['fraction_equal'] = (actual == expected).float().mean().item()
        return result
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return {**result, 'finite': False}
    delta = actual - expected
    return {**result, 'finite': True, 'max_absolute_error': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    torch.set_num_threads(4)
    reference = json.loads((options.reference / 'results.json').read_text())
    revision = reference.get('source_revision', reference.get('source', {}).get('source_revision'))
    if revision not in (SOURCE_REVISION, FP8_REVISION):
        raise ValueError('Reference revision differs')
    expected_cases = {case['id']: case for case in reference['cases']}
    report = {'source_revision': revision, 'reference_sha256': sha256_file(options.reference / 'results.json'),
              'harness_sha256': sha256_file(__file__), 'runs': [], 'cases': [], 'production_accepted': False}
    identity = None
    seen = set()
    for directory in options.native:
        path = directory / 'results.json'
        native = json.loads(path.read_text())
        if native['source_revision'] != revision:
            raise ValueError('Native revision differs')
        current = {key: native.get(key) for key in ('artifact_sha256', 'source_code_sha256', 'cases_sha256', 'precision', 'nvfp4_input_passes_override', 'vram_headroom_gib')}
        if identity is None:
            identity = current
        elif current != identity or not current['source_code_sha256']:
            raise ValueError('Cannot combine different artifacts, inference sources, fixture lists or precision policies')
        report['runs'].append({'path': str(directory), 'results_sha256': sha256_file(path),
                               'execution_status': native.get('status', 'not recorded'), 'identity': current})
        for case in native['cases']:
            name = case['id']
            if name not in expected_cases or name in seen or re.fullmatch(r'[A-Za-z0-9_-]+', name) is None:
                raise ValueError('Unknown, duplicate or unsafe fixture ID')
            seen.add(name)
            expected = expected_cases[name]
            if [item['alternatives'] for item in case['assertions']] != [item['alternatives'] for item in expected['assertions']]:
                raise ValueError('Fixture assertions differ: ' + name)
            common = 0
            for a, b in zip(case['generated_ids'], expected['generated_ids']):
                if a != b:
                    break
                common += 1
            a_path, b_path = directory / (name + '.pt'), options.reference / (name + '.pt')
            actual = torch.load(a_path, map_location='cpu', weights_only=True)
            baseline = torch.load(b_path, map_location='cpu', weights_only=True)
            required = {key for key in baseline if key in ('decode_logits', 'position_ids', 'audio_features', 'vision_features')
                        or key.startswith(('router_', 'vision_deepstack_'))}
            if not required <= actual.keys():
                raise ValueError(f'Missing native captures for {name}: {sorted(required - actual.keys())}')
            count = min(common + 1, actual['decode_logits'].shape[0], baseline['decode_logits'].shape[0])
            item = {'id': name, 'run': str(directory), 'capture_sha256': sha256_file(a_path),
                    'new_assertion_failures': [index for index, (a, b) in enumerate(zip(case['assertions'], expected['assertions'])) if b['passed'] and not a['passed']],
                    'fixture_passed': case['fixture_passed'], 'generated_ids_equal': case['generated_ids'] == expected['generated_ids'],
                    'matching_generated_prefix': common, 'matching_context_logits': count, 'tensors': {}}
            for key in sorted(actual.keys() & baseline.keys()):
                a, b = actual[key], baseline[key]
                if key == 'decode_logits':
                    a, b = a[:count], b[:count]
                metrics = difference(a, b)
                if key.endswith('_indices') and a.shape == b.shape:
                    metrics['expert_set_overlap'] = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean().item()
                item['tensors'][key] = metrics
            if 'pixel_values_compute' in actual and 'pixel_values' in baseline:
                item['tensors']['compute_pixels'] = difference(actual['pixel_values_compute'], baseline['pixel_values'].to(actual['pixel_values_compute'].dtype))
            report['cases'].append(item)
    covered = {case['id'] for case in report['cases']}
    report['missing_cases'] = sorted(set(expected_cases) - covered)
    report['complete_fixture_coverage'] = not report['missing_cases']
    report['new_assertion_failure_count'] = sum(len(case['new_assertion_failures']) for case in report['cases'])
    report['scope'] = 'Numerical/fixture evidence only. Incomplete or interrupted runs do not establish lifecycle, performance or live installation.'
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete_fixture_coverage': report['complete_fixture_coverage'],
                      'missing_cases': report['missing_cases'], 'new_assertion_failure_count': report['new_assertion_failure_count']}))


if __name__ == '__main__':
    main()
