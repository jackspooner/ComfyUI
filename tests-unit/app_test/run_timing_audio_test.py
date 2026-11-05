import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_audio import observe_audio_metrics, waveform_features
from app.run_timing_workload import describe, resolved_inputs


def description(graph):
    return describe(graph, {'profileRevision': 'implementation'},
                    {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'},
                    '2026-11-05T00:00:00Z', lambda key, value: digest([key, value]), digest)


def latest(store):
    return store.page('0', 100, [], [])['observations'][-1]


def test_linked_audio_seconds_use_native_even_rounding_without_video_axes():
    graph = {'seconds': {'class_type': 'PrimitiveFloat', 'inputs': {'value': 1.1}},
             'latent': {'class_type': 'EmptyLatentAudio', 'inputs': {'seconds': ['seconds', 0], 'batch_size': 2}},
             'sample': {'class_type': 'KSampler', 'inputs': {'latent_image': ['latent', 0], 'steps': 4}}}
    before = copy.deepcopy(graph)
    evidence, stages = description(graph)
    sample = stages[-1]['workload']
    assert sample['profile'] == evidence['profile'] == 'audio'
    assert sample['resolved']['audioLatentLength']['value'] == 24
    assert sample['resolved']['latentChannels']['value'] == 64
    assert sample['resolved']['batch']['value'] == 2
    assert 'latent_shape_unresolved' not in sample['reasons']
    assert not any(key in sample['resolved'] for key in ('latentWidth', 'latentHeight', 'latentFrames'))
    assert graph == before


def test_audio_encode_source_determines_sampling_modality_but_not_latent_dimensions():
    graph = {'encode': {'class_type': 'VAEEncodeAudio', 'inputs': {}},
             'sample': {'class_type': 'KSampler', 'inputs': {'latent_image': ['encode', 0]}}}
    evidence, stages = description(graph)
    assert evidence['profile'] == stages[-1]['workload']['profile'] == 'audio'
    assert 'latent_shape_unresolved' in stages[-1]['reasons']
    assert 'audioLatentLength' not in stages[-1]['workload']['resolved']


def test_audio_reference_loader_does_not_reclassify_text_inference_as_audio():
    evidence, stages = description({'audio': {'class_type': 'LoadAudio', 'inputs': {'audio': 'private.wav'}},
                                    'text': {'class_type': 'TextGenerate', 'inputs': {'audio': ['audio', 0]}}})
    assert evidence['profile'] == 'text'
    assert stages[0]['workload']['profile'] == 'audio'
    assert stages[1]['workload']['profile'] == 'text'


@pytest.mark.parametrize('kind', ['VAEEncodeAudio', 'SaveAudio', 'SaveAudioMP3', 'SaveAudioOpus', 'PreviewAudio'])
def test_resolved_waveforms_expose_only_metadata_at_input_boundary(kind):
    facts, references = resolved_inputs(kind, {'audio': {'waveform': SimpleNamespace(shape=(2, 2, 48000)),
                                                       'sample_rate': 24000}, 'filename_prefix': '/PRIVATE/output'}, 'now')
    assert facts['audioInput.durationSeconds'] == {'value': 2, 'availableAt': 'execution_start',
                                                  'observedAt': 'now', 'source': kind + '.resolved_inputs'}
    assert facts['audioInput.batch']['value'] == 2
    assert references[0][1]['samples']['value'] == 48000
    assert '/PRIVATE' not in json.dumps([facts, references])


@pytest.mark.parametrize('shape,rate', [((1, 2), 24000), ((1, 2, 0), 24000), ((1, True, 5), 24000),
                                       ((1, 2, 5), 0), ((1, 2, 5), True), ((1, 2, 5), float('nan'))])
def test_invalid_waveform_metadata_stays_unknown(shape, rate):
    assert waveform_features(shape, rate) is None
    facts, references = resolved_inputs('VAEEncodeAudio', {'audio': {'waveform': SimpleNamespace(shape=shape), 'sample_rate': rate}}, 'now')
    assert facts == {} and references == []


def test_processed_audio_is_live_and_decoded_output_never_becomes_input_evidence(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    graph = {'encode': {'class_type': 'VAEEncodeAudio', 'inputs': {}},
             'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {'samples': ['encode', 0]}}}
    item = (1, 'prompt', graph, {}, ['decode'])
    store.event('accepted', item)
    accepted = latest(store)
    store.event('started', item)
    store.safe_stage('prompt', 'encode', 'started')
    store.safe_metric('prompt', 'encode', 'audio_vae_encode_input',
                      {'shape': (1, 2, 24000), 'sample_rate': 24000, 'source_sample_rate': 12000})
    live = latest(store)
    assert live['stages'][0]['workload']['resolved']['audioProcessed.0.durationSeconds']['availableAt'] == 'live'
    assert 'audio_processed_workload_unmeasured' not in live['stages'][0]['reasons']
    assert not accepted['stages'][0]['workload']['resolved']
    store.safe_stage('prompt', 'encode', 'finished')
    store.safe_stage('prompt', 'decode', 'started')
    store.safe_metric('prompt', 'decode', 'audio_vae_decode_input',
                      {'shape': (1, 64, 12), 'tiled': False, 'normalize': True})
    before_output = latest(store)
    assert 'audio_decode_workload_unmeasured' not in before_output['stages'][1]['reasons']
    assert 'audio_output_unmeasured' in before_output['stages'][1]['reasons']
    assert before_output['stages'][1]['workload']['complete']
    assert 'audio_output_unmeasured' not in before_output['stages'][1]['workload']['reasons']
    assert before_output['workload']['evidence']['complete']
    store.safe_metric('prompt', 'decode', 'audio_decoded', {'shape': (1, 2, 24000), 'sample_rate': 24000})
    store.safe_stage('prompt', 'decode', 'metrics')
    measured = latest(store)
    output = measured['stages'][1]['workload']['measurements']['audioOutput.0.durationSeconds']
    assert output['value'] == 1 and output['availableAt'] == 'completion'
    assert not any('audioOutput' in key for key in measured['workload']['evidence']['resolved'])
    assert not before_output['stages'][1]['workload']['measurements']
    store.safe_stage('prompt', 'decode', 'finished')
    stage = latest(store)['stages'][1]
    assert stage['workload']['complete'] and stage['reasons'] == []
    schema = json.loads((Path(__file__).parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


def test_output_metadata_failures_preserve_input_completeness_and_explicit_stage_gap(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', {'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {}}}, {}, ['decode'])
    store.event('accepted', item)
    accepted = latest(store)
    assert 'audio_decode_workload_unmeasured' in accepted['workload']['evidence']['reasons']
    assert 'audio_output_unmeasured' not in accepted['workload']['evidence']['reasons']
    store.event('started', item)
    store.safe_stage('prompt', 'decode', 'started')
    store.safe_metric('prompt', 'decode', 'audio_vae_decode_input', {'shape': (1, 64, 12), 'tiled': False, 'normalize': False})
    store.safe_metric('prompt', 'decode', 'audio_decoded', {'shape': (1, 0), 'sample_rate': 0})
    store.safe_metric('prompt', 'decode', 'tensor_shape_unavailable', {'operation': 'audio_decoded'})
    store.safe_stage('prompt', 'decode', 'finished')
    stage = latest(store)['stages'][0]
    assert stage['workload']['complete'] and stage['workload']['reasons'] == []
    assert 'audio_output_metadata_unavailable' in stage['reasons']
    assert not stage['workload']['measurements']
    store.event('finished', item, SimpleNamespace(completed=True, messages=[]))
    final = latest(store)
    assert final['workload']['evidence']['complete']
    assert 'audio_output_metadata_unavailable' in final['stages'][0]['reasons']


def test_missing_stage_end_is_measurement_only_but_unobserved_execution_stays_incomplete(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', {'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {}}}, {}, ['decode'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'decode', 'started')
    store.safe_metric('prompt', 'decode', 'audio_vae_decode_input', {'shape': (1, 64, 12), 'tiled': False, 'normalize': False})
    store.event('finished', item, SimpleNamespace(completed=True, messages=[]))
    final = latest(store)
    assert final['stages'][0]['phase'] == 'unknown'
    assert final['stages'][0]['durationSeconds'] is None
    assert 'stage_completion_unobserved' in final['stages'][0]['reasons']
    assert final['workload']['evidence']['complete']
    other = (2, 'unobserved', {'prepare': {'class_type': 'EmptyLatentAudio', 'inputs': {'seconds': 1.1, 'batch_size': 1}}}, {}, ['prepare'])
    store.event('accepted', other)
    store.event('started', other)
    store.event('finished', other, SimpleNamespace(completed=True, messages=[]))
    final = latest(store)
    assert 'stage_execution_unobserved' in final['workload']['evidence']['reasons']
    assert not final['workload']['evidence']['complete']


@pytest.mark.parametrize('shape', [(2, 64, 12), (2, 32, 2, 12)])
def test_runtime_audio_sampling_shape_keeps_reliable_counts_without_spatial_inference(tmp_path, shape):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    graph = {'encode': {'class_type': 'VAEEncodeAudio', 'inputs': {}},
             'sample': {'class_type': 'KSampler', 'inputs': {'latent_image': ['encode', 0]}}}
    item = (1, 'prompt', graph, {}, ['sample'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'sample', 'started')
    store.safe_metric('prompt', 'sample', 'sampling_plan', {'shape': shape, 'evaluations': 2, 'branches': 1, 'precision': 'torch.bfloat16'})
    stage = latest(store)['stages'][1]
    precision = stage['workload']['resolved']['executionPrecision']
    assert precision['value'] == 'torch.bfloat16'
    assert precision['availableAt'] == 'execution_start' and precision['source'] == 'native_sampler'
    assert not any(key in stage['workload']['resolved'] for key in ('latentWidth', 'latentHeight', 'latentFrames'))
    if len(shape) == 3:
        assert stage['workload']['resolved']['audioLatentLength']['value'] == 12
        assert stage['workload']['resolved']['audioLatentChannels']['value'] == 1
        assert 'latent_shape_unresolved' not in stage['reasons']
    else:
        assert 'audioLatentLength' not in stage['workload']['resolved']
        assert stage['workload']['resolved']['audioLatent.axis.2']['value'] == 2
        assert 'audio_latent_layout_unresolved' in stage['reasons']
    for _ in range(2):
        store.safe_metric('prompt', 'sample', 'noise_evaluation', {'branches': 1})
    store.safe_stage('prompt', 'sample', 'finished')
    stage = latest(store)['stages'][1]
    assert stage['workload']['complete'] is (len(shape) == 3)
    assert stage['workload']['measurements']['noiseEvaluations']['value'] == 2


def test_tiled_and_repeated_audio_keep_explicit_gaps_and_bounded_features():
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    assert not observe_audio_metrics(state, 'unknown', {})
    observe_audio_metrics(state, 'audio_vae_decode_input',
                          {'shape': (1, 32, 2, 3), 'tiled': True, 'normalize': False, 'tile_size': 8, 'overlap': 2})
    assert 'audio_tile_work_unmeasured' in state['reasons']
    assert state['resolved']['audioDecode.0.latent.axis.2'] == 2
    for _ in range(50):
        observe_audio_metrics(state, 'audio_decoded', {'shape': (1, 2, 24000), 'sample_rate': 24000})
    assert state['audioOutput_count'] == 50
    assert len(state['measurements']) == 16 * 5
    assert 'multiple_audio_invocations' in state['reasons'] and 'audio_invocation_limit' in state['reasons']


def test_decode_fallback_changes_live_regime_without_backdating_or_claiming_tile_counts(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', {'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {}}}, {}, ['decode'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'decode', 'started')
    store.safe_metric('prompt', 'decode', 'audio_vae_decode_input', {'shape': (1, 64, 3), 'tiled': False, 'normalize': False})
    before = latest(store)
    store.safe_metric('prompt', 'decode', 'vae_decode_tiled_fallback', {})
    changed = latest(store)
    assert 'vaeDecode.tiledFallback' not in before['stages'][0]['workload']['resolved']
    fallback = changed['stages'][0]['workload']['resolved']['vaeDecode.tiledFallback']
    assert fallback['value'] is True and fallback['availableAt'] == 'live'
    assert changed['stages'][0]['workload']['resolved']['audioDecode.0.tiled']['value'] is False
    assert 'vae_decode_tile_work_unmeasured' in changed['stages'][0]['reasons']
