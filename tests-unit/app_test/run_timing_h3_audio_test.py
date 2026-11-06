import copy

import pytest

from app.run_timing_h3_audio import (MAX_AUDIO_OPERATIONS, observe_h3_audio_metrics,
    h3_audio_complete, h3_reference_audio_complete, scoped_h3_audio_replaces_legacy_limit)
from app.run_timing_h3_references import observe_h3_reference_metrics
from app_test.run_timing_h3_references_test import metadata, source, start, latest


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def events(binding, index=0, resample=False, crop=True, channels=2):
    prepared_samples = 3202 if resample else 1601
    samples = prepared_samples // 800 * 800 if crop else prepared_samples
    frames = (samples + 799) // 800
    result = [('h3_audio_source', metadata({'waveform': (2, channels, 1601)}, binding=binding,
               reference_index=index, sample_rate=16000 if resample else 32000, target_rate=32000))]
    if resample:
        result.append(('h3_audio_resample', dict(binding=binding, seconds=.01, succeeded=True)))
    result.extend([('h3_audio_prepared', metadata({'pixels': (1, prepared_samples, channels)}, binding=binding)),
        ('h3_audio_wrapper', metadata({'pixels': (1, samples, 2)}, binding=binding, native=True,
            crop_input=crop, crop_ratio=800 if crop else None, output_channels=2, pad_channel_value='replicate')),
        ('h3_audio_plan', metadata({'waveform': (1, 2, samples)}, binding=binding, native=True,
            hop_length=800, sample_rate=32000, right_pad=(-samples) % 800,
            latent_width=32, output_channels=32, heads=8, training=False, grad_enabled=False, inference_mode=True))])
    for phase, shape in [('encoder', (2, 1, frames * 800)), ('projection', (2, frames, 32)), ('mean', (2, 32, frames))]:
        result.extend([('h3_audio_' + phase + '_input', metadata({'input': shape}, binding=binding)),
                       ('h3_audio_' + phase, dict(binding=binding, seconds=.01, succeeded=True))])
    result.extend([(event, metadata({'latents': (1, 32, 2, frames)}, binding=binding))
                   for event in ('h3_audio_output', 'h3_audio_result')])
    return result


def reference_events(index=0):
    facts = source(index, role='audio')
    facts['shapes']['waveform'] = (2, 2, 1601)
    facts['sample_rate'] = 32000
    return [('h3_reference_source', facts), ('h3_reference_output', metadata({'audio': (1, 32, 2, 2)},
        index=index, role='audio', slot='ref_audios.ref_audio_' + str(index)))]


@pytest.mark.parametrize('resample,crop,channels', [(False, True, 2), (True, True, 1), (False, False, 2)])
def test_actual_resample_wrapper_and_encode_phases_define_work_not_outcomes(resample, crop, channels):
    value = state()
    for event, facts in events(object(), resample=resample, crop=crop, channels=channels):
        observe_h3_audio_metrics(value, event, facts)
    assert h3_audio_complete(value)
    assert value['resolved']['managedLoading.required'] is True
    assert value['resolved']['h3AudioVae.0.wrapper.channelWork'] == ('replicate' if channels == 1 else 'identity')
    assert ('h3AudioVae.0.resample.batch' in value['resolved']) is resample
    assert value['resolved']['h3AudioVae.0.encoder.axis.0'] == 2
    assert not any('.output.' in key or '.result.' in key or 'cpuSeconds' in key for key in value['resolved'])
    assert value['measurement_availability']['h3AudioVae.0.result.axis.3'] == 'completion'


@pytest.mark.parametrize('missing', ['prepared', 'wrapper', 'plan', 'encoder_input', 'encoder',
    'projection_input', 'projection', 'mean_input', 'mean', 'output', 'result'])
def test_missing_phase_or_completion_cannot_qualify(missing):
    value = state()
    for event, facts in events(object()):
        if event != 'h3_audio_' + missing:
            observe_h3_audio_metrics(value, event, facts)
    assert not h3_audio_complete(value)


@pytest.mark.parametrize('event,mutate', [('plan', lambda f: f.update(native=False)),
    ('plan', lambda f: f.update(right_pad=True)), ('plan', lambda f: f.update(heads=3)),
    ('plan', lambda f: f.update(training=None)), ('wrapper', lambda f: f.update(crop_ratio=0)),
    ('projection_input', lambda f: f['shapes'].update(input=(2, 3, 32))),
    ('mean', lambda f: f.update(succeeded=False))])
def test_malformed_native_evidence_keeps_explicit_gaps(event, mutate):
    value = state()
    for name, facts in events(object()):
        if name == 'h3_audio_' + event:
            mutate(facts)
        observe_h3_audio_metrics(value, name, facts)
    assert not h3_audio_complete(value) and value['reasons']


def test_reference_indices_source_and_outputs_must_all_agree():
    value = state()
    observe_h3_reference_metrics(value, 'h3_reference_begin', {})
    for event, facts in reference_events():
        observe_h3_reference_metrics(value, event, facts)
    for event, facts in events(object()):
        observe_h3_audio_metrics(value, event, facts)
    assert h3_reference_audio_complete(value)
    for mutation in ('index', 'source', 'output', 'duplicate'):
        other = copy.deepcopy(value)
        op = next(iter(other['h3_audio_operations'].values()))
        if mutation == 'index':
            op['reference_index'] = None
        elif mutation == 'source':
            op['source'] = (1, 2, 1601)
        elif mutation == 'output':
            op['result'] = (1, 32, 2, 3)
        else:
            other['h3_audio_operations'][object()] = copy.deepcopy(op)
        assert not h3_reference_audio_complete(other)


def test_legacy_multiple_call_gap_is_replaced_only_by_complete_scoped_reference_proof():
    value = state()
    observe_h3_reference_metrics(value, 'h3_reference_begin', {})
    for index in range(2):
        for event, facts in reference_events(index):
            observe_h3_reference_metrics(value, event, facts)
        for event, facts in events(object(), index):
            observe_h3_audio_metrics(value, event, facts)
    value['audioProcessed_count'] = 2
    assert scoped_h3_audio_replaces_legacy_limit(value)
    value['audioProcessed_count'] = 3
    assert not scoped_h3_audio_replaces_legacy_limit(value)
    value['audioProcessed_count'], value['audioDecode_count'] = 2, 2
    assert not scoped_h3_audio_replaces_legacy_limit(value)


def test_store_coalesces_phases_and_preserves_loading_and_reference_gaps(tmp_path):
    store = start(tmp_path)
    sequence = events(object())
    store.safe_metric('prompt', 'refs', *sequence[0])
    changes = store.db.total_changes
    for event, facts in sequence[1:-1]:
        store.safe_metric('prompt', 'refs', event, facts)
    assert store.db.total_changes == changes
    store.safe_metric('prompt', 'refs', *sequence[-1])
    work = latest(store)['stages'][1]['workload']
    assert 'h3_audio_work_unmeasured' not in work['reasons']
    assert {'managed_load_plan_unresolved', 'h3_reference_preparation_unmeasured'} <= set(work['reasons'])
    assert not work['complete'] and 'binding' not in repr(work)


def test_limits_duplicates_and_probe_failures_stay_bounded():
    value = state()
    for _ in range(MAX_AUDIO_OPERATIONS + 3):
        observe_h3_audio_metrics(value, *events(object())[0])
    assert len(value['h3_audio_operations']) == MAX_AUDIO_OPERATIONS and value['reasons']
    other = state()
    sequence = events(object())
    for event, facts in sequence + [sequence[-1]]:
        observe_h3_audio_metrics(other, event, facts)
    assert not h3_audio_complete(other)
    observe_h3_audio_metrics(other, 'tensor_shape_unavailable', {'operation': 'h3_audio_output'})
    assert other['reasons'] == ['h3_audio_metadata_unresolved']
