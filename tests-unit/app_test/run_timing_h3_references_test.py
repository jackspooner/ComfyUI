import json
from pathlib import Path

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_h3_references import MAX_REFERENCES, h3_reference_preparation_complete, observe_h3_reference_metrics
from app.run_timing_workload import describe


def metadata(shapes, **facts):
    return dict(shapes=shapes, precisions={key: 'torch.float32' for key in shapes}, devices={key: 'cpu' for key in shapes}, **facts)


def source(index=0, role='image', encode=True, soundtrack=False):
    slot = {'image': 'ref_images.ref_image_', 'video': 'ref_videos.ref_video_', 'audio': 'ref_audios.ref_audio_'}[role] + str(index)
    shapes = {'source': (3, 64, 32, 4) if role == 'image' else (23, 32, 32, 3)} if role != 'audio' else {}
    if role == 'audio' or soundtrack:
        shapes['waveform'] = (2, 2, 16)
    return metadata(shapes, index=index, role=role, slot=slot, resize_width=32, resize_height=32, frame_limit=22,
                    visual_encode=role != 'audio' and encode, audio_encode=(role == 'audio' or soundtrack) and encode,
                    soundtrack=soundtrack, sample_rate=8000)


def processed(index=0, role='image'):
    return metadata({'visual': (1 if role == 'image' else 22, 32, 32, 3),
                     'tokenizer': (1 if role == 'image' else 2, 32, 32, 3)}, index=index, role=role,
                    slot=('ref_images.ref_image_' if role == 'image' else 'ref_videos.ref_video_') + str(index))


def ready(state):
    for event, facts in [('h3_reference_begin', {}), ('h3_reference_source', source()),
                         ('h3_reference_processed', processed()),
                         ('h3_reference_output', metadata({'visual': (1, 24, 2, 2, 2)}, index=0, role='image', slot='ref_images.ref_image_0')),
                         ('h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 1})]:
        observe_h3_reference_metrics(state, event, facts)


def graph(custom=False):
    return {'image': {'class_type': 'LoadImage', 'inputs': {'image': 'private-source.png'}},
            'refs': {'class_type': 'CustomH3' if custom else 'MiniMaxH3ReferenceToVideo',
                     'inputs': {'width': 32, 'height': 32, 'length': 6, 'ref_images.ref_image_0': ['image', 0], 'prompt': 'private prompt'}}}


def start(tmp_path, custom=False):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', graph(custom), {}, ['refs'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'refs', 'started')
    return store


def latest(store):
    return store.page('0', 100, [], [])['observations'][-1]


def test_native_adapter_requires_actual_work_and_preserves_source_relationships():
    evidence, stages = describe(graph(), {'profileRevision': 'fixture'},
        {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: None, digest)
    stage = stages[1]
    assert stage['kind'] == 'preparation' and stage['workload']['profile'] == 'diffusion_video'
    assert stage['workload']['requested'] == {'width': 32, 'height': 32, 'length': 6}
    assert 'unsupported_workload_adapter' not in stage['reasons']
    assert 'h3_reference_preparation_unmeasured' in stage['reasons']
    reference = stage['workload']['components'][0]
    assert reference['role'] == 'refs:ref_images.ref_image_0'
    assert reference['features']['sourceStage']['value'] == 'image' and reference['features']['sourceOutput']['value'] == 0
    assert 'private prompt' not in repr(evidence) and 'private-source.png' not in repr(evidence)


def test_live_preparation_snapshot_is_bounded_and_does_not_complete_encoders(tmp_path):
    store = start(tmp_path)
    store.safe_metric('prompt', 'refs', 'h3_reference_begin', {})
    changes = store.db.total_changes
    for event, facts in [('h3_reference_source', source()), ('h3_reference_processed', processed()),
        ('h3_reference_output', metadata({'visual': (1, 24, 2, 2, 2)}, index=0, role='image', slot='ref_images.ref_image_0')),
        ('h3_reference_resize', {'index': 0, 'seconds': .01, 'succeeded': True})]:
        store.safe_metric('prompt', 'refs', event, facts)
    assert store.db.total_changes == changes
    store.safe_metric('prompt', 'refs', 'h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 1})
    assert store.db.total_changes > changes
    work = latest(store)['stages'][1]['workload']
    assert 'h3_reference_preparation_unmeasured' not in work['reasons']
    assert {'h3_conditioning_text_work_unmeasured', 'h3_reference_encoding_work_unmeasured',
            'h3_allocation_workload_unmeasured'} <= set(work['reasons'])
    assert not work['complete']
    assert work['resolved']['h3References.entry.0.resize.frames']['value'] == 1
    assert work['resolved']['h3References.entry.0.processedFrames']['value'] == 1
    assert work['measurements']['h3References.entry.0.source.source.axis.0']['value'] == 3
    assert work['measurements']['h3References.entry.0.output.visual.axis.2']['availableAt'] == 'completion'
    assert work['measurements']['h3References.resize.cpuSeconds']['value'] == .01
    assert not any('.output.' in key or '.source.source.axis.' in key for key in work['resolved'])
    assert work['components'][0]['features']['processedWidth']['source'] == 'native_h3_reference'
    store.safe_stage('prompt', 'refs', 'finished')
    schema = json.loads((Path(__file__).parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


def test_video_full_resize_trim_tokenizer_sampling_and_paired_audio_are_distinct():
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    observe_h3_reference_metrics(state, 'h3_reference_begin', {})
    observe_h3_reference_metrics(state, 'h3_reference_source', source(role='video', soundtrack=True))
    observe_h3_reference_metrics(state, 'h3_reference_processed', processed(role='video'))
    observe_h3_reference_metrics(state, 'h3_reference_prepared', {'references': 1, 'tokenizer_items': 2, 'blocks': 1})
    assert h3_reference_preparation_complete(state)
    values = state['resolved']
    assert values['h3References.entry.0.resize.frames'] == 23
    assert values['h3References.entry.0.processedFrames'] == 22
    assert values['h3References.entry.0.tokenizerFrames'] == 2
    assert values['h3References.entry.0.audio.batch'] == 1
    assert values['h3References.entry.0.audio.samples'] == 16
    assert values['h3References.entry.0.audio.sampleRate'] == 8000
    assert state['reference_features']['ref_video_audios.ref_video_audio_0']['audio.samples']['value'] == 16
    assert state['reference_features']['ref_videos.ref_video_0']['referenceOrder']['value'] == 0


def test_audio_only_skipped_encoding_metadata_is_audit_only():
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    observe_h3_reference_metrics(state, 'h3_reference_begin', {})
    observe_h3_reference_metrics(state, 'h3_reference_source', source(role='audio', encode=False))
    observe_h3_reference_metrics(state, 'h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 0})
    assert h3_reference_preparation_complete(state)
    assert 'h3References.entry.0.audio.samples' not in state['resolved']
    assert state['measurements']['h3References.entry.0.audio.samples'] == 16
    assert state['measurement_availability']['h3References.entry.0.audio.samples'] == 'live'


def test_equal_packed_shapes_do_not_hide_extra_video_resize_work():
    states = []
    for frames in (23, 24):
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        facts = source(role='video')
        facts['shapes']['source'] = (frames, 32, 32, 3)
        for event, value in [('h3_reference_begin', {}), ('h3_reference_source', facts),
                             ('h3_reference_processed', processed(role='video')),
                             ('h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 1})]:
            observe_h3_reference_metrics(state, event, value)
        assert h3_reference_preparation_complete(state)
        states.append(state)
    assert [state['resolved']['h3References.entry.0.resize.frames'] for state in states] == [23, 24]
    assert [state['resolved']['h3References.entry.0.processedFrames'] for state in states] == [22, 22]


def test_unused_still_batch_items_do_not_fragment_effective_reference_work():
    states = []
    for batch in (1, 9):
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        facts = source()
        facts['shapes']['source'] = (batch, 64, 32, 4)
        for event, value in [('h3_reference_begin', {}), ('h3_reference_source', facts),
                             ('h3_reference_processed', processed()),
                             ('h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 1})]:
            observe_h3_reference_metrics(state, event, value)
        states.append(state)
    assert states[0]['resolved'] == states[1]['resolved']
    assert states[0]['measurements'] != states[1]['measurements']


@pytest.mark.parametrize('event,facts', [
    ('h3_reference_processed', metadata({'visual': (21, 32, 32, 3), 'tokenizer': (2, 32, 32, 3)},
        index=0, role='video', slot='ref_videos.ref_video_0')),
    ('h3_reference_prepared', {'references': 1, 'tokenizer_items': 99, 'blocks': 1}),
    ('h3_reference_resize', {'index': 0, 'seconds': float('nan'), 'succeeded': True}),
    ('h3_reference_resize', {'index': 10, 'seconds': .1, 'succeeded': True}),
])
def test_mismatched_processed_work_summary_or_timer_is_explicit(event, facts):
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    observe_h3_reference_metrics(state, 'h3_reference_begin', {})
    observe_h3_reference_metrics(state, 'h3_reference_source', source(role='video'))
    observe_h3_reference_metrics(state, event, facts)
    assert state['reasons'] and not h3_reference_preparation_complete(state)


@pytest.mark.parametrize('event,facts', [('h3_reference_begin', {}), ('h3_reference_source', source()),
    ('h3_reference_processed', processed()), ('h3_reference_prepared', {'references': 1, 'tokenizer_items': 1, 'blocks': 1}),
    ('h3_reference_metadata_unavailable', {}), ('tensor_shape_unavailable', {'operation': 'h3_reference_source'})])
def test_repeated_late_or_unavailable_evidence_never_stays_complete(event, facts):
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    ready(state)
    assert h3_reference_preparation_complete(state)
    observe_h3_reference_metrics(state, event, facts)
    assert not h3_reference_preparation_complete(state)


@pytest.mark.parametrize('mutate', [lambda f: f.update(index=1), lambda f: f.update(index=True),
    lambda f: f.update(slot='private filename'), lambda f: f.update(visual_encode=1), lambda f: f.update(role=[]),
    lambda f: f['shapes'].update(source=(1, 32, 32, 2)), lambda f: f['devices'].clear()])
def test_malformed_source_plans_remain_incomplete(mutate):
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    observe_h3_reference_metrics(state, 'h3_reference_begin', {})
    facts = source()
    mutate(facts)
    observe_h3_reference_metrics(state, 'h3_reference_source', facts)
    assert state['reasons'] and not h3_reference_preparation_complete(state)
    assert not state['h3_reference_entries']


def test_reference_count_is_bounded_without_silent_truncation():
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    observe_h3_reference_metrics(state, 'h3_reference_begin', {})
    for index in range(MAX_REFERENCES + 20):
        observe_h3_reference_metrics(state, 'h3_reference_source', source(index, role='audio', encode=False))
    assert len(state['h3_reference_entries']) == MAX_REFERENCES
    assert 'h3_reference_limit_or_socket_unresolved' in state['reasons']
    assert not h3_reference_preparation_complete(state)


def test_unknown_custom_and_cached_nodes_keep_coverage_gaps(tmp_path):
    store = start(tmp_path, custom=True)
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    ready(state)
    for event, facts in [('h3_reference_begin', {}), ('h3_reference_prepared', {'references': 0, 'tokenizer_items': 0, 'blocks': 0})]:
        store.safe_metric('prompt', 'refs', event, facts)
    assert 'unsupported_workload_adapter' in latest(store)['stages'][1]['workload']['reasons']
    cached = start(tmp_path / 'cached')
    cached.safe_cache('prompt', ['refs'])
    work = latest(cached)['stages'][1]['workload']
    assert not work['complete'] and 'h3_reference_preparation_unmeasured' in work['reasons']
