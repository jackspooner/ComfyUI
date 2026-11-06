import pytest

from app.run_timing_h3_vision import MAX_VISION_CALLS, h3_vision_complete, observe_h3_vision_metrics
from app_test.run_timing_h3_references_test import start, latest
from app_test.run_timing_h3_conditioning_test import plan


def state():
    return {'resolved': {'h3Conditioning.imagePositions': 4, 'h3Conditioning.audioPositions': 0, 'h3Conditioning.width': 4},
            'measurements': {}, 'reasons': []}


def metadata(shapes, **facts):
    return dict(shapes=shapes, precisions={key: 'torch.float32' for key in shapes}, devices={key: 'cpu' for key in shapes}, **facts)


def begin(binding, video=False):
    return metadata({'source': (2 if video else 1, 32, 32, 3)}, binding=binding, native=True,
        mode='video' if video else 'image', layers=1, hidden_size=8, heads=1, patch_size=16, merge_size=2,
        temporal_patch_size=2, output_width=4, deepstack=(0,), compute_precision='torch.float32',
        training=False, grad_enabled=False, inference_mode=True)


def geometry(binding, video=False):
    return metadata({'patches': (16, 1536)}, binding=binding, processor='h3_video' if video else 'qwen_image',
        source_frames=2 if video else 1, source_height=32, source_width=32, height=64, width=64,
        grid_t=1, grid_h=4, grid_w=4, patch_size=16, temporal_patch_size=2, merge_size=2, interpolation='bilinear')


def events(binding, reused=False, video=False):
    return [('h3_vision_begin', begin(binding, video)), ('h3_vision_geometry', geometry(binding, video)),
            ('h3_vision_execution', metadata({'patches': (16, 1536)}, binding=binding, native=True,
                reused=reused, cache_eligible=not video, device='cpu')),
            ('h3_vision_output', metadata({'merged': (4, 4), 'deepstack.0': (4, 4)}, binding=binding)),
            ('h3_vision_reuse' if reused else 'h3_vision_forward', {'binding': binding, 'seconds': .01, 'succeeded': True})]


@pytest.mark.parametrize('reused,video', [(False, False), (True, False), (False, True)])
def test_correlated_geometry_and_reuse_are_inputs_but_outputs_and_spans_are_not(reused, video):
    value, binding = state(), object()
    for event, facts in events(binding, reused, video):
        assert observe_h3_vision_metrics(value, event, facts)
    assert h3_vision_complete(value)
    assert value['resolved']['h3Vision.0.tokens'] == 4
    assert value['resolved']['h3Vision.0.reused'] is reused
    assert value['resolved']['h3Vision.0.source_frames'] == (2 if video else 1)
    assert value['measurement_availability']['h3Vision.0.output.merged.axis.0'] == 'completion'
    assert not any('.output.' in key or 'cpuSeconds' in key or 'source.axis.' in key for key in value['resolved'])
    assert not any(type(fact) is object for fact in value['resolved'].values())
    assert value['resolved']['managedLoading.required'] is True


def test_coalesced_stage_snapshot_clears_only_proven_vision_work(tmp_path):
    store, binding = start(tmp_path), object()
    for event, facts in events(binding)[:1]:
        store.safe_metric('prompt', 'refs', event, facts)
    changes = store.db.total_changes
    for event, facts in events(binding)[1:]:
        store.safe_metric('prompt', 'refs', event, facts)
    assert store.db.total_changes == changes
    for event, facts in [('text_embedding_started', {}), ('text_embedding_row',
        dict(positions=5, text=1, image=4, audio=0, unknown=0, padding=0)), ('h3_conditioning_plan', plan())]:
        store.safe_metric('prompt', 'refs', event, facts)
    work = latest(store)['stages'][1]['workload']
    assert 'h3_conditioning_text_work_unmeasured' not in work['reasons']
    assert 'h3_conditioning_vision_work_unmeasured' not in work['reasons']
    assert 'h3_reference_preparation_unmeasured' in work['reasons'] and 'managed_load_plan_unresolved' in work['reasons']
    assert not work['complete']
    assert 'binding' not in repr(work)


def test_missing_or_mismatched_expanded_positions_do_not_invent_coverage():
    for positions in (None, 0, 3, 8):
        value, binding = state(), object()
        value['resolved']['h3Conditioning.imagePositions'] = positions
        for event, facts in events(binding):
            observe_h3_vision_metrics(value, event, facts)
        assert not h3_vision_complete(value)
    value = state()
    value['resolved']['h3Conditioning.imagePositions'] = 0
    assert h3_vision_complete(value)
    value['resolved']['h3Conditioning.audioPositions'] = 1
    assert not h3_vision_complete(value)


@pytest.mark.parametrize('mutate', [lambda f: f.update(native=False), lambda f: f.update(deepstack=(0, 0)),
    lambda f: f.update(deepstack=(1,)), lambda f: f.update(compute_precision='unknown'),
    lambda f: f.update(training=None), lambda f: f['shapes'].update(source=(1, 32, 32))])
def test_invalid_native_metadata_cannot_complete_work(mutate):
    value, binding = state(), object()
    facts = begin(binding)
    mutate(facts)
    observe_h3_vision_metrics(value, 'h3_vision_begin', facts)
    assert value['reasons'] and not h3_vision_complete(value)


@pytest.mark.parametrize('event,mutate', [
    ('h3_vision_geometry', lambda f: f.update(grid_h=5)),
    ('h3_vision_geometry', lambda f: f['shapes'].update(patches=(15, 1536))),
    ('h3_vision_execution', lambda f: f.update(native=False)),
    ('h3_vision_execution', lambda f: f.update(reused=True, cache_eligible=False)),
    ('h3_vision_output', lambda f: f['shapes'].update(merged=(5, 4))),
    ('h3_vision_output', lambda f: f['shapes'].pop('deepstack.0')),
])
def test_malformed_geometry_decisions_and_output_verification_keep_gaps(event, mutate):
    value, binding = state(), object()
    for name, facts in events(binding):
        if name == event:
            mutate(facts)
        observe_h3_vision_metrics(value, name, facts)
    assert value['reasons'] and not h3_vision_complete(value)


def test_duplicate_or_uncorrelated_events_invalidate_once_without_repeated_sql(tmp_path):
    store, binding = start(tmp_path), object()
    store.safe_metric('prompt', 'refs', 'h3_vision_begin', begin(binding))
    store.safe_metric('prompt', 'refs', 'h3_vision_begin', begin(binding))
    changes = store.db.total_changes
    for _ in range(20):
        store.safe_metric('prompt', 'refs', 'h3_vision_begin', begin(binding))
    assert store.db.total_changes == changes
    assert 'h3_vision_multiple_invocations' in latest(store)['stages'][1]['reasons']
    value = state()
    observe_h3_vision_metrics(value, 'h3_vision_geometry', geometry(object()))
    assert 'h3_vision_correlation_unresolved' in value['reasons']


def test_operation_limit_and_probe_failure_stay_explicit():
    value = state()
    for _ in range(MAX_VISION_CALLS + 10):
        observe_h3_vision_metrics(value, 'h3_vision_begin', begin(object()))
    assert len(value['h3_vision_operations']) == MAX_VISION_CALLS
    assert 'h3_vision_call_limit' in value['reasons']
    other, binding = state(), object()
    for event, facts in events(binding):
        observe_h3_vision_metrics(other, event, facts)
    assert h3_vision_complete(other)
    observe_h3_vision_metrics(other, 'tensor_shape_unavailable', {'operation': 'h3_vision_output'})
    assert not h3_vision_complete(other)
