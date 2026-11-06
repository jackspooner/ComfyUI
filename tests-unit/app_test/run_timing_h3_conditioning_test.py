import copy

import pytest

from app.run_timing_h3_conditioning import h3_conditioning_complete, observe_h3_conditioning_metrics
from app.run_timing_text import observe_text_metrics
from app_test.run_timing_h3_references_test import start, latest


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def plan():
    return dict(shapes={'input': (1, 5, 4)}, precisions={'input': 'torch.float32'}, devices={'input': 'cpu'},
                native=True, input_kind='embeddings', layers=2, compute_precision='torch.float32', output_precision='torch.float32',
                selection='last', indices=(), layer_norm=False, attention_mask=False, extra_options=False)


def rows(value, image=0):
    observe_text_metrics(value, 'text_embedding_started', {})
    observe_text_metrics(value, 'text_embedding_row', dict(positions=5, text=4-image, image=image, audio=0, unknown=0, padding=1))


def test_expanded_input_metadata_is_live_and_durations_are_measurements():
    value = state()
    rows(value)
    observe_h3_conditioning_metrics(value, 'h3_conditioning_plan', plan())
    assert h3_conditioning_complete(value)
    assert value['resolved']['h3Conditioning.positions'] == 5
    assert value['resolved']['h3Conditioning.textPositions'] == 4
    assert value['resolved']['h3Conditioning.paddingPositions'] == 1
    assert value['resolved']['managedLoading.required'] is True
    assert value['availability']['h3Conditioning.positions'] == 'live'
    observe_h3_conditioning_metrics(value, 'h3_conditioning_forward', {'seconds': .01, 'succeeded': True})
    assert value['measurements']['h3Conditioning.forward.cpuSeconds'] == .01
    assert not any('cpuSeconds' in key or 'completed' in key for key in value['resolved'])


def test_no_vae_selection_clears_only_proven_encoding_work(tmp_path):
    store = start(tmp_path)
    store.safe_metric('prompt', 'refs', 'h3_reference_begin', {})
    store.safe_metric('prompt', 'refs', 'h3_reference_prepared', {'references': 0, 'tokenizer_items': 0, 'blocks': 0})
    work = latest(store)['stages'][1]['workload']
    assert 'h3_reference_encoding_work_unmeasured' not in work['reasons']
    assert 'h3_conditioning_text_work_unmeasured' in work['reasons']
    store.safe_metric('prompt', 'refs', 'text_embedding_started', {})
    store.safe_metric('prompt', 'refs', 'text_embedding_row', dict(positions=5, text=4, image=0, audio=0, unknown=0, padding=1))
    store.safe_metric('prompt', 'refs', 'h3_conditioning_plan', plan())
    work = latest(store)['stages'][1]['workload']
    assert 'h3_conditioning_text_work_unmeasured' not in work['reasons']
    assert 'managed_load_plan_unresolved' in work['reasons']
    assert not work['complete']
    store.safe_metric('prompt', 'refs', 'h3_reference_begin', {})
    assert 'h3_reference_encoding_work_unmeasured' in latest(store)['stages'][1]['workload']['reasons']


def test_vision_positions_remain_an_independent_coverage_gap(tmp_path):
    store = start(tmp_path)
    for event, facts in [('text_embedding_started', {}), ('text_embedding_row',
        dict(positions=5, text=2, image=2, audio=0, unknown=0, padding=1)), ('h3_conditioning_plan', plan())]:
        store.safe_metric('prompt', 'refs', event, facts)
    work = latest(store)['stages'][1]['workload']
    assert 'h3_conditioning_text_work_unmeasured' not in work['reasons']
    assert 'h3_conditioning_vision_work_unmeasured' in work['reasons']
    assert not work['complete']


@pytest.mark.parametrize('mutate', [lambda f: f.update(native=False), lambda f: f.update(input_kind='input_ids'),
    lambda f: f.update(extra_options=True), lambda f: f.update(attention_mask=True), lambda f: f.update(layers=0),
    lambda f: f.update(selection='custom'), lambda f: f.update(indices=tuple(range(65))),
    lambda f: f.update(compute_precision='unknown'), lambda f: f['shapes'].update(input=(2, 5, 4))])
def test_custom_malformed_or_unmatched_plans_keep_gaps(mutate):
    value, facts = state(), plan()
    rows(value)
    mutate(facts)
    observe_h3_conditioning_metrics(value, 'h3_conditioning_plan', facts)
    assert value['reasons'] and not h3_conditioning_complete(value)


def test_missing_unknown_or_over_limit_modality_rows_never_complete_work():
    for bad_rows in (None, [], [{'positions': 5, 'text': 3, 'image': 0, 'audio': 0, 'unknown': 1, 'padding': 1}]):
        value = state()
        value['embedding_rows'] = copy.deepcopy(bad_rows)
        observe_h3_conditioning_metrics(value, 'h3_conditioning_plan', plan())
        assert 'h3_conditioning_modalities_unresolved' in value['reasons']
        assert not h3_conditioning_complete(value)


def test_repetition_and_probe_failure_invalidate_resolved_work_without_growing_plans():
    value = state()
    rows(value)
    observe_h3_conditioning_metrics(value, 'h3_conditioning_plan', plan())
    size = len(value['resolved'])
    for _ in range(20):
        rows(value)
        observe_h3_conditioning_metrics(value, 'h3_conditioning_plan', plan())
    assert len(value['resolved']) == size and not h3_conditioning_complete(value)
    assert 'h3_conditioning_multiple_invocations' in value['reasons']
    other = state()
    rows(other)
    observe_h3_conditioning_metrics(other, 'h3_conditioning_plan', plan())
    observe_h3_conditioning_metrics(other, 'tensor_shape_unavailable', {'operation': 'h3_conditioning_plan'})
    assert not h3_conditioning_complete(other)
