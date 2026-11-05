import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from app import run_timing
from app.run_timing_model_loading import observe_model_loading_metrics, managed_loading_complete


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def plan(operation, models=1):
    return {'operation': operation, 'models': models, 'force_patch_weights': False, 'force_full_load': False,
            'requested_bytes': 1000, 'minimum_bytes': 2000}


def entries(operation, index=0, loaded=0):
    return [('managed_load_selection', {'operation': operation, 'index': index, 'reused': True, 'dynamic': False, 'device': 'cuda:0'}),
            ('managed_load_clones', {'operation': operation, 'index': index, 'count': 0}),
            ('managed_load_requirement', {'operation': operation, 'index': index, 'bytes': 1000}),
            ('managed_load_entry_plan', {'operation': operation, 'index': index, 'vram_state': 'NORMAL_VRAM',
                                         'loaded_bytes': loaded, 'budget_bytes': 10000})]


@pytest.mark.parametrize('loaded', [None, 0, 512])
def test_input_plan_and_diagnostics_are_separate_and_unknown_is_not_warm(loaded):
    value, operation = state(), object()
    observe_model_loading_metrics(value, 'managed_load_plan', plan(operation))
    assert not managed_loading_complete(value)
    for event, facts in entries(operation, loaded=loaded):
        observe_model_loading_metrics(value, event, facts)
    assert managed_loading_complete(value)
    prefix = 'managedLoading.0.entry.0.'
    assert value['resolved'][prefix + 'managerLoadedBytes'] == loaded
    assert value['resolved'][prefix + 'managerLoadedBytesKnown'] is (loaded is not None)
    assert value['resolved'][prefix + 'managerEntryReused'] is True
    assert 'residency' not in value['resolved'] and 'cache' not in value['resolved']
    assert prefix + 'requiredBytesBeforePressure' not in value['resolved']
    assert value['measurements'][prefix + 'requiredBytesBeforePressure'] == 1000
    assert value['resolved'][prefix + 'weightLoadBudgetBytes'] == 10000
    assert value['availability'][prefix + 'managerLoadedBytes'] == 'live'


@pytest.mark.parametrize('event,change', [
    ('managed_load_selection', {'reused': 1}), ('managed_load_selection', {'device': ''}),
    ('managed_load_clones', {'count': -1}), ('managed_load_requirement', {'bytes': float('nan')}),
    ('managed_load_entry_plan', {'vram_state': 'custom'}), ('managed_load_entry_plan', {'loaded_bytes': -1}),
    ('managed_load_entry_plan', {'budget_bytes': float('inf')}), ('managed_load_entry_plan', {'index': True}),
])
def test_invalid_metadata_cannot_qualify_an_input_plan(event, change):
    value, operation = state(), object()
    observe_model_loading_metrics(value, 'managed_load_plan', plan(operation))
    for name, facts in entries(operation):
        observe_model_loading_metrics(value, name, {**facts, **change} if name == event else facts)
    assert not managed_loading_complete(value)
    assert value['reasons'] == ['managed_load_metadata_unresolved']


def test_device_probe_failure_duplicates_limits_and_missing_correlations_are_explicit():
    class BadDevice:
        def __str__(self):
            raise OSError('metadata unavailable')
    value, operation = state(), object()
    observe_model_loading_metrics(value, 'managed_load_plan', plan(operation))
    event, facts = entries(operation)[0]
    observe_model_loading_metrics(value, event, {**facts, 'device': BadDevice()})
    assert not managed_loading_complete(value)
    for facts in (plan(object(), models=65), plan([], models=1)):
        value = state()
        observe_model_loading_metrics(value, 'managed_load_plan', facts)
        assert not managed_loading_complete(value)
    value = state()
    for _ in range(16):
        observe_model_loading_metrics(value, 'managed_load_plan', plan(object(), models=0))
    assert managed_loading_complete(value)
    observe_model_loading_metrics(value, 'managed_load_plan', plan(object(), models=0))
    assert len(value['managed_load_operations']) == 16 and not managed_loading_complete(value)
    value = state()
    observe_model_loading_metrics(value, 'managed_load_plan', plan(operation))
    for event, facts in entries(operation):
        observe_model_loading_metrics(value, event, facts)
    observe_model_loading_metrics(value, *entries(operation)[0])
    assert not managed_loading_complete(value)


def test_measurements_keep_failure_and_partial_unload_results_without_input_facts():
    value = state()
    for event, facts in [('managed_model_load', {'seconds': .3, 'succeeded': True}),
                         ('managed_model_load', {'seconds': .2, 'succeeded': False}),
                         ('managed_model_unload', {'seconds': .1, 'succeeded': True, 'requested_bytes': 400}),
                         ('managed_model_unload_result', {'detached': False}),
                         ('managed_model_clone_detach', {'seconds': .2, 'succeeded': True})]:
        observe_model_loading_metrics(value, event, facts)
    assert value['resolved'] == {}
    assert value['measurements']['managedLoading.load.seconds'] == .5
    assert value['measurements']['managedLoading.load.completed'] == value['measurements']['managedLoading.load.failed'] == 1
    assert value['measurements']['managedLoading.unload.retained'] == 1


def test_signed_native_reclaim_budget_is_known_without_inventing_transfer_bytes():
    value, operation = state(), object()
    observe_model_loading_metrics(value, 'managed_load_plan', plan(operation))
    for event, facts in entries(operation, loaded=512):
        observe_model_loading_metrics(value, event, {**facts, 'budget_bytes': -100} if event == 'managed_load_entry_plan' else facts)
    assert managed_loading_complete(value)
    assert value['resolved']['managedLoading.0.entry.0.weightLoadBudgetBytes'] == -100


def test_declared_loading_requirement_cannot_be_satisfied_by_sampler_geometry_alone(tmp_path):
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: 0.)
    item = (1, 'fixture', {'node': {'class_type': 'SamplerCustomAdvanced', 'inputs': {}}}, {}, ['node'])
    try:
        store.event('accepted', item)
        store.event('started', item)
        store.safe_stage('fixture', 'node', 'started')
        store.safe_metric('fixture', 'node', 'sampling_plan', {'shape': (1, 4, 2, 2), 'evaluations': 2, 'branches': 1,
                                                           'precision': 'float32', 'managed_load_required': True})
        record = store.page('0', 100, [], [])['observations'][-1]
        assert 'managed_load_plan_unresolved' in record['stages'][0]['reasons']
        assert not record['stages'][0]['workload']['complete']
        fact = record['stages'][0]['workload']['resolved']['managedLoading.required']
        assert fact['source'] == 'native_sampler' and fact['availableAt'] == 'execution_start'
    finally:
        store.db.close()


def test_persistence_coalesces_entry_writes_preserves_preplan_snapshots_and_cleans_bindings(tmp_path):
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: 0.)
    graph = {'node': {'class_type': 'PrimitiveInt', 'inputs': {'value': 1}}}
    item, operation = (1, 'fixture', graph, {}, ['node']), object()
    try:
        store.event('accepted', item)
        store.event('started', item)
        store.safe_stage('fixture', 'node', 'started')
        def emit(event, facts):
            store.safe_metric('fixture', 'node', event, facts)
        def current():
            return store.page('0', 100, [], [])['observations'][-1]
        emit('managed_load_plan', plan(operation, models=2))
        pending, writes = current(), store.db.total_changes
        assert not pending['stages'][0]['workload']['complete']
        for event, facts in entries(operation, 0):
            emit(event, facts)
        assert store.db.total_changes == writes
        for event, facts in entries(operation, 1):
            emit(event, facts)
        assert store.db.total_changes > writes and current()['stages'][0]['workload']['complete']
        assert not pending['stages'][0]['workload']['complete']
        emit('managed_model_load', {'seconds': .5, 'succeeded': True})
        for index in (0, 1):
            emit('managed_load_usage', {'operation': operation, 'index': index, 'ram_bytes': 1000, 'vram_bytes': 500})
        record = current()
        assert record['workload']['residency'] == 'unknown' and record['workload']['cache'] == 'unknown'
        assert 'managed_load_operations' not in json.dumps(record)
        assert record['stages'][0]['workload']['measurements']['managedLoading.load.seconds']['availableAt'] == 'live'
        schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
        jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])
        emit('managed_load_plan', {'operation': [], **{key: value for key, value in plan(operation).items() if key != 'operation'}})
        assert 'managed_load_metadata_unresolved' in current()['stages'][0]['reasons']
        store.safe_stage('fixture', 'node', 'finished')
        store.event('finished', item, SimpleNamespace(completed=True, messages=[]))
        assert not store.metric_states
    finally:
        store.db.close()
