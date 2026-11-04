import json
from pathlib import Path
from types import SimpleNamespace

from app.run_timing import RunTimingStore, workload
from app.run_timing_http import local_request
from app.run_timing import WORKLOAD_RESOLVERS
import jsonschema


def item(prompt='p1', steps=20):
    return (1, prompt, {'a': {'class_type': 'Loader', 'inputs': {'ckpt_name': '/secret/model.safetensors'}},
                        'b': {'class_type': 'Sampler', 'inputs': {'steps': steps, 'prompt': 'private text', 'seed': 123}}}, {}, ['b'])


def test_monotonic_phases_and_durable_cursor(tmp_path):
    now = [0.0]
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='host', now=lambda: now[0])
    store.event('accepted', item())
    now[0] = 7
    store.event('started', item())
    now[0] = 17
    store.event('finished', item(), SimpleNamespace(completed=True, messages=[]))
    page = store.page('0', 2, [], [])
    assert page['nextCursor'] == '2'
    result = store.page(page['cursor'], 2, [], [])['observations'][0]
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])
    assert result['durations'] == {'queueSeconds': 7, 'executionSeconds': 10, 'publicationSeconds': None}
    assert result['phase'] == 'succeeded'
    assert 'private text' not in json.dumps(result)
    assert '/secret/' not in json.dumps(result)
    assert 'seed' not in json.dumps(result)
    assert store.page('3', 10, [], [])['observations'] == []


def test_restart_does_not_invent_completion(tmp_path):
    filename = str(tmp_path / 'timing.sqlite')
    first = RunTimingStore(filename)
    first.event('accepted', item())
    first.event('started', item())
    second = RunTimingStore(filename)
    records = second.page('0', 100, [], [])['observations']
    assert records[-1]['phase'] == 'unknown'
    assert records[-1]['durations']['executionSeconds'] is None
    assert second.source_id == first.source_id
    assert second.instance_id != first.instance_id


def test_queue_order_and_telemetry_failure(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    for prompt in ('p1', 'p2'):
        store.event('accepted', item(prompt))
    page = store.page('0', 100, [item('p1')], [item('p2')])
    assert [row['queueOrder'] for row in page['active']['records']] == [0, 1]
    store.db.close()
    store.safe_event('started', item())
    assert store.issue == 'timing_store_unavailable'


def test_features_are_parameter_sensitive_but_prompt_independent():
    assert workload(item()[2]) != workload(item(steps=40)[2])
    graph = item()[2]
    graph['b']['inputs'].update(prompt='same length!', seed=999)
    assert workload(graph) == workload(item()[2])


def test_selected_file_drift_changes_identity_and_repeated_units_keep_profile(tmp_path, monkeypatch):
    import folder_paths
    filename = tmp_path / 'weights.fixture'
    filename.write_bytes(b'one')
    monkeypatch.setattr(folder_paths, 'get_full_path', lambda category, name: str(filename))
    first = workload(item()[2])
    assert first['settings']['artifactAssurance'] == 'selected_file_identity'
    filename.write_bytes(b'two-longer')
    second = workload(item()[2])
    assert second['model'] != first['model']
    graph = item()[2]
    graph['c'] = {'class_type': 'Loader', 'inputs': {'ckpt_name': '/secret/model.safetensors'}}
    repeated = workload(graph)
    assert repeated['model'] == second['model']
    assert repeated['profileRevision'] == second['profileRevision']
    assert repeated['features']['Loader.nodeInstances'] == 2


def test_browser_and_remote_requests_are_rejected():
    assert local_request(SimpleNamespace(remote='127.0.0.1', host='localhost:8188', headers={}))
    assert not local_request(SimpleNamespace(remote='192.168.1.2', host='localhost', headers={}))
    assert not local_request(SimpleNamespace(remote='127.0.0.1', host='localhost', headers={'Origin': 'http://localhost'}))


def test_duplicate_completion_and_read_only_page(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    store.event('accepted', item())
    store.event('started', item())
    done = SimpleNamespace(completed=True, messages=[])
    store.event('finished', item(), done)
    store.event('finished', item(), done)
    before = store.db.total_changes
    assert len(store.page('0', 100, [], [])['observations']) == 3
    assert store.db.total_changes == before


def test_cache_evidence_is_available_while_running_and_failure_is_nonfatal(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    store.event('accepted', item())
    store.event('started', item())
    store.safe_cache('p1', ['a'])
    assert store.page('0', 100, [item()], [])['active']['records'][0]['workload']['cache'] == 'partial'
    store.db.close()
    store.safe_cache('p1', [])
    assert store.issue == 'timing_cache_observation_unavailable'


def test_deferred_stage_description_and_numeric_text_evidence(tmp_path):
    available = [False]
    def resolver(graph, phase):
        if not available[0]:
            raise ValueError('selection not saved')
        return {'workload': workload(graph), 'relatedOperationIds': ['batch-child']}
    WORKLOAD_RESOLVERS.append(resolver)
    try:
        store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
        store.event('accepted', item())
        assert 'workload_resolution_unavailable' in store.page('0', 10, [], [item()])['active']['records'][0]['coverageReasons']
        available[0] = True
        before = store.db.total_changes
        active = store.page('0', 10, [], [item()])['active']['records'][0]
        assert active['relatedOperationIds'] == ['batch-child']
        assert store.db.total_changes == before
        store.event('started', item())
        store.numeric_evidence('p1', {'inputTokens': 100, 'outputTokens': 40, 'prompt': 'private', 'outputAllowance': 512})
        store.event('finished', item(), SimpleNamespace(completed=True, messages=[]))
        final = store.page('0', 100, [], [])['observations'][-1]
        assert final['workload']['features']['observed.outputTokens'] == 40
        assert final['coverageReasons'] == []
        assert 'private' not in json.dumps(final)
    finally:
        WORKLOAD_RESOLVERS.remove(resolver)
