import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_vision import observe_moge_metrics
from app.run_timing_workload import describe, resolved_inputs


def graph():
    return {
        'image': {'class_type': 'LoadImage', 'inputs': {'image': '/PRIVATE/source.png'}},
        'model': {'class_type': 'LoadMoGeModel', 'inputs': {'model_name': '/PRIVATE/moge.safetensors'}},
        'detail': {'class_type': 'PrimitiveInt', 'inputs': {'value': 6}},
        'infer': {'class_type': 'MoGeInference', 'inputs': {'image': ['image', 0], 'moge_model': ['model', 0],
            'resolution_level': ['detail', 0], 'refine_steps': 8, 'batch_size': 99, 'force_projection': True}},
        'save': {'class_type': 'SaveMoGeDepthPNG', 'inputs': {'moge_geometry': ['infer', 0], 'filename_prefix': '/PRIVATE/output'}},
    }


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def plan(batch=1, version='v2', passes=0):
    return {'shape': (batch, 3, 4, 6), 'version': version, 'num_tokens': 2800, 'refine_steps': passes,
            'precision': 'torch.float16', 'activation_precision': 'torch.float16' if version == 'v3' else 'torch.float32', 'load_device': 'cpu',
            'offload_device': 'cpu', 'memory_strategy': 'standard', 'force_projection': True,
            'apply_mask': True, 'apply_metric_scale': True, 'fov_x': None}


def test_explicit_moge_selections_keep_requests_out_of_effective_work_and_paths_private():
    resource = {'hostId': 'host', 'gpuUuid': 'GPU-fixture', 'instanceId': 'boot', 'laneId': 'comfyui'}
    calls = []
    def identity(key, value):
        calls.append((key, value))
        return digest([key, value])
    evidence, stages = describe(graph(), {'profileRevision': 'fixture'}, resource, 'now', identity, digest)
    assert evidence['profile'] == 'vision'
    assert evidence['extractorVersion'] == 'comfy-explicit-6'
    assert calls == [('moge_model_name', '/PRIVATE/moge.safetensors')]
    stage = next(stage for stage in stages if stage['id'] == 'infer')
    assert stage['kind'] == 'inference' and stage['workload']['profile'] == 'vision'
    assert stage['workload']['resolved'] == {}
    assert evidence['requested']['infer.resolution_level'] == 6
    assert evidence['requested']['infer.refine_steps'] == 8
    assert 'moge_workload_unmeasured' in stage['workload']['reasons']
    assert stage['workload']['components'][0]['features']['sourceStage']['value'] == 'image'
    assert '/PRIVATE' not in json.dumps([evidence, stages])
    facts, refs = resolved_inputs('MoGeInference', {'image': SimpleNamespace(shape=(1, 4, 6, 4)), 'batch_size': 99}, 'now')
    assert facts == {} and refs[0][1]['width']['value'] == 6
    unknown = {'node': {'class_type': 'LooksLikeMoGe', 'inputs': {'model_name': '/PRIVATE/unknown'}}}
    missing, stages = describe(unknown, {'profileRevision': 'fixture'}, resource, 'now', identity, digest)
    assert missing['profile'] == 'unknown' and stages[0]['kind'] == 'unknown'
    assert len(calls) == 1
    malformed = {'infer': {'class_type': 'MoGeInference', 'inputs': {
        'image': ['/PRIVATE/not-a-node', 0], 'refine_steps': '/PRIVATE/not-a-number'}}}
    malformed_evidence, malformed_stages = describe(malformed, {'profileRevision': 'fixture'}, resource, 'now', identity, digest)
    assert '/PRIVATE' not in json.dumps([malformed_evidence, malformed_stages])


def test_effective_chunk_limit_is_normalized_and_ignored_refinement_is_not_matching_work():
    value = state()
    observe_moge_metrics(value, 'moge_node_plan', {'shape': (1, 3, 4, 6), 'views': 1, 'batch_size': 99, 'operation': 'perspective'})
    observe_moge_metrics(value, 'moge_model_plan', plan())
    observe_moge_metrics(value, 'moge_processed_input', {'shape': (1, 3, 28, 42)})
    assert value['moge_workload_complete'] is True
    assert value['resolved']['moge.chunkSize'] == 1
    assert value['resolved']['mogeModel.0.refineSteps'] == 0
    assert value['resolved']['mogeModel.0.tokens'] == 2800
    assert value['availability']['mogeModel.0.processed.axis.2'] == 'live'


def test_live_chunk_work_clears_only_local_gaps_and_keeps_completed_seconds_as_measurements(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='host', gpu_uuid='GPU-fixture', now=lambda: 0)
    item = (1, 'prompt', graph(), {}, ['save'])
    with patch('app.run_timing.selected_file_identity', return_value='weights-fixture'):
        store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'infer', 'started')
    store.safe_inputs('prompt', 'infer', 'MoGeInference', {'image': SimpleNamespace(shape=(3, 4, 6, 3))})
    store.safe_metric('prompt', 'infer', 'moge_node_plan', {'shape': (3, 3, 4, 6), 'views': 3, 'batch_size': 2, 'operation': 'perspective'})
    for index, batch in enumerate((2, 1)):
        store.safe_metric('prompt', 'infer', 'moge_load', {'seconds': .1, 'succeeded': True})
        store.safe_metric('prompt', 'infer', 'moge_model_plan', plan(batch))
        store.safe_metric('prompt', 'infer', 'moge_processed_input', {'shape': (batch, 3, 28, 42)})
        changes = store.db.total_changes
        store.safe_metric('prompt', 'infer', 'moge_forward', {'seconds': .25, 'succeeded': True})
        assert store.db.total_changes == changes
        record = store.page('0', 100, [item], [])['active']['records'][0]
        stage = next(stage for stage in record['stages'] if stage['id'] == 'infer')
        assert stage['workload']['complete'] is (index == 1)
    assert stage['workload']['resolved']['executionPrecision']['source'] == 'native_moge'
    assert stage['workload']['resolved']['mogeModel.1.input.axis.0']['value'] == 1
    assert stage['workload']['components'][0]['features']['sourceStage']['value'] == 'image'
    assert 'moge_workload_unmeasured' not in stage['workload']['reasons']
    assert not record['workload']['evidence']['complete']
    assert 'reference_processing_unmeasured' in record['workload']['evidence']['reasons']
    assert 'moge_publication_input_unmeasured' in record['workload']['evidence']['reasons']
    store.safe_stage('prompt', 'infer', 'finished')
    store.safe_stage('prompt', 'save', 'started')
    store.safe_metric('prompt', 'save', 'moge_export_input', {'shape': (1, 4, 6), 'mask_shape': (1, 4, 6)})
    record = store.page('0', 100, [item], [])['active']['records'][0]
    stage = next(stage for stage in record['stages'] if stage['id'] == 'infer')
    assert stage['workload']['measurements']['mogeModel.1.forwardSeconds']['value'] == .25
    assert stage['workload']['measurements']['mogeModel.1.forwardSeconds']['availableAt'] == 'completion'
    assert 'mogeModel.1.forwardSeconds' not in stage['workload']['resolved']
    assert next(stage for stage in record['stages'] if stage['id'] == 'save')['workload']['complete']
    assert '/PRIVATE' not in json.dumps(record)
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [item], []), schema['$defs']['TimingObservationPage'])


def test_refinement_spatial_work_becomes_known_only_at_each_actual_entry_boundary():
    value = state()
    observe_moge_metrics(value, 'moge_node_plan', {'shape': (1, 3, 4, 6), 'views': 1, 'batch_size': 1, 'operation': 'perspective'})
    observe_moge_metrics(value, 'moge_model_plan', plan(version='v3', passes=2))
    observe_moge_metrics(value, 'moge_processed_input', {'shape': (1, 3, 28, 42)})
    assert not value['moge_workload_complete']
    for index, depth in enumerate((5, 7)):
        observe_moge_metrics(value, 'moge_refinement_plan', {'points': 6, 'spatial': (2, 3, depth), 'conditioning_shape': (1, 2, 2, 3)})
        assert value['moge_workload_complete'] is (index == 1)
    assert value['resolved']['mogeModel.0.refine.1.spatial.axis.2'] == 7
    assert value['availability']['mogeModel.0.refine.1.spatial.axis.2'] == 'live'
    observe_moge_metrics(value, 'moge_refinement_plan', {'points': 6, 'spatial': (2, 3, 9), 'conditioning_shape': (1, 2, 2, 3)})
    assert not value['moge_workload_complete'] and 'moge_refinement_layout_unresolved' in value['reasons']


@pytest.mark.parametrize('change', [{'precision': 'unknown'}, {'memory_strategy': 'unknown'},
                                  {'activation_precision': 'torch.bfloat16'}, {'version': 'v1', 'refine_steps': 1}])
def test_unknown_precision_regimes_and_unreviewed_refinement_paths_are_not_complete(change):
    value = state()
    observe_moge_metrics(value, 'moge_node_plan', {'shape': (1, 3, 4, 6), 'views': 1, 'batch_size': 1, 'operation': 'perspective'})
    observe_moge_metrics(value, 'moge_model_plan', {**plan(), **change})
    observe_moge_metrics(value, 'moge_processed_input', {'shape': (1, 3, 28, 42)})
    assert not value['moge_workload_complete'] and value['reasons']


@pytest.mark.parametrize('event,facts', [
    ('moge_model_plan', {'shape': (1, 3, 4, 6)}),
    ('moge_node_plan', {'shape': (1, 3, 4), 'views': 1, 'batch_size': 1, 'operation': 'perspective'}),
    ('moge_processed_input', {'shape': (1, 3, 28, 42)}),
    ('moge_export_input', {'shape': (1, 4, 6), 'mask_shape': None}),
])
def test_malformed_or_unattributed_boundaries_remain_incomplete(event, facts):
    value = state()
    assert observe_moge_metrics(value, event, facts)
    assert value['reasons'] and not value.get('moge_workload_complete')


def test_limits_duplicate_chunks_and_panorama_merge_do_not_acquire_coverage():
    value = state()
    observe_moge_metrics(value, 'moge_node_plan', {'shape': (17, 3, 4, 6), 'views': 17, 'batch_size': 1, 'operation': 'perspective'})
    for _ in range(18):
        observe_moge_metrics(value, 'moge_model_plan', plan())
    assert 'moge_invocation_limit' in value['reasons']
    assert len(value['moge_model_work']) == 16
    assert not any(key.startswith('mogeModel.16.') for key in value['resolved'])
    value = state()
    observe_moge_metrics(value, 'moge_node_plan', {'shape': (1, 3, 4, 6), 'views': 1, 'batch_size': 1, 'operation': 'perspective'})
    observe_moge_metrics(value, 'moge_model_plan', plan())
    observe_moge_metrics(value, 'moge_processed_input', {'shape': (1, 3, 28, 42)})
    assert value['moge_workload_complete']
    observe_moge_metrics(value, 'moge_load', {'seconds': .1, 'succeeded': True})
    assert not value['moge_workload_complete'] and 'moge_chunk_layout_unresolved' in value['reasons']
    source = graph()
    source['infer']['class_type'] = 'MoGePanoramaInference'
    _, stages = describe(source, {'profileRevision': 'fixture'},
        {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: 'weights', digest)
    assert 'moge_panorama_merge_work_unmeasured' in next(stage for stage in stages if stage['id'] == 'infer')['workload']['reasons']
