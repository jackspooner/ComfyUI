import copy
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch

import jsonschema

from app.run_timing import RunTimingStore, digest
from app.run_timing_workload import describe, resolved_inputs


def test_composite_graph_survives_acceptance_and_nested_metrics_keep_execution_context(tmp_path):
    from app import run_timing
    from app.run_timing_metrics import attach_runtime_metrics
    from comfy import execution_metrics
    from comfy_execution.utils import CurrentNodeContext, get_executing_context
    expanded = {'composite.sample': graph()['sample']}
    expanded['composite.sample']['inputs'] = {'steps': 2}
    ready = [False]
    def resolver(*_):
        if not ready[0]:
            raise ValueError('bounded selection has not been saved')
        return {'workload': run_timing.workload(expanded), 'stageGraph': expanded,
                'stageMetadata': {'composite.sample': {'sharedOperationId': 'shared-sample'}}}
    now = [0.0]
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: now[0])
    item = (1, 'prompt', {'outer': {'class_type': 'UnknownComposite', 'inputs': {}}}, {}, ['outer'])
    with patch.object(run_timing, 'WORKLOAD_RESOLVERS', [resolver]):
        store.event('accepted', item)
        store.event('started', item)
        attach_runtime_metrics(SimpleNamespace(timing=store, mutex=Lock(), currently_running={0: item}))
        ready[0] = True
        with CurrentNodeContext('prompt', 'outer'):
            with execution_metrics.stage_scope('composite.sample', 'KSampler', {'steps': 2}):
                assert get_executing_context().node_id == 'outer'
                now[0] = 3
                execution_metrics.emit('noise_evaluation', branches=1)
    try:
        record = store.page('0', 100, [], [])['observations'][-1]
        assert [stage['id'] for stage in record['stages']] == ['composite.sample']
        stage = record['stages'][0]
        assert stage['sharedOperationId'] == 'shared-sample'
        assert stage['workload']['resolved']['steps']['availableAt'] == 'execution_start'
        assert 'workload_resolution_unavailable' not in record['coverageReasons']
        assert stage['durationSeconds'] == 3 and stage['phase'] == 'succeeded'
        assert stage['workload']['measurements']['noiseEvaluations']['value'] == 1
        with CurrentNodeContext('prompt', 'outer'):
            with execution_metrics.stage_scope('composite.sample', 'KSampler', reused=True):
                pass
        stage = store.page('0', 100, [], [])['observations'][-1]['stages'][0]
        assert stage['reused'] is False and stage['durationSeconds'] == 3
        assert stage['workload']['measurements']['verifiedReuseHits']['value'] == 1
        now[0] = 10
        store.safe_stage('prompt', 'composite.sample', 'started')
        now[0] = 12
        store.safe_stage('prompt', 'composite.sample', 'finished')
        stage = store.page('0', 100, [], [])['observations'][-1]['stages'][0]
        assert stage['durationSeconds'] == 5
        assert 'multiple_stage_invocations' in stage['reasons']
    finally:
        execution_metrics.set_observer(None)


def test_terminal_storage_failure_cleans_only_attempt_memory(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', graph(), {}, ['decode'])
    store.event('accepted', item)
    store.event('started', item)
    store.metric_states[('prompt', 'sample')] = {}
    store.metric_states[('other', 'sample')] = {}
    store.control_origins['prompt'] = {object(): 'apply'}
    store.control_origins['other'] = {object(): 'apply'}
    store.db.close()
    store.safe_event('finished', item, SimpleNamespace(completed=True, messages=[]))
    assert 'prompt' not in store.clocks
    assert ('prompt', 'sample') not in store.metric_states
    assert ('other', 'sample') in store.metric_states
    assert 'prompt' not in store.control_origins
    assert 'other' in store.control_origins


def graph():
    return {
        'base': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'model'}},
        'lora': {'class_type': 'LoraLoader', 'inputs': {'model': ['base', 0], 'lora_name': 'first', 'strength_model': 1, 'strength_clip': 0}},
        'lora2': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['lora', 0], 'lora_name': 'second', 'strength_model': 0}},
        'size': {'class_type': 'PrimitiveInt', 'inputs': {'value': 1024}},
        'latent': {'class_type': 'EmptyLTXVLatentVideo', 'inputs': {'width': ['size', 0], 'height': 512, 'length': 98, 'batch_size': 1}},
        'sample': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['lora2', 0], 'latent_image': ['latent', 0], 'steps': 20, 'start_at_step': 4, 'end_at_step': 15}},
        'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['base', 2]}},
    }


def description(value, **options):
    return describe(value, {'profileRevision': 'implementation'},
                    {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'},
                    '2026-11-05T00:00:00Z', lambda key, value: digest([key, value]), digest, **options)


def test_owner_declared_loader_aliases_do_not_broaden_unknown_node_inference():
    value = {'pool': {'class_type': 'OwnedPoolLoader', 'inputs': {'clip_name': 'encoder', 'type': 'flux'}},
             'custom': {'class_type': 'LooksLikeSampler', 'inputs': {'lora_name': 'PRIVATE', 'steps': 20}}}
    _, unknown = description(value)
    assert unknown[0]['kind'] == 'unknown'
    evidence, stages = description(value, loader_aliases={'OwnedPoolLoader': 'CLIPLoader', 'LooksLikeSampler': 'KSampler'})
    assert stages[0]['kind'] == 'loading' and stages[1]['kind'] == 'unknown'
    assert len(evidence['components']) == 1 and evidence['components'][0]['role'] == 'clip_name'
    assert evidence['components'][0]['identity'] == digest(['clip_name', 'encoder'])
    assert 'PRIVATE' not in json.dumps(evidence)


def test_linked_dimensions_video_padding_and_separate_stage_work():
    value = graph()
    evidence, stages = description(value)
    assert evidence['profile'] == 'diffusion_video'
    sample = next(stage for stage in stages if stage['id'] == 'sample')
    facts = sample['workload']['resolved']
    assert {key: facts[key]['value'] for key in ('latentWidth', 'latentHeight', 'latentFrames')} == {
        'latentWidth': 32, 'latentHeight': 16, 'latentFrames': 13}
    assert facts['start_at_step']['value'] == 4
    assert sample['dependsOn'] == ['latent', 'lora2']
    value['refine'] = copy.deepcopy(value['sample'])
    value['refine']['inputs'].update(latent_image=['sample', 0], steps=8)
    evidence, stages = description(value)
    assert evidence['resolved']['sample.steps']['value'] == 20
    assert evidence['resolved']['refine.steps']['value'] == 8


def test_ordered_loras_bypass_and_control_intervals():
    value = graph()
    value['control'] = {'class_type': 'ControlNetApplyAdvanced', 'inputs': {'strength': 0.7, 'start_percent': 0.2, 'end_percent': 0.8}}
    evidence, _ = description(value)
    loras = [component for component in evidence['components'] if component['kind'] == 'lora']
    assert [component['features']['bypassed']['value'] for component in loras] == [False, True]
    assert loras[0]['order'] < loras[1]['order']
    assert evidence['resolved']['control.start_percent']['value'] == 0.2
    assert evidence['resolved']['control.end_percent']['value'] == 0.8
    assert 'model_evaluations_unmeasured' in evidence['reasons']
    assert not evidence['complete']


def test_unknown_nodes_and_references_do_not_acquire_invented_coverage():
    value = graph()
    value['custom'] = {'class_type': 'LooksLikeKnownModelSampler', 'inputs': {'width': 4096, 'prompt': 'SECRET', 'lora_name': 'not-an-attributed-loader'}}
    value['image'] = {'class_type': 'LoadImage', 'inputs': {'image': '/PRIVATE/path.png'}}
    evidence, stages = description(value)
    assert 'unsupported_workload_adapter' in evidence['reasons']
    assert 'reference_processing_unmeasured' in evidence['reasons']
    assert next(stage for stage in stages if stage['id'] == 'custom')['kind'] == 'unknown'
    assert len([component for component in evidence['components'] if component['kind'] == 'lora']) == 2
    assert 'SECRET' not in json.dumps(evidence)
    assert '/PRIVATE' not in json.dumps(evidence)


def test_stage_durations_overlap_cache_and_progress_are_observations(tmp_path):
    now = [0.0]
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: now[0])
    item = (1, 'prompt', graph(), {}, ['decode'])
    store.event('accepted', item)
    now[0] = 2
    store.event('started', item)
    store.safe_cache('prompt', ['base'])
    store.safe_stage('prompt', 'sample', 'started')
    now[0] = 4
    store.safe_stage('prompt', 'decode', 'started')
    store.safe_stage('prompt', 'sample', 'progress', 4, 20)
    before = store.db.total_changes
    store.safe_stage('prompt', 'sample', 'progress', 5, 20)
    assert store.db.total_changes == before
    now[0] = 8
    store.safe_stage('prompt', 'sample', 'finished')
    now[0] = 9
    store.safe_stage('prompt', 'decode', 'finished')
    store.event('finished', item, SimpleNamespace(completed=True, messages=[]))
    page = store.page('0', 100, [], [])
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(page, schema['$defs']['TimingObservationPage'])
    record = page['observations'][-1]
    stages = {stage['id']: stage for stage in record['stages']}
    assert stages['base']['reused'] and stages['base']['durationSeconds'] == 0
    assert stages['sample']['durationSeconds'] == 6
    assert stages['decode']['durationSeconds'] == 5
    assert stages['sample']['progress']['reliable'] is False
    assert record['durations']['executionSeconds'] == 7
    assert stages['sample']['durationSeconds'] + stages['decode']['durationSeconds'] > 7
    changes = store.db.total_changes
    store.page('0', 100, [], [])
    assert store.db.total_changes == changes
    store.db.close()
    store.safe_stage('prompt', 'sample', 'started')
    assert store.issue == 'timing_stage_observation_unavailable'


def test_native_sampling_metadata_counts_and_reliable_progress(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', graph(), {}, ['decode'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_cache('prompt', [])
    store.safe_stage('prompt', 'sample', 'started')
    store.safe_metric('prompt', 'sample', 'sampling_plan', {'shape': (1, 128, 13, 16, 32), 'shapes': [(1, 128, 13, 16, 32)], 'evaluations': 2,
                      'branches': 1, 'precision': 'torch.bfloat16', 'memory_strategy': 'dynamic', 'load_device': 'cuda:0', 'offload_device': 'cpu'})
    before = store.db.total_changes
    for _ in range(2):
        store.safe_metric('prompt', 'sample', 'model_forward', {'elements': 1024, 'conditional': 1, 'unconditional': 0})
        store.safe_metric('prompt', 'sample', 'noise_evaluation', {'branches': 1})
    assert store.db.total_changes == before
    store.safe_metric('prompt', 'sample', 'sampling_progress', {'completed': 1, 'total': 2})
    store.safe_stage('prompt', 'sample', 'progress', 50, 100)
    store.safe_stage('prompt', 'sample', 'finished')
    store.event('finished', item, SimpleNamespace(completed=True, messages=[]))
    record = store.page('0', 100, [], [])['observations'][-1]
    stage = next(stage for stage in record['stages'] if stage['id'] == 'sample')
    assert stage['workload']['resolved']['modelEvaluations']['value'] == 2
    assert not any(key.startswith('stream.') for key in stage['workload']['resolved'])
    assert stage['workload']['resolved']['cfgBranches']['value'] == 1
    assert stage['workload']['measurements']['modelForwardCalls']['value'] == 2
    assert stage['workload']['measurements']['modelForwardElements']['value'] == 2048
    assert stage['progress'] == {'unit': 'sampling_steps', 'completed': 1, 'total': 2, 'reliable': True}
    assert record['workload']['precision'] == 'torch.bfloat16'
    assert 'model_evaluations_unmeasured' not in stage['reasons']
    assert not store.metric_states


def test_processed_reference_shapes_and_encoded_conditioning_tokens_exclude_contents():
    facts, references = resolved_inputs('VAEEncodeForInpaint', {
        'pixels': SimpleNamespace(shape=(2, 384, 640, 3)), 'mask': SimpleNamespace(shape=(2, 384, 640)),
        'grow_mask_by': 6, 'private_path': '/PRIVATE/source.png'}, '2026-11-05T00:00:00Z')
    assert facts['grow_mask_by']['value'] == 6
    assert [(role, values['width']['value'], values['height']['value'], values['batch']['value'])
            for role, values in references] == [('pixels', 640, 384, 2), ('mask', 640, 384, 2)]
    tokens, _ = resolved_inputs('KSampler', {'positive': [[SimpleNamespace(shape=(1, 256, 4096)), {'private': 'SECRET'}]]}, '2026-11-05T00:00:00Z')
    assert tokens['positive.0.encodedTokens']['value'] == 256
    assert 'SECRET' not in json.dumps(tokens)
    assert '/PRIVATE' not in json.dumps(references)
    assert resolved_inputs('UnknownLooksLikeVAE', {'pixels': SimpleNamespace(shape=(2, 384, 640, 3))}, 'now') == ({}, [])


def test_lora_live_component_facts_and_completed_preparation_times_remain_separate(tmp_path):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', graph(), {}, ['decode'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'lora', 'started')
    store.safe_metric('prompt', 'lora', 'lora_file_cache', {'reused': False})
    store.safe_metric('prompt', 'lora', 'lora_patch_metadata', {'parsed_patches': 3, 'rank_count': 2, 'rank_min': 2,
                       'rank_max': 4, 'application_mode': 'native_patcher'})
    store.safe_metric('prompt', 'lora', 'lora_file_load', {'seconds': .2, 'succeeded': True})
    store.safe_metric('prompt', 'lora', 'lora_patch_preparation', {'seconds': .1, 'succeeded': True})
    store.safe_stage('prompt', 'lora', 'finished')
    record = store.page('0', 100, [], [])['observations'][-1]
    stage = next(stage for stage in record['stages'] if stage['id'] == 'lora')
    component = stage['workload']['components'][0]
    assert component['features']['rankCoverageComplete']['value'] is False
    assert component['features']['minRank']['value'] == 2
    assert component['features']['fileReused']['availableAt'] == 'live'
    assert component == next(candidate for candidate in record['workload']['evidence']['components']
                             if candidate['kind'] == 'lora' and candidate['order'] == component['order'])
    measured = stage['workload']['measurements']['loraFileLoadSeconds']
    assert measured['value'] == .2 and measured['source'] == 'native_lora'
    assert measured['availableAt'] == 'completion'


def test_processed_reference_boundaries_and_tile_coverage_do_not_use_contents(tmp_path):
    value = {'image': {'class_type': 'LoadImage', 'inputs': {'image': '/PRIVATE/image.png'}},
             'encode': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['image', 0]}}}
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', value, {}, ['encode'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'image', 'started')
    store.safe_metric('prompt', 'image', 'image_loaded', {'shape': (1, 10, 14, 3)})
    store.safe_stage('prompt', 'image', 'finished')
    store.safe_stage('prompt', 'encode', 'started')
    store.safe_inputs('prompt', 'encode', 'VAEEncode', {'pixels': SimpleNamespace(shape=(1, 10, 14, 3))})
    store.safe_metric('prompt', 'encode', 'vae_processed_input', {'shape': (1, 3, 8, 12), 'tiled': True, 'tile_count_complete': True})
    before = store.db.total_changes
    for _ in range(12):
        store.safe_metric('prompt', 'encode', 'vae_forward_attempt', {'shape': (1, 3, 4, 4), 'tiled': True})
        store.safe_metric('prompt', 'encode', 'vae_forward_completed', {'tiled': True})
    assert store.db.total_changes == before
    store.safe_stage('prompt', 'encode', 'finished')
    record = store.page('0', 100, [], [])['observations'][-1]
    stage = next(stage for stage in record['stages'] if stage['id'] == 'encode')
    component = stage['workload']['components'][0]
    assert component['features']['height']['value'] == 10
    assert component['features']['vaeEncode.0.axis.2']['value'] == 8
    assert component['features']['vaeEncode.0.axis.3']['value'] == 12
    assert stage['workload']['measurements']['vaeTileForwardCompleted']['value'] == 12
    assert stage['workload']['measurements']['vaeTileCountingComplete']['value'] is True
    assert '/PRIVATE' not in json.dumps(record)


def test_control_attribution_stores_scalar_work_without_runtime_handles(tmp_path):
    value = {'loader': {'class_type': 'ControlNetLoader', 'inputs': {'control_net_name': 'weight'}},
             'apply': {'class_type': 'ControlNetApplyAdvanced', 'inputs': {'control_net': ['loader', 0], 'strength': .7, 'start_percent': .2, 'end_percent': .8}},
             'sample': {'class_type': 'KSampler', 'inputs': {'positive': ['apply', 0]}}}
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', value, {}, ['sample'])
    with patch('app.run_timing.selected_file_identity', return_value='selected-control'):
        store.event('accepted', item)
        store.event('started', item)
    binding, evaluation = object(), object()
    store.safe_metric('prompt', 'apply', 'control_bound', {'binding': binding})
    store.safe_stage('prompt', 'sample', 'started')
    changes = store.db.total_changes
    for _ in range(2):
        store.safe_metric('prompt', 'sample', 'control_applied', {'binding': binding, 'evaluation': evaluation,
                          'shape': (1, 3, 32, 64), 'adapter': 'controlnet'})
    assert store.db.total_changes == changes
    store.safe_stage('prompt', 'sample', 'finished')
    record = store.page('0', 100, [], [])['observations'][-1]
    stage = next(item for item in record['stages'] if item['id'] == 'sample')
    component = stage['workload']['components'][0]
    assert component['identity'] == 'selected-control'
    assert component['features']['sourceStage']['value'] == 'apply'
    assert component['features']['hint.axis.3']['value'] == 64
    assert stage['workload']['measurements']['control.0.activeEvaluations']['value'] == 1
    assert stage['workload']['measurements']['control.0.activeCalls']['value'] == 2
    assert stage['workload']['measurements']['control.0.activeEvaluations']['availableAt'] == 'completion'
    assert not any('activeEvaluations' in key for key in stage['workload']['resolved'])
    assert 'object at' not in json.dumps(record)
