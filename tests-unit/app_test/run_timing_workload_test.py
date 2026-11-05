import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from app.run_timing import RunTimingStore, digest
from app.run_timing_workload import describe, resolved_inputs


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


def description(value):
    return describe(value, {'profileRevision': 'implementation'},
                    {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'},
                    '2026-11-05T00:00:00Z', lambda key, value: digest([key, value]), digest)


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
    store.safe_metric('prompt', 'sample', 'sampling_plan', {'shape': (1, 128, 13, 16, 32), 'evaluations': 2,
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
