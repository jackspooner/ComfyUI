import copy
import json
from pathlib import Path

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_av import observe_av_metrics
from app.run_timing_workload import describe


def graph():
    return {'video': {'class_type': 'EmptyLTXVLatentVideo', 'inputs': {'width': 64, 'height': 64, 'length': 9, 'batch_size': 1}},
            'audio': {'class_type': 'EmptyLatentAudio', 'inputs': {'seconds': 1}},
            'concat': {'class_type': 'LTXVConcatAVLatent', 'inputs': {'video_latent': ['video', 0], 'audio_latent': ['audio', 0]}},
            'separate': {'class_type': 'LTXVSeparateAVLatent', 'inputs': {'av_latent': ['concat', 0]}},
            'sample': {'class_type': 'KSampler', 'inputs': {'latent_image': ['separate', 1]}}}


def plan(nested=False, masked=False):
    shapes = {'video': (1, 128, 2, 2, 2), 'audio': (1, 8, 6, 2)}
    if nested:
        shapes['target_audio'] = (1, 8, 4, 2)
    if masked:
        shapes['audio_mask'] = (1, 8, 6, 2)
    return dict(shapes=shapes, input_streams=2 if nested else 1,
                precisions={key: 'torch.float32' for key in shapes}, devices={key: 'cpu' for key in shapes})


def latest(store):
    return store.page('0', 100, [], [])['observations'][-1]


def start(tmp_path, node='concat'):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    item = (1, 'prompt', graph(), {}, [node])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', node, 'started')
    return store


def test_graph_adapter_retains_roles_slots_and_audio_video_ancestry_without_execution():
    value = graph()
    value['video_sample'] = {'class_type': 'KSampler', 'inputs': {'latent_image': ['separate', 0]}}
    value['joint_sample'] = {'class_type': 'KSampler', 'inputs': {'latent_image': ['concat', 0]}}
    evidence, stages = describe(value, {'profileRevision': 'fixture'},
        {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: None, digest)
    by_id = {stage['id']: stage for stage in stages}
    assert by_id['concat']['kind'] == by_id['separate']['kind'] == 'preparation'
    assert by_id['concat']['workload']['profile'] == 'diffusion_video'
    assert by_id['sample']['workload']['profile'] == 'audio'
    assert by_id['video_sample']['workload']['profile'] == by_id['joint_sample']['workload']['profile'] == 'diffusion_video'
    assert evidence['profile'] == 'unknown'
    assert 'joint_preparation_workload_unmeasured' in by_id['concat']['workload']['reasons']
    assert 'unsupported_workload_adapter' not in evidence['reasons']
    components = by_id['concat']['workload']['components']
    assert [component['role'] for component in components] == ['concat:video_latent', 'concat:audio_latent']
    assert [component['features']['sourceStage']['value'] for component in components] == ['video', 'audio']
    assert all(component['features']['sourceOutput']['value'] == 0 for component in components)


@pytest.mark.parametrize('nested,masked', [(False, False), (False, True), (True, False), (True, True)])
def test_plan_clears_only_own_gap_and_outputs_remain_completion_measurements(tmp_path, nested, masked):
    store = start(tmp_path)
    facts = plan(nested, masked)
    store.safe_metric('prompt', 'concat', 'av_concat_plan', facts)
    record = latest(store)
    stage = next(stage for stage in record['stages'] if stage['id'] == 'concat')
    assert stage['workload']['complete']
    assert stage['workload']['resolved']['jointPreparation.fitMode']['value'] == ('trim' if nested else 'none')
    assert stage['workload']['resolved']['executionPrecision']['source'] == 'native_av_preparation'
    assert 'joint_preparation_workload_unmeasured' in record['workload']['evidence']['reasons']  # separate remains unresolved
    components = stage['workload']['components']
    assert components[1]['features']['audio.axis.2']['value'] == 6
    assert components[1]['features']['sourceStage']['value'] == 'audio'
    output = copy.deepcopy(facts)
    output['shapes'] = store.metric_states[('prompt', 'concat')]['av_expected_output']
    output['precisions'] = {key: 'torch.float32' for key in output['shapes']}
    output['devices'] = {key: 'cpu' for key in output['shapes']}
    if nested:
        store.safe_metric('prompt', 'concat', 'av_concat_fit', {'seconds': .01, 'succeeded': True})
    store.safe_metric('prompt', 'concat', 'av_concat_output', output)
    store.safe_stage('prompt', 'concat', 'finished')
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')
    assert stage['workload']['complete'] and not stage['reasons']
    measurements = stage['workload']['measurements']
    assert measurements['jointPreparation.output.audio.axis.2']['value'] == (4 if nested else 6)
    assert measurements['jointPreparation.output.audio.axis.2']['availableAt'] == 'completion'
    assert not any(key.startswith('jointPreparation.output.') for key in stage['workload']['resolved'])
    schema = json.loads((Path(__file__).parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


@pytest.mark.parametrize('event', ['av_concat_plan', 'av_concat_output'])
def test_duplicate_and_invalid_live_evidence_immediately_invalidates_once(tmp_path, event):
    store = start(tmp_path)
    store.safe_metric('prompt', 'concat', 'av_concat_plan', plan())
    store.safe_metric('prompt', 'concat', event, plan())
    if event == 'av_concat_output':
        store.safe_metric('prompt', 'concat', event, plan())
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')
    assert not stage['workload']['complete']
    assert 'multiple_joint_preparation_invocations' in stage['reasons']
    changes = store.db.total_changes
    for _ in range(30):
        store.safe_metric('prompt', 'concat', event, plan())
    assert store.db.total_changes == changes


def test_unknown_masks_shapes_missing_output_and_probe_failures_keep_gaps(tmp_path):
    store = start(tmp_path)
    store.safe_metric('prompt', 'concat', 'av_concat_plan', plan())
    store.safe_metric('prompt', 'concat', 'tensor_shape_unavailable', {'operation': 'av_concat_output'})
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')
    assert 'joint_preparation_metadata_unresolved' in stage['workload']['reasons']
    store.safe_stage('prompt', 'concat', 'finished')
    assert 'joint_preparation_execution_unresolved' in next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')['reasons']
    for mutate in [lambda f: f.update(input_streams=3), lambda f: f['shapes'].update(audio=(2, 8, 6, 2)),
                   lambda f: f['shapes'].update(target_audio=(1, 8, 4, 3)), lambda f: f.pop('precisions')]:
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        facts = plan(True)
        mutate(facts)
        observe_av_metrics(state, 'av_concat_plan', facts)
        assert 'joint_preparation_metadata_unresolved' in state['reasons']


def test_reuse_without_prior_shape_proof_does_not_invent_preparation_work(tmp_path):
    store = start(tmp_path)
    store.safe_cache('prompt', ['concat'])
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')
    assert stage['reused'] and stage['durationSeconds'] == 0
    assert not stage['workload']['complete'] and not stage['workload']['measurements']


def test_separation_and_output_failure_keep_source_shapes_out_of_completion_work(tmp_path):
    store = start(tmp_path, 'separate')
    facts = plan(masked=True)
    facts['shapes']['video_mask'] = (1, 1, 2, 2, 2)
    facts['precisions']['video_mask'] = 'torch.float32'
    facts['devices']['video_mask'] = 'cpu'
    facts['input_streams'] = 2
    store.safe_metric('prompt', 'separate', 'av_separate_plan', facts)
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'separate')
    assert stage['workload']['complete']
    assert stage['workload']['resolved']['jointPreparation.input.audio.axis.2']['availableAt'] == 'live'
    assert stage['workload']['components'][0]['features']['sourceStage']['value'] == 'concat'
    store.safe_metric('prompt', 'separate', 'av_separate_output', facts)
    store.safe_stage('prompt', 'separate', 'finished')
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'separate')
    assert stage['workload']['complete'] and not stage['reasons']
    assert stage['workload']['measurements']['jointPreparation.output.audio.precision']['availableAt'] == 'completion'


@pytest.mark.parametrize('failure', ['output_kind', 'output_dtype', 'output_shape', 'fit_twice'])
def test_contradictory_execution_evidence_invalidates_live_coverage(tmp_path, failure):
    store = start(tmp_path)
    facts = plan()
    if failure == 'fit_twice':
        facts = plan(nested=True)
    store.safe_metric('prompt', 'concat', 'av_concat_plan', facts)
    if failure == 'fit_twice':
        for _ in range(2):
            store.safe_metric('prompt', 'concat', 'av_concat_fit', {'seconds': .01, 'succeeded': True})
    else:
        event = 'av_separate_output' if failure == 'output_kind' else 'av_concat_output'
        if failure == 'output_dtype':
            facts['precisions']['audio'] = 'torch.float16'
        if failure == 'output_shape':
            facts['shapes']['audio'] = (1, 8, 7, 2)
        store.safe_metric('prompt', 'concat', event, facts)
    stage = next(stage for stage in latest(store)['stages'] if stage['id'] == 'concat')
    assert not stage['workload']['complete']
