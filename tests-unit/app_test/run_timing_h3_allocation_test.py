import json
from pathlib import Path

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_h3_allocation import h3_allocation_complete, observe_h3_allocation_metrics
from app.run_timing_workload import describe


def graph(length=6):
    return {'latent': {'class_type': 'EmptyMiniMaxH3LatentAV', 'inputs': {'width': 65, 'height': 47, 'length': length}},
            'sample': {'class_type': 'KSampler', 'inputs': {'latent_image': ['latent', 0], 'steps': 2}}}


def plan():
    return dict(video_shape=(1, 24, 7, 2, 4), audio_shape=(1, 32, 2, 37))


def temporal():
    return dict(frames=22, video_latents=7, audio_latents=37, frame_rate=24, audio_latent_rate=40)


def output():
    return dict(shapes={'video': (1, 24, 7, 2, 4), 'audio': (1, 32, 2, 37)},
                precisions={'video': 'torch.float32', 'audio': 'torch.float32'}, devices={'video': 'cpu', 'audio': 'cpu'})


def latest(store):
    return store.page('0', 100, [], [])['observations'][-1]


def start(tmp_path, custom=False):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    value = graph()
    if custom:
        value['latent']['class_type'] = 'CustomH3Allocator'
    item = (1, 'prompt', value, {}, ['sample'])
    store.event('accepted', item)
    store.event('started', item)
    store.safe_stage('prompt', 'latent', 'started')
    return store


def test_allocator_requests_are_audit_only_until_native_effective_work_is_observed():
    evidence, stages = describe(graph(), {'profileRevision': 'fixture'},
        {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: None, digest)
    work = stages[0]['workload']
    assert stages[0]['kind'] == 'preparation' and work['profile'] == evidence['profile'] == 'diffusion_video'
    assert work['requested'] == {'width': 65, 'height': 47, 'length': 6}
    assert work['resolved'] == {'jointAllocation.nativePlanRequired': {
        'value': True, 'availableAt': 'submission', 'observedAt': 'now', 'source': 'EmptyMiniMaxH3LatentAV'}}
    assert work['reasons'] == ['h3_allocation_workload_unmeasured']
    assert 'latent_shape_unresolved' in stages[1]['workload']['reasons']


def test_live_allocation_metadata_resolves_work_without_promoting_output_shapes(tmp_path):
    store = start(tmp_path)
    changes = store.db.total_changes
    store.safe_metric('prompt', 'latent', 'h3_av_temporal_plan', temporal())
    assert store.db.total_changes == changes
    store.safe_metric('prompt', 'latent', 'h3_av_allocation_plan', plan())
    assert store.db.total_changes > changes
    assert not latest(store)['stages'][0]['workload']['complete']
    store.safe_metric('prompt', 'latent', 'h3_av_allocate', {'seconds': .01, 'succeeded': True})
    store.safe_metric('prompt', 'latent', 'h3_av_allocation_output', output())
    work = latest(store)['stages'][0]['workload']
    assert work['complete'] and not work['reasons']
    assert work['resolved']['jointAllocation.frames']['value'] == 22
    assert work['resolved']['jointAllocation.plan.video.axis.2']['availableAt'] == 'execution_start'
    assert work['resolved']['jointAllocation.audio.precision']['availableAt'] == 'live'
    assert work['resolved']['executionPrecision']['source'] == 'native_h3_allocation'
    assert work['measurements']['jointAllocation.output.audio.axis.3']['availableAt'] == 'completion'
    assert not any(key.startswith('jointAllocation.output.') for key in work['resolved'])
    store.safe_stage('prompt', 'latent', 'finished')
    assert not latest(store)['stages'][0]['reasons']
    schema = json.loads((Path(__file__).parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


@pytest.mark.parametrize('event,facts', [('h3_av_temporal_plan', temporal()), ('h3_av_allocation_plan', plan()),
                                       ('h3_av_allocation_output', output())])
def test_repeated_evidence_invalidates_live_coverage_once(tmp_path, event, facts):
    store = start(tmp_path)
    for name, value in [('h3_av_temporal_plan', temporal()), ('h3_av_allocation_plan', plan()),
                        ('h3_av_allocation_output', output())]:
        store.safe_metric('prompt', 'latent', name, value)
    assert latest(store)['stages'][0]['workload']['complete']
    store.safe_metric('prompt', 'latent', event, facts)
    assert 'multiple_h3_allocation_invocations' in latest(store)['stages'][0]['workload']['reasons']
    changes = store.db.total_changes
    for _ in range(30):
        store.safe_metric('prompt', 'latent', event, facts)
    assert store.db.total_changes == changes


def test_mismatched_plans_failed_allocations_and_probe_failures_remain_incomplete(tmp_path):
    store = start(tmp_path)
    store.safe_metric('prompt', 'latent', 'h3_av_temporal_plan', temporal())
    store.safe_metric('prompt', 'latent', 'h3_av_allocation_plan', plan())
    changed = output()
    changed['shapes']['audio'] = (1, 32, 2, 38)
    store.safe_metric('prompt', 'latent', 'h3_av_allocation_output', changed)
    assert 'h3_allocation_output_mismatch' in latest(store)['stages'][0]['workload']['reasons']
    store.safe_metric('prompt', 'latent', 'tensor_shape_unavailable', {'operation': 'h3_av_allocation_output'})
    assert 'h3_allocation_metadata_unresolved' in latest(store)['stages'][0]['workload']['reasons']
    store.safe_metric('prompt', 'latent', 'h3_av_allocate', {'seconds': .01, 'succeeded': False})
    assert 'h3_allocation_execution_unresolved' in latest(store)['stages'][0]['workload']['reasons']


def test_unmeasured_execution_unknown_owners_and_cached_allocations_keep_gaps(tmp_path):
    store = start(tmp_path)
    for name, value in [('h3_av_temporal_plan', temporal()), ('h3_av_allocation_plan', plan()),
                        ('h3_av_allocation_output', output())]:
        store.safe_metric('prompt', 'latent', name, value)
    store.safe_stage('prompt', 'latent', 'finished')
    assert 'h3_allocation_execution_unresolved' in latest(store)['stages'][0]['workload']['reasons']
    other = start(tmp_path / 'custom', custom=True)
    for name, value in [('h3_av_temporal_plan', temporal()), ('h3_av_allocation_plan', plan()),
                        ('h3_av_allocation_output', output())]:
        other.safe_metric('prompt', 'latent', name, value)
    assert 'unsupported_workload_adapter' in latest(other)['stages'][0]['workload']['reasons']
    cached = start(tmp_path / 'cached')
    cached.safe_cache('prompt', ['latent'])
    stage = latest(cached)['stages'][0]
    assert stage['reused'] and not stage['workload']['complete'] and not stage['workload']['measurements']


@pytest.mark.parametrize('mutate', [lambda f: f.update(video_shape=(1, 25, 7, 2, 4)),
    lambda f: f.update(audio_shape=(1, 32, 3, 37)), lambda f: f.update(audio_shape=(2, 32, 2, 37)),
    lambda f: f.update(audio_shape=(1, 32, 2, 0)), lambda f: f.update(video_shape=(1, 24, 7, 2))])
def test_unknown_or_malformed_layouts_cannot_complete_work(mutate):
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    facts = plan()
    mutate(facts)
    observe_h3_allocation_metrics(state, 'h3_av_allocation_plan', facts)
    assert not h3_allocation_complete(state)
    assert 'h3_allocation_metadata_unresolved' in state['reasons']
