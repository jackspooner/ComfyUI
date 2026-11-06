import pytest

from app.run_timing_refmod import MAX_MEMBERS, observe_refmod_loader_metrics, refmod_loader_complete
from app.run_timing_workload import describe
from app.run_timing import digest


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def trace(reused=False, bundle=False):
    binding = object()
    def event(name, **facts):
        return 'refmod_loader_' + name, dict(binding=binding, **facts)
    rows = [event('begin', mode='slots', slots=8, version=1)]
    rows += [event('slot', slot=index, active=index == 1, side='single', copies=2, selection='All',
                   strength=.5, visual_strength=1, audio_strength=1) for index in range(1, 9)]
    rows += [event('duration', slot=1, phase='metadata_read', seconds=.001, succeeded=True)]
    if not reused or bundle:
        rows += [event('duration', slot=1, phase='bundle_read' if bundle else 'standalone_read', seconds=.002, succeeded=True)]
    rows += [event('loaded', slot=1, container='bundle' if bundle else 'standalone', reused=reused,
                   kind='image', shape=(1, 24, 1, 4, 4), precision='torch.float32', device='cpu', tokens=4,
                   member=0 if bundle else None, stamp_identity='a' * 64, stamp_assurance='size_mtime_ctime'),
             event('selection', slot=1, container='bundle' if bundle else 'standalone', members=1),
             event('duration', phase='total', slot=None, seconds=.005, succeeded=True), event('end', references=2)]
    return rows


def collect(rows):
    value = state()
    for event, facts in rows:
        assert observe_refmod_loader_metrics(value, event, facts)
    return value


@pytest.mark.parametrize('reused,bundle', [(False, False), (True, False), (False, True)])
def test_valid_cache_and_bundle_work_is_scalar_and_audit_is_separate(reused, bundle):
    value = collect(trace(reused, bundle))
    assert refmod_loader_complete(value)
    assert value['resolved']['refmodLoader.slot.1.member.0.reused'] is reused
    assert value['resolved']['refmodLoader.references'] == 2
    assert not any('stamp' in key or 'cpuSeconds' in key or key.endswith('.member') for key in value['resolved'])
    assert not any(type(value) is object for value in value['resolved'].values())
    assert set(value['availability'].values()) == {'live'}


@pytest.mark.parametrize('name,key,value', [
    ('begin', 'version', 2), ('begin', 'mode', 'custom'), ('begin', 'binding', None),
    ('slot', 'copies', True), ('slot', 'strength', float('nan')), ('slot', 'side', 'private-name'),
    ('slot', 'selection', 'Other'), ('slot', 'active', 'yes'),
    ('loaded', 'shape', (1, 24, 1, 3, 4)), ('loaded', 'shape', (1, 32, 2, 4)),
    ('loaded', 'tokens', 5), ('loaded', 'device', 'cuda:0'), ('loaded', 'precision', 'unknown'),
    ('loaded', 'reused', 1), ('loaded', 'member', 3), ('loaded', 'binding', None),
    ('loaded', 'stamp_identity', 'raw/path'), ('loaded', 'stamp_assurance', 'content_verified'),
    ('selection', 'members', 2), ('selection', 'container', 'bundle'),
    ('duration', 'seconds', -1), ('duration', 'succeeded', False), ('end', 'references', 1),
])
def test_invalid_metadata_correlation_and_spans_never_prove_work(name, key, value):
    rows = trace()
    next(facts for event, facts in rows if event == 'refmod_loader_' + name)[key] = value
    result = collect(rows)
    assert result['reasons'] and not refmod_loader_complete(result)


@pytest.mark.parametrize('name', ['begin', 'slot', 'loaded', 'selection', 'duration', 'end'])
def test_missing_or_duplicate_observations_stay_incomplete(name):
    rows = trace()
    index = next(index for index, (event, _) in enumerate(rows) if event == 'refmod_loader_' + name)
    for duplicate in (False, True):
        edited = list(rows)
        if duplicate:
            edited.insert(index, rows[index])
        else:
            edited.pop(index)
        assert not refmod_loader_complete(collect(edited))


def test_bounded_members_do_not_silently_truncate():
    rows = trace(bundle=True)
    loaded = next(facts for name, facts in rows if name == 'refmod_loader_loaded')
    start = rows[:rows.index(('refmod_loader_loaded', loaded))]
    # All eight slots can legally select a 256-member bundle; telemetry caps the
    # combined snapshot and reports incomplete coverage rather than hiding work.
    for _, facts in start:
        if 'active' in facts:
            facts['active'] = True
    value = collect(start)
    for slot in (1, 2):
        for phase in ('metadata_read', 'bundle_read') if slot == 2 else ():
            observe_refmod_loader_metrics(value, 'refmod_loader_duration', dict(binding=loaded['binding'],
                slot=slot, phase=phase, seconds=.01, succeeded=True))
        for member in range(MAX_MEMBERS):
            observe_refmod_loader_metrics(value, 'refmod_loader_loaded', dict(loaded, slot=slot, member=member))
    assert value['refmod_loader']['members'] == MAX_MEMBERS
    assert 'refmod_loader_member_limit' in value['reasons']
    assert not refmod_loader_complete(value)


def test_bypass_filtering_and_uncached_bundles_cannot_be_forged():
    for key, wrong in [('selection', 'Audio'), ('visual_strength', 0)]:
        rows = trace()
        next(facts for event, facts in rows if event == 'refmod_loader_slot')[key] = wrong
        assert not refmod_loader_complete(collect(rows))
    assert not refmod_loader_complete(collect(trace(reused=True, bundle=True)))
    value = state()
    assert not observe_refmod_loader_metrics(value, 'unrelated', {})


def test_only_explicit_loader_adapters_and_no_backbone_or_private_inputs():
    for kind in ('MiniMaxH3RefModsLoader', 'MiniMaxH3RefModsAxis', 'CustomRefMod'):
        graph = {'loader': {'class_type': kind, 'inputs': {'mod_1': 'private/person', 'show_info': True}}}
        evidence, stages = describe(graph, {'profileRevision': 'fixture'},
            {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: None, digest)
        stage = stages[0]
        assert stage['kind'] == ('unknown' if kind == 'CustomRefMod' else 'loading')
        assert stage['workload']['modelIdentity'] is None
        assert stage['workload']['profile'] == 'unknown'
        assert 'unsupported_workload_adapter' in stage['reasons'] if kind == 'CustomRefMod' else 'refmod_loader_work_unmeasured' in stage['reasons']
        assert not stage['workload']['complete']
        assert 'private/person' not in repr(evidence)
