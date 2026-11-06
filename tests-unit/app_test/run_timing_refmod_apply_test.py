import pytest

from app.run_timing import digest
from app.run_timing_refmod_apply import observe_refmod_apply_metrics, refmod_apply_complete
from app.run_timing_workload import describe


def metadata(shape, **facts):
    return dict(shape=shape, precision='torch.float32', device='cpu', requires_grad=False,
                default_precision='torch.float32', grad_enabled=True, inference_mode=False, **facts)


def trace(branch='reuse', preset=False, references=1):
    binding = object()
    def event(name, index=None, **facts):
        return 'refmod_apply_' + name, dict(binding=binding, index=index, **facts)
    rows = [event('begin', version=1), event('settings', conditioning='native', preset_read=preset, preset_save=preset)]
    if preset:
        rows += [event('duration', phase='preset_read', seconds=.001, succeeded=True)]
    rows += [event('debug_plan', width=1024, height=1024, samples=160),
             event('duration', phase='debug_render', seconds=.02, succeeded=True)]
    if preset:
        rows += [event('duration', phase='preset_save', seconds=.001, succeeded=True)]
    rows += [event('selection', inputs=references, selected=references, scrambled=False)]
    for index in range(references):
        shape = (1, 24, 2, 4, 4)
        rows += [event('reference_begin', index=index, source=index),
                 event('reference_plan', index=index, **metadata(shape, native=True, kind='video', frames=2,
                       branch=branch, weights=2 if branch == 'frame_mix' else 0,
                       strength=0 if branch == 'drop' else .5 if branch != 'reuse' else 1))]
        if branch in ('frame_mix', 'scalar_mix'):
            rows += [event('blur_plan', index=index, **metadata(shape, factor=8, backend='cpu')),
                     event('duration', index=index, phase='blur', seconds=.001, succeeded=True)]
        if branch != 'drop':
            rows += [event('reference_output', index=index, **metadata(shape))]
        rows += [event('duration', index=index, phase='reference', seconds=.002, succeeded=True)]
    return rows + [event('duration', phase='debug_conversion', seconds=.001, succeeded=True),
                   event('debug_output', **metadata((1, 1024, 1024, 3))),
                   event('attached', entries=1, blocks=0 if branch == 'drop' else references),
                   event('duration', phase='total', seconds=.03, succeeded=True), event('end')]


def collect(rows):
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    for event, facts in rows:
        assert observe_refmod_apply_metrics(state, event, facts)
    return state


@pytest.mark.parametrize('branch', ['drop', 'reuse', 'scalar_mix', 'frame_mix'])
@pytest.mark.parametrize('preset', [False, True])
def test_actual_work_is_separate_from_strength_outputs_cpu_spans_and_provenance(branch, preset):
    state = collect(trace(branch, preset))
    assert refmod_apply_complete(state), state['reasons']
    assert state['resolved']['refmodApply.reference.0.branch'] == branch
    assert not any('cpuSeconds' in key or '.output.' in key or key.endswith(('.strength', '.sourceOrdinal')) for key in state['resolved'])
    assert all(type(value) in (str, bool, int, float) for value in state['resolved'].values())
    assert state['resolved']['refmodApply.preset_read'] is preset
    assert state['resolved']['executionPrecision'] == 'torch.float32'
    if branch == 'drop':
        assert not any('reference.0.input.' in key for key in state['resolved'])


def test_empty_application_has_a_complete_real_debug_and_attachment_plan():
    state = collect(trace(references=0))
    assert refmod_apply_complete(state), state['reasons']
    assert state['resolved']['refmodApply.selected'] == 0


@pytest.mark.parametrize('event,key,value', [
    ('begin', 'version', 2), ('begin', 'binding', None), ('begin', 'index', 0),
    ('settings', 'conditioning', 'custom'), ('settings', 'preset_read', 'yes'),
    ('selection', 'selected', 2), ('selection', 'inputs', False), ('selection', 'scrambled', 1),
    ('reference_begin', 'source', 2), ('reference_begin', 'index', 256),
    ('reference_plan', 'native', False), ('reference_plan', 'shape', (1, 24, 3, 4, 4)),
    ('reference_plan', 'precision', []), ('reference_plan', 'requires_grad', None),
    ('reference_plan', 'default_precision', 'unknown'), ('reference_plan', 'strength', float('nan')),
    ('reference_plan', 'branch', 'custom'), ('reference_plan', 'weights', 1),
    ('blur_plan', 'factor', 4), ('blur_plan', 'backend', 'mps'), ('blur_plan', 'device', 'cuda:0'),
    ('reference_output', 'shape', (1, 24, 1, 4, 4)), ('reference_output', 'device', 'cuda:0'),
    ('debug_plan', 'samples', 0), ('debug_output', 'precision', 'torch.float64'),
    ('attached', 'blocks', 2), ('duration', 'seconds', -1), ('duration', 'succeeded', False),
])
def test_invalid_or_unverified_work_never_clears_coverage(event, key, value):
    rows = trace('frame_mix')
    next(facts for name, facts in rows if name == 'refmod_apply_' + event)[key] = value
    result = collect(rows)
    assert result['reasons'] and not refmod_apply_complete(result)


@pytest.mark.parametrize('name', ['begin', 'settings', 'debug_plan', 'selection', 'reference_begin',
                                 'reference_plan', 'blur_plan', 'reference_output', 'debug_output', 'attached', 'end'])
def test_missing_and_duplicate_boundaries_remain_incomplete(name):
    rows = trace('frame_mix')
    index = next(index for index, (event, _) in enumerate(rows) if event == 'refmod_apply_' + name)
    for duplicate in (False, True):
        modified = list(rows)
        if duplicate:
            modified.insert(index, rows[index])
        else:
            modified.pop(index)
        assert not refmod_apply_complete(collect(modified))


def test_limit_and_uncorrelated_events_are_explicit_and_do_not_grow_work():
    rows = trace('reuse')
    index = next(i for i, (event, _) in enumerate(rows) if event == 'refmod_apply_reference_begin')
    value = collect(rows[:index])
    facts = dict(rows[0][1])
    observe_refmod_apply_metrics(value, 'refmod_apply_limit', facts)
    assert 'refmod_apply_reference_limit' in value['reasons']
    assert not refmod_apply_complete(value)
    assert not observe_refmod_apply_metrics(value, 'unrelated', {})


def test_explicit_adapter_keeps_both_source_stages_and_output_slots_without_names():
    graph = {'loader': {'class_type': 'MiniMaxH3RefModsLoader', 'inputs': {'mod_1': 'private/person'}},
             'conditioning': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {'width': 32, 'height': 32, 'length': 6}},
             'apply': {'class_type': 'MiniMaxH3RefModApply', 'inputs': {'conditioning': ['conditioning', 0], 'mods': ['loader', 0],
                       'graph_preset': 'private/preset', 'save_preset_as': 'private/output'}}}
    evidence, stages = describe(graph, {'profileRevision': 'fixture'},
        {'hostId': 'host', 'gpuUuid': None, 'instanceId': 'boot', 'laneId': 'comfyui'}, 'now', lambda *_: None, digest)
    stage = stages[-1]
    assert stage['kind'] == 'preparation' and stage['dependsOn'] == ['conditioning', 'loader']
    assert stage['workload']['profile'] == 'diffusion_video'
    assert 'refmod_apply_work_unmeasured' in stage['reasons']
    for component, source in zip(stage['workload']['components'], ['conditioning', 'loader']):
        assert component['features']['sourceStage']['value'] == source
        assert component['features']['sourceOutput']['value'] == 0
    assert 'private/' not in repr(evidence)
