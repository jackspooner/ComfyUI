from types import SimpleNamespace

import pytest
import torch

from app.run_timing_controls import observe_control_metrics
from comfy import controlnet, execution_metrics, samplers


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def native_control():
    control = controlnet.ControlNet(None, compression_ratio=1, load_device=torch.device('cpu'))
    class Model:
        dtype = torch.float32
        def __call__(self, **values):
            return {'output': [values['x'].clone()]}
    control.control_model = Model()
    control.control_model_wrapped = None
    control.model_sampling_current = SimpleNamespace(timestep=lambda t: t, calculate_input=lambda t, x: x)
    return control


def test_native_intervals_count_evaluations_once_across_cfg_calls_and_preserve_results():
    control = native_control()
    events = []
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    def observe(event, facts):
        events.append((event, facts))
        observe_control_metrics(state, event, facts, 'apply')
    execution_metrics.set_observer(observe)
    control.set_cond_hint(torch.zeros(1, 3, 6, 10), strength=.5, timestep_percent_range=(.25, .75))
    control.timestep_range = (.75, .25)
    x, cond = torch.ones(1, 4, 4, 8), {'c_crossattn': torch.zeros(1, 2, 3)}
    for timestep in (.9, .6, .3, .1):
        with execution_metrics.evaluation_scope():
            for _ in range(2):
                result = control.get_control(x, torch.tensor([timestep]), cond, 1, {})
                if timestep in (.6, .3):
                    torch.testing.assert_close(result['output'][0], x * .5)
                else:
                    assert result is None
    measured = state['measurements']
    assert measured['control.0.activeEvaluations'] == 2
    assert measured['control.0.skippedEvaluations'] == 2
    assert measured['control.0.forwardAttempts'] == measured['control.0.activeCalls'] == 4
    assert measured['control.0.hintPreparations'] == 1
    assert state['control_features']['control.0.']['hint.axis.2'] == 4
    assert state['control_features']['control.0.']['hint.axis.3'] == 8
    assert events[0][0] == 'control_bound'
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    result = control.get_control(x, torch.tensor([.5]), cond, 1, {})
    torch.testing.assert_close(result['output'][0], x * .5)


def test_native_adapter_feature_reuse_is_separate_from_active_evaluations():
    calls, events = [], []
    class Adapter:
        unshuffle_amount = 1
        def to(self, *_):
            return self
        def cpu(self):
            return self
        def __call__(self, pixels):
            calls.append(tuple(pixels.shape))
            return {'input': [pixels]}
    adapter = controlnet.T2IAdapter(Adapter(), 3, compression_ratio=1, upscale_algorithm='nearest-exact', device=torch.device('cpu'))
    adapter.set_cond_hint(torch.zeros(1, 3, 4, 8))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    for _ in range(2):
        with execution_metrics.evaluation_scope():
            result = adapter.get_control(torch.ones(1, 4, 4, 8), torch.tensor([.5]), {}, 1, {})
            torch.testing.assert_close(result['input'][0], torch.zeros(1, 3, 4, 8))
    assert calls == [(1, 3, 4, 8)]
    names = [event for event, _ in events]
    assert names.count('control_applied') == 2
    assert names.count('control_forward_attempt') == names.count('control_features_encoded') == names.count('control_features_reused') == 1


def test_copies_preserve_binding_until_new_application_and_uncorrelated_work_stays_unknown():
    first = native_control().set_cond_hint(torch.zeros(1, 3, 4, 8))
    clone = first.copy()
    assert clone._timing_binding is first._timing_binding
    clone.set_cond_hint(torch.zeros(1, 3, 4, 8))
    assert clone._timing_binding is not first._timing_binding
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    execution_metrics.set_observer(lambda event, facts: observe_control_metrics(state, event, facts, None))
    first.get_control(torch.ones(1, 4, 4, 8), torch.tensor([.5]), {'c_crossattn': torch.zeros(1, 2, 3)}, 1, {})
    assert 'control_application_binding_unresolved' in state['reasons']
    assert 'control_evaluation_correlation_unavailable' in state['reasons']
    assert 'control.0.activeEvaluations' not in state['measurements']


def test_native_noise_scope_restores_parent_on_exception(monkeypatch):
    evaluations = []
    execution_metrics.set_observer(lambda event, facts: evaluations.append(facts['evaluation']) if event == 'probe' else None)
    def fail(*_):
        execution_metrics.observe_control('probe', object())
        raise ValueError('native failure')
    monkeypatch.setattr(samplers, 'calc_cond_batch', fail)
    with execution_metrics.evaluation_scope():
        execution_metrics.observe_control('probe', object())
        with pytest.raises(ValueError, match='native failure'):
            samplers.sampling_function(None, None, None, None, [], 1)
        execution_metrics.observe_control('probe', object())
    execution_metrics.observe_control('probe', object())
    assert evaluations[0] is evaluations[2]
    assert evaluations[1] is not evaluations[0]
    assert evaluations[3] is None


def test_zero_strength_native_application_bypasses_control_without_changing_conditioning():
    import nodes
    positive, negative = [], []
    events = []
    execution_metrics.set_observer(lambda *event: events.append(event))
    result = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, native_control(), torch.zeros(1, 4, 8, 3), 0, .2, .8)
    assert result[0] is positive and result[1] is negative
    assert events == []


def test_stacked_controls_keep_separate_activity_when_later_control_is_outside_interval():
    first = native_control().set_cond_hint(torch.zeros(1, 3, 4, 8), strength=.25)
    second = native_control().set_cond_hint(torch.zeros(1, 3, 4, 8), strength=.5).set_previous_controlnet(first)
    second.timestep_range = (.75, .25)
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    origins = {first._timing_binding: 'first', second._timing_binding: 'second'}
    execution_metrics.set_observer(lambda event, facts: observe_control_metrics(state, event, facts, origins.get(facts.get('binding'))))
    x, cond = torch.ones(1, 4, 4, 8), {'c_crossattn': torch.zeros(1, 2, 3)}
    for timestep, expected in ((.9, .25), (.5, .75)):
        with execution_metrics.evaluation_scope():
            result = second.get_control(x, torch.tensor([timestep]), cond, 1, {})
        torch.testing.assert_close(result['output'][0], x * expected)
    assert state['measurements']['control.0.activeEvaluations'] == 2
    assert state['measurements']['control.1.activeEvaluations'] == 1
    assert state['measurements']['control.1.skippedEvaluations'] == 1
    assert state['control_features']['control.0.']['sourceStage'] == 'first'
    assert state['control_features']['control.1.']['sourceStage'] == 'second'
