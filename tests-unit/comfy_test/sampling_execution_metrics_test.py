import json
from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics, model_sampling, samplers


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def guider():
    result = samplers.CFGGuider.__new__(samplers.CFGGuider)
    result.model_patcher = SimpleNamespace(model_dtype=lambda: torch.float32, is_dynamic=lambda: False,
                                           load_device='cpu', offload_device='cpu')
    result.cfg, result.model_options, result.original_conds = 1, {}, {'positive': [{}]}
    return result


def plan(sampler, sigmas):
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    samplers.sampling_metrics_callback(guider(), [(1, 4, 2, 2)], sampler, sigmas, None)
    return dict(events)['sampling_plan']


@pytest.mark.parametrize('name', ['euler', 'euler_ancestral', 'res_multistep', 'er_sde'])
@pytest.mark.parametrize('flow', [False, True])
@pytest.mark.parametrize('terminal_zero', [False, True])
def test_reviewed_plans_match_real_native_solver_calls_and_preserve_output(name, flow, terminal_zero):
    sampling = model_sampling.CONST() if flow else SimpleNamespace()

    class Denoiser:
        def __init__(self):
            self.calls = 0
            self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=sampling),
                                               model_patcher=SimpleNamespace(get_model_object=lambda _: sampling))

        def __call__(self, x, sigma, **kwargs):
            self.calls += 1
            return x * .2 + .1

    sampler = samplers.ksampler(name)
    sigmas = torch.tensor([.9, .7, .4, .2, 0 if terminal_zero else .1])
    expected_callbacks, observed_callbacks, events = [], [], []
    execution_metrics.set_observer(None)
    first = Denoiser()
    expected = sampler.sampler_function(first, torch.ones(1, 4, 2, 2), sigmas,
        extra_args={'seed': 123}, disable=True, callback=lambda values: expected_callbacks.append(values['i']))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    callback = samplers.sampling_metrics_callback(guider(), [(1, 4, 2, 2)], sampler, sigmas,
        lambda step, *args: observed_callbacks.append(step))
    second = Denoiser()
    actual = sampler.sampler_function(second, torch.ones(1, 4, 2, 2), sigmas,
        extra_args={'seed': 123}, disable=True,
        callback=lambda values: callback(values['i'], values['denoised'], values['x'], len(sigmas) - 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    facts = dict(events)['sampling_plan']
    assert facts['sampler'] == name
    assert facts['evaluations'] == first.calls == second.calls == 4
    assert expected_callbacks == observed_callbacks == [0, 1, 2, 3]
    assert [facts['completed'] for event, facts in events if event == 'sampling_progress'] == [1, 2, 3, 4]


def test_sampler_options_follow_current_native_defaults_and_explicit_overrides():
    euler = plan(samplers.ksampler('euler'), [1, .5, 0])
    assert euler['sampler_options'] == {'s_churn': 0., 's_tmin': 0., 's_tmax': 'positive_infinity', 's_noise': 1.}
    assert json.loads(json.dumps(euler, allow_nan=False))['evaluations'] == 2
    ancestral = plan(samplers.ksampler('euler_ancestral', {'eta': .25, 's_noise': 0.}), [1, .5, 0])
    assert ancestral['sampler_options'] == {'eta': .25, 's_noise': 0., 'noise_sampler': 'native_default'}
    er = plan(samplers.ksampler('er_sde', {'max_stage': 2}), [1, .5, 0])
    assert er['sampler_options']['max_stage'] == 2
    assert er['sampler_options']['noise_scaler'] == 'native_default'


@pytest.mark.parametrize('name,options', [
    ('euler_ancestral', {'noise_sampler': lambda *_: None}),
    ('er_sde', {'noise_scaler': lambda x: x}),
    ('er_sde', {'s_noise': float('nan')}),
    ('er_sde', {'max_stage': torch.tensor(3)}),
    ('res_multistep', {'extra_unreviewed_option': 1}),
])
def test_opaque_or_invalid_options_keep_advance_evaluations_unknown(name, options):
    facts = plan(samplers.ksampler(name, options), [1, .5, 0])
    assert facts['evaluations'] is None and facts['sampler'] is None
    assert facts['sampler_options'] == {}


@pytest.mark.parametrize('name', ['euler_cfg_pp', 'res_multistep_cfg_pp', 'res_multistep_ancestral',
                                'dpmpp_sde', 'heun', 'dpm_adaptive'])
def test_unreviewed_cfgpp_multievaluation_and_adaptive_paths_remain_unknown(name):
    facts = plan(samplers.ksampler(name), [1, .5, 0])
    assert facts['evaluations'] is None and facts['sampler'] is None


def test_custom_sampler_subclasses_wrappers_and_inpaint_options_remain_unknown():
    class CustomSampler(samplers.KSAMPLER):
        pass
    for sampler in (CustomSampler(samplers.k_diffusion_sampling.sample_euler),
                    samplers.KSAMPLER(lambda *args: None), samplers.ksampler('euler', inpaint_options={'random': True})):
        assert plan(sampler, [1, .5, 0])['evaluations'] is None


def test_replaced_native_module_attribute_does_not_authorize_a_custom_sampler(monkeypatch):
    monkeypatch.setattr(samplers.k_diffusion_sampling, 'sample_euler', lambda *args, **kwargs: None)
    assert plan(samplers.ksampler('euler'), [1, .5, 0])['evaluations'] is None


def test_plan_never_reads_sigma_values_and_telemetry_failures_preserve_callback(monkeypatch):
    class LengthOnly:
        def __len__(self):
            return 7

        def __getitem__(self, key):
            raise AssertionError('Telemetry must not read sigma values')

    assert plan(samplers.ksampler('er_sde'), LengthOnly())['evaluations'] == 6
    monkeypatch.setattr(samplers.inspect, 'signature', lambda *_: (_ for _ in ()).throw(ValueError('probe failed')))
    events, calls = [], []
    execution_metrics.set_observer(lambda event, facts: events.append(event))
    original = lambda *values: calls.append(values) or 'preview-result'
    callback = samplers.sampling_metrics_callback(guider(), [], samplers.ksampler('er_sde'), LengthOnly(), original)
    assert events == ['sampling_plan_unavailable']
    assert callback(0, None, None, 6) == 'preview-result' and len(calls) == 1
    execution_metrics.set_observer(None)
    assert samplers.sampling_metrics_callback(guider(), [], None, None, original) is original
