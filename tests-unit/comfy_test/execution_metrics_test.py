from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics, samplers


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def test_sampling_metrics_preserve_result_callback_and_inference_count(monkeypatch):
    model = SimpleNamespace(model_dtype=lambda: torch.bfloat16, is_dynamic=lambda: True,
                            offload_device='cpu', load_device='cpu')
    sampler = samplers.KSampler.__new__(samplers.KSampler)
    sampler.model, sampler.device, sampler.sampler, sampler.scheduler = model, 'cpu', 'euler', 'normal'
    sampler.sigmas, sampler.model_options = torch.tensor([1., .5, 0.]), {}
    calls, callbacks, observations = [], [], []
    monkeypatch.setattr(samplers, 'sampler_object', lambda name: name)
    expected = object()

    def sample(*args, **kwargs):
        calls.append(args)
        kwargs['callback'](0, None, None, 2)
        return expected

    monkeypatch.setattr(samplers, 'sample', sample)
    execution_metrics.set_observer(lambda event, facts: observations.append((event, facts)))
    assert sampler.sample(None, [object()], [object()], 1, latent_image=torch.zeros(1, 4, 8, 8),
                          callback=lambda *values: callbacks.append(values)) is expected
    assert len(calls) == len(callbacks) == 1
    plan = dict(observations)['sampling_plan']
    assert plan['shape'] == (1, 4, 8, 8)
    assert plan['evaluations'] == 2 and plan['branches'] == 1
    assert dict(observations)['sampling_progress'] == {'completed': 1, 'total': 2}
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    assert sampler.sample(None, [object()], [object()], 1, latent_image=torch.zeros(1, 4, 8, 8)) is expected
    assert len(calls) == 2


def test_native_cfg_evaluation_reports_actual_conditional_branches(monkeypatch):
    observations, supplied = [], []
    execution_metrics.set_observer(lambda event, facts: observations.append((event, facts)))
    monkeypatch.setattr(samplers, 'calc_cond_batch', lambda model, conds, *args: supplied.append(conds) or [1, 2])
    monkeypatch.setattr(samplers, 'cfg_function', lambda *args, **kwargs: 'original-result')
    assert samplers.sampling_function(None, None, None, 'negative', 'positive', 1) == 'original-result'
    assert supplied == [['positive', None]]
    assert observations == [('noise_evaluation', {'branches': 1})]
    observations.clear()
    samplers.sampling_function(None, None, None, 'negative', 'positive', 1, {'disable_cfg1_optimization': True})
    assert observations == [('noise_evaluation', {'branches': 2})]


def test_optional_input_progress_observers_preserve_older_handlers_and_failure_isolation():
    from comfy_execution.progress import ProgressRegistry
    registry = ProgressRegistry('prompt', None)
    registry.register_handler(SimpleNamespace(name='old', enabled=True))
    registry.register_handler(SimpleNamespace(name='broken', enabled=True, inputs_handler=lambda *_: (_ for _ in ()).throw(OSError('offline'))))
    registry.inputs_ready('node', {'text': 'private fixture'})
