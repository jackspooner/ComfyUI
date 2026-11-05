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
                            offload_device='cpu', load_device='cpu', model=object(), hook_mode=None,
                            restore_hook_patches=lambda: None)
    guider = samplers.CFGGuider.__new__(samplers.CFGGuider)
    guider.model_patcher, guider.cfg, guider.model_options = model, 1, {}
    guider.original_conds = {'positive': [{}], 'negative': [{}]}
    calls, callbacks, observations = [], [], []
    expected = object()
    sampler = samplers.KSAMPLER(samplers.k_diffusion_sampling.sample_euler)
    def native_sample(noise, latent, sampler, sigmas, denoise_mask, callback, *args, **kwargs):
        calls.append(1)
        callback(0, None, None, 2)
        return expected
    monkeypatch.setattr(samplers.comfy.sampler_helpers, 'prepare_model_patcher', lambda *_: None)
    monkeypatch.setattr(samplers, 'filter_registered_hooks_on_conds', lambda *_: None)
    monkeypatch.setattr(samplers.comfy.patcher_extension.WrapperExecutor, 'new_class_executor',
                        lambda *_: SimpleNamespace(execute=native_sample))
    def sample():
        return guider.sample(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), sampler, torch.tensor([1., .5, 0.]),
                             callback=lambda *values: callbacks.append(values))
    execution_metrics.set_observer(lambda event, facts: observations.append((event, facts)))
    assert sample() is expected
    assert len(calls) == len(callbacks) == 1
    plan = dict(observations)['sampling_plan']
    assert plan['shape'] == (1, 4, 8, 8)
    assert plan['evaluations'] == 2 and plan['branches'] == 1
    assert dict(observations)['sampling_progress'] == {'completed': 1, 'total': 2}
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    assert sample() is expected
    assert len(calls) == 2


def test_nested_sampling_streams_and_custom_guiders_do_not_invent_cfg_work():
    class CustomGuider(samplers.CFGGuider):
        pass
    guider = CustomGuider.__new__(CustomGuider)
    guider.model_patcher = SimpleNamespace(model_dtype=lambda: torch.bfloat16, is_dynamic=lambda: False,
                                           offload_device='cpu', load_device='cpu')
    guider.cfg, guider.model_options, guider.original_conds = 1, {}, {'positive': [{}]}
    observations = []
    execution_metrics.set_observer(lambda event, facts: observations.append((event, facts)))
    shapes = [(1, 128, 4, 8, 8), (1, 8, 40)]
    callback = samplers.sampling_metrics_callback(guider, shapes, samplers.KSAMPLER(lambda: None), torch.zeros(3), None)
    callback(1, None, None, 2)
    facts = dict(observations)['sampling_plan']
    assert facts['shape'] is None and facts['shapes'] == shapes
    assert facts['evaluations'] is None and facts['branches'] is None


def test_nested_stage_scope_restores_context_and_preserves_exceptions():
    observations = []
    execution_metrics.set_observer(lambda event, facts: observations.append((event, execution_metrics.current_stage())))
    with execution_metrics.stage_scope('outer', 'Outer'):
        with execution_metrics.stage_scope('inner', 'Inner', reused=True):
            assert execution_metrics.current_stage() == 'inner'
        assert execution_metrics.current_stage() == 'outer'
    assert execution_metrics.current_stage() is None
    with pytest.raises(ValueError, match='original'):
        with execution_metrics.stage_scope('failure', 'Failure'):
            raise ValueError('original')
    assert ('stage_reused', 'inner') in observations
    assert observations[-1] == ('stage_failed', 'failure')
    assert execution_metrics.current_stage() is None


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
