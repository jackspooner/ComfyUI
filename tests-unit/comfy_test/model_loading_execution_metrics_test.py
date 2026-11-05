import weakref
from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics, model_management as manager


class Patcher:
    def __init__(self, name, calls, device='cuda:0', dynamic=False, loaded=0, clone=None, failure=None):
        self.name, self.calls, self.load_device = name, calls, torch.device(device)
        self.dynamic, self.loaded, self.clone, self.failure = dynamic, loaded, clone, failure
        self.model, self.parent, self.offload_device = torch.nn.Module(), None, torch.device('cpu')

    def is_dynamic(self):
        self.calls.append(('dynamic', self.name))
        return self.dynamic

    def model_patches_models(self):
        self.calls.append(('patch_models', self.name))
        return []

    def model_size(self):
        self.calls.append(('size', self.name))
        return 1000

    def loaded_size(self):
        self.calls.append(('loaded_size', self.name))
        return self.loaded

    def current_loaded_device(self):
        self.calls.append(('loaded_device', self.name))
        return self.load_device

    def is_clone(self, other):
        self.calls.append(('clone', self.name, other.name))
        return self.clone is not None and self.clone == other.clone and self is not other

    def model_patches_to(self, value):
        self.calls.append(('patches_to', self.name, str(value)))

    def model_dtype(self):
        self.calls.append(('dtype', self.name))
        return torch.float32

    def partially_load(self, device, budget, force_patch_weights=False):
        self.calls.append(('load', self.name, str(device), budget, force_patch_weights))
        if self.failure is not None:
            raise self.failure
        self.loaded = 1000
        return None if self.dynamic else 1000

    def loaded_ram_size(self):
        self.calls.append(('ram', self.name))
        return 2000

    def detach(self, *args, **kwargs):
        self.calls.append(('detach', self.name, args, kwargs))

    def partially_unload(self, device, amount):
        self.calls.append(('partial_unload', self.name, str(device), amount))
        self.loaded -= amount
        return amount


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch):
    monkeypatch.setattr(manager, 'current_loaded_models', [])
    monkeypatch.setattr(manager, 'vram_state', manager.VRAMState.NORMAL_VRAM)
    monkeypatch.setattr(manager, 'lowvram_available', True)
    monkeypatch.setattr(manager, 'minimum_inference_memory', lambda: 20)
    monkeypatch.setattr(manager, 'extra_reserved_memory', lambda: 0)
    monkeypatch.setattr(manager, 'MIN_WEIGHT_MEMORY_RATIO', .5)
    yield
    execution_metrics.set_observer(None)
    for entry in manager.current_loaded_models:
        if entry.model_finalizer is not None:
            entry.model_finalizer.detach()


def existing(patcher, calls):
    entry = manager.LoadedModel(patcher)
    entry.real_model = weakref.ref(patcher.model)
    entry.model_finalizer = SimpleNamespace(detach=lambda: calls.append(('finalizer_detach', patcher.name)))
    return entry


@pytest.mark.parametrize('device,dynamic,reuse,clones,full', [
    ('cpu', False, False, False, False), ('cpu', True, True, False, False),
    ('cuda:0', False, False, False, False), ('cuda:0', False, True, False, False),
    ('cuda:0', True, True, False, False), ('cuda:0', False, False, True, False),
    ('cuda:0', False, True, False, True),
])
def test_loading_observation_preserves_calls_order_policies_and_existing_queries(monkeypatch, device, dynamic, reuse, clones, full):
    results = []
    for mode in ('disabled', 'enabled', 'failed_observer'):
        calls, events = [], []
        manager.current_loaded_models = []
        monkeypatch.setattr(manager, 'cleanup_models_gc', lambda: calls.append(('cleanup',)))
        monkeypatch.setattr(manager, 'get_free_memory', lambda *args, **kwargs: calls.append(('free_query', args, kwargs)) or 5000)
        monkeypatch.setattr(manager, 'free_memory', lambda *args, **kwargs: calls.append(('free', args, kwargs)) or [])
        model = Patcher('primary', calls, device, dynamic, 100 if reuse else 0, clone='family' if clones else None)
        keep = [model]
        if reuse:
            manager.current_loaded_models.append(existing(model, calls))
        if clones:
            clone = Patcher('clone', calls, device, clone='family')
            keep.append(clone)
            manager.current_loaded_models.append(existing(clone, calls))
        execution_metrics.set_observer(None if mode == 'disabled' else
            (lambda *_: (_ for _ in ()).throw(OSError('observer offline'))) if mode == 'failed_observer' else
            lambda event, facts: events.append((event, facts)))
        assert manager.load_models_gpu([model, model], memory_required=30, force_full_load=full) is None
        results.append(calls.copy())
        if mode == 'enabled':
            selection = next(facts for event, facts in events if event == 'managed_load_selection')
            plan = next(facts for event, facts in events if event == 'managed_load_entry_plan')
            assert selection['reused'] is reuse and selection['dynamic'] is dynamic
            expected_loaded = (100 if reuse else 0) if device != 'cpu' and not full else None
            assert plan['loaded_bytes'] == expected_loaded
            assert sum(event == 'managed_model_load' for event, _ in events) == 1
            assert all(facts['succeeded'] for event, facts in events if event in ('managed_model_load', 'managed_model_clone_detach'))
            assert sum(event == 'managed_model_clone_detach' for event, _ in events) == int(clones)
            assert next(facts for event, facts in events if event == 'managed_load_plan')['models'] == 1
        for entry in manager.current_loaded_models:
            if entry.model_finalizer is not None:
                entry.model_finalizer.detach()
    assert results[0] == results[1] == results[2]


def test_model_loading_failure_is_original_exception_and_not_retried(monkeypatch):
    calls, events, failure = [], [], ValueError('original load failure')
    monkeypatch.setattr(manager, 'cleanup_models_gc', lambda: None)
    model = Patcher('primary', calls, 'cpu', failure=failure)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(ValueError) as caught:
        manager.load_models_gpu([model])
    assert caught.value is failure
    assert len([call for call in calls if call[0] == 'load']) == 1
    assert next(facts for event, facts in events if event == 'managed_model_load')['succeeded'] is False
    assert not manager.current_loaded_models


@pytest.mark.parametrize('required,detached', [(50, False), (150, True)])
def test_native_partial_and_full_unload_keep_return_and_registry_semantics(monkeypatch, required, detached):
    calls, events = [], []
    monkeypatch.setattr(manager, 'cleanup_models_gc', lambda: None)
    monkeypatch.setattr(manager, 'get_free_memory', lambda *args, **kwargs: 0)
    monkeypatch.setattr(manager, 'soft_empty_cache', lambda: calls.append(('empty_cache',)))
    monkeypatch.setattr(manager, 'vram_state', manager.VRAMState.HIGH_VRAM)
    model = Patcher('primary', calls, loaded=100)
    entry = existing(model, calls)
    manager.current_loaded_models = [entry]
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = manager.free_memory(required, model.load_device)
    assert result == ([entry] if detached else [])
    assert manager.current_loaded_models == ([] if detached else [entry])
    assert next(facts for event, facts in events if event == 'managed_model_unload')['succeeded'] is True
    assert next(facts for event, facts in events if event == 'managed_model_unload_result')['detached'] is detached
    assert sum(call[0] == 'partial_unload' for call in calls) == int(not detached)
    assert sum(call[0] == 'empty_cache' for call in calls) == int(detached)
