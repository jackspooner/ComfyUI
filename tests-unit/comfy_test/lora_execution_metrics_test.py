from itertools import count
from types import SimpleNamespace

import pytest
import torch

import nodes
from comfy import execution_metrics, sd, weight_adapter


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def test_lora_bypass_file_reuse_rank_coverage_and_original_outputs(monkeypatch):
    calls, events = [], []
    patched = SimpleNamespace(add_patches=lambda values, strength: calls.append(('patch', strength)) or list(values))
    model = SimpleNamespace(model=object(), clone=lambda: patched)
    adapters = {
        'a': weight_adapter.LoRAAdapter(set(), (torch.zeros(8, 2), torch.zeros(2, 8), None, None, None, None)),
        'b': weight_adapter.LoRAAdapter(set(), (torch.zeros(8, 4), torch.zeros(4, 8), None, None, None, None)),
        'other': ('diff', (torch.zeros(8, 8),)),
    }
    monkeypatch.setattr(sd.comfy.lora, 'model_lora_keys_unet', lambda *_: {})
    monkeypatch.setattr(sd.comfy.lora_convert, 'convert_lora', lambda value: value)
    monkeypatch.setattr(sd.comfy.lora, 'load_lora', lambda *_: adapters)
    monkeypatch.setattr(nodes.folder_paths, 'get_full_path_or_raise', lambda *args: '/PRIVATE/' + args[-1])
    def load(*args, **kwargs):
        calls.append(('load', args[0]))
        return {}, None
    monkeypatch.setattr(nodes.comfy.utils, 'load_torch_file', load)
    clock = count()
    monkeypatch.setattr(execution_metrics, 'monotonic', lambda: float(next(clock)))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    loader = nodes.LoraLoader()
    assert loader.load_lora(model, None, 'adapter', 0, 0) == (model, None)
    assert calls == [] and events == [('lora_bypass', {})]
    events.clear()
    assert loader.load_lora(model, None, 'adapter', .5, 0) == (patched, None)
    assert loader.load_lora(model, None, 'adapter', 1, 0) == (patched, None)
    assert calls == [('load', '/PRIVATE/adapter'), ('patch', .5), ('patch', 1)]
    assert [facts['reused'] for event, facts in events if event == 'lora_file_cache'] == [False, True]
    assert [facts['seconds'] for event, facts in events if event == 'lora_file_load'] == [1]
    metadata = next(facts for event, facts in events if event == 'lora_patch_metadata')
    assert metadata == {'parsed_patches': 3, 'rank_count': 2, 'rank_min': 2, 'rank_max': 4, 'application_mode': 'native_patcher'}
    assert next(facts for event, facts in events if event == 'lora_patch_registered') == {'model_patches': 3, 'clip_patches': 0}
    assert '/PRIVATE' not in str(events)
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    assert loader.load_lora(model, None, 'adapter', 1, 0) == (patched, None)
    assert len(calls) == 4
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    monkeypatch.setattr(nodes.comfy.utils, 'load_torch_file', lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError('original load failure')))
    with pytest.raises(ValueError, match='original load failure'):
        loader.load_lora(model, None, 'different', 1, 0)
    assert events[-1] == ('lora_file_load', {'seconds': 1., 'succeeded': False})
