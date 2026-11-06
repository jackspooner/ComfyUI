from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics, sd1_clip
from comfy.text_encoders import minimax


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def conditioning_backend(calls, failure=None, model_class=minimax.MiniMaxQwen3VL):
    class Embedding:
        def __call__(self, tokens, out_dtype):
            calls.append(('embedding', tuple(tokens.shape), str(out_dtype)))
            return torch.ones(*tokens.shape, 4, dtype=out_dtype)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Embedding()

        def forward(self, input_ids, **kwargs):
            embeds = kwargs['embeds']
            calls.append(('forward', tuple(embeds.shape), kwargs['intermediate_output'],
                          kwargs['final_layer_norm_intermediate'], str(kwargs['dtype']), kwargs['num_tokens']))
            if failure is not None:
                raise failure
            return embeds + 1, embeds + 2

    transformer = model_class.__new__(model_class)
    torch.nn.Module.__init__(transformer)
    transformer.model = Backbone()
    transformer.dtype, transformer.num_layers = torch.float32, 2
    owner = SimpleNamespace(transformer=transformer, execution_device='cpu', special_tokens={'pad': 0},
        enable_attention_masks=False, layer='last', layer_idx=None, layer_norm_hidden_state=False,
        zero_out_masked=False, return_attention_masks=False, return_projected_pooled=True)
    owner.process_tokens = lambda tokens, device: sd1_clip.SDClipModel.process_tokens(owner, tokens, device)
    return owner


@pytest.mark.parametrize('selection', [None, -1, [0, 1], 'all'])
def test_native_expansion_and_full_sequence_forward_preserve_calls_and_outputs(selection):
    calls, events = [], []
    owner = conditioning_backend(calls)
    owner.layer = 'last' if selection is None else 'hidden' if type(selection) is int else selection
    owner.layer_idx = selection if type(selection) is int else None
    baseline = sd1_clip.SDClipModel.forward(owner, [[1, 2, 0]])
    baseline_calls = list(calls)
    calls.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = sd1_clip.SDClipModel.forward(owner, [[1, 2, 0]])
    assert calls == baseline_calls
    for a, b in zip(actual, baseline):
        torch.testing.assert_close(a, b)
    plan = next(facts for event, facts in events if event == 'h3_conditioning_plan')
    assert plan['shapes'] == {'input': (1, 3, 4)} and plan['native'] is True
    assert plan['compute_precision'] == 'torch.float32' and plan['layers'] == 2
    assert plan['selection'] == ('last' if selection is None else 'indices' if selection != 'all' else 'all')
    row = next(facts for event, facts in events if event == 'text_embedding_row')
    assert row == dict(positions=3, text=2, image=0, audio=0, unknown=0, padding=1)
    forward = next(facts for event, facts in events if event == 'h3_conditioning_forward')
    assert forward['succeeded'] is True and forward['seconds'] >= 0
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    calls.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('timing unavailable')))
    sd1_clip.SDClipModel.forward(owner, [[1, 2, 0]])
    assert calls == baseline_calls


def test_native_forward_failure_is_preserved_without_retry():
    calls, events = [], []
    failure = RuntimeError('original conditioning failure')
    owner = conditioning_backend(calls, failure)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError) as raised:
        sd1_clip.SDClipModel.forward(owner, [[1]])
    assert raised.value is failure and [call[0] for call in calls] == ['embedding', 'forward']
    assert events[-1][0] == 'h3_conditioning_forward' and events[-1][1]['succeeded'] is False


def test_custom_subclasses_and_extra_options_are_observed_without_claiming_native_work():
    class CustomH3(minimax.MiniMaxQwen3VL):
        pass
    calls, events = [], []
    owner = conditioning_backend(calls, model_class=CustomH3)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = owner.transformer(None, embeds=torch.zeros(1, 2, 4), custom_option=True)
    assert result[0].shape == (1, 2, 4)
    plan = next(facts for event, facts in events if event == 'h3_conditioning_plan')
    assert plan['native'] is False and plan['extra_options'] is True


def test_metadata_probe_failure_cannot_fail_conditioning(monkeypatch):
    calls, events = [], []
    owner = conditioning_backend(calls)
    monkeypatch.setattr(execution_metrics, 'observe_shapes', lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('metadata unavailable')))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = sd1_clip.SDClipModel.forward(owner, [[1]])
    assert result[0].shape == (1, 1, 4)
    assert ('h3_conditioning_metadata_unavailable', {}) in events


def test_unexpected_owner_metadata_cannot_expose_tensor_values_or_change_results():
    calls, events = [], []
    owner = conditioning_backend(calls)
    owner.transformer.dtype = torch.tensor([12345., 67890.])
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = sd1_clip.SDClipModel.forward(owner, [[1]])
    assert result[0].shape == (1, 1, 4)
    assert ('h3_conditioning_metadata_unavailable', {}) in events
    assert '12345' not in repr(events) and '67890' not in repr(events)
