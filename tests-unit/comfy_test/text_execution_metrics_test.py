from types import SimpleNamespace

import pytest
import torch

from app.run_timing_text import observe_text_metrics
from comfy import execution_metrics
from comfy.text_encoders.llama import BaseGenerate


class TinyGenerator(BaseGenerate):
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8, stop_tokens=[2]),
                                     graph_dynamic_vbar_blocks=False, embed_tokens=torch.nn.Embedding(8, 2),
                                     forward=lambda *args, **kwargs: (kwargs['embeds'], None, kwargs['past_key_values']))
        self.steps = 0

    def init_kv_cache(self, batch, capacity, device, dtype):
        return []

    def logits(self, x):
        return torch.zeros(1, 1, 8)

    def sample_token(self, *args, **kwargs):
        self.steps += 1
        return torch.tensor([[self.steps]])


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def test_native_text_loop_keeps_eos_output_and_reports_fresh_kv_prefill_and_decode():
    events = []
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    def observe(event, facts):
        events.append((event, facts))
        observe_text_metrics(state, event, facts)
    execution_metrics.set_observer(observe)
    assert TinyGenerator().generate(torch.zeros(1, 3, 2), do_sample=False, max_length=0) == [1, 2]
    assert state['resolved']['inputPositions'] == 3
    assert state['resolved']['outputTokenLimit'] == 5
    assert state['resolved']['kvCapacity'] == 8
    assert state['resolved']['prefixReused'] is False
    assert state['measurements']['prefillForwardsCompleted'] == state['measurements']['decodeForwardsCompleted'] == 1
    assert state['measurements']['generatedTokens'] == 2
    assert state['progress'] == {'unit': 'generated_tokens', 'completed': 2, 'total': None, 'reliable': True}
    assert events[-1] == ('text_generation_completed', {'generated': 2})
    assert all(not isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    assert TinyGenerator().generate(torch.zeros(1, 3, 2), do_sample=False, max_length=0) == [1, 2]


def test_native_text_forward_failure_is_not_replayed_or_reported_as_completion():
    model, events = TinyGenerator(), []
    def fail(*args, **kwargs):
        raise ValueError('native forward failed')
    model.model.forward = fail
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(ValueError, match='native forward failed'):
        model.generate(torch.zeros(1, 3, 2), do_sample=False)
    assert [event for event, _ in events] == ['text_generation_plan', 'text_prefill_forward']
    assert events[-1][1]['succeeded'] is False
    assert model.steps == 0


def test_repeated_text_invocations_do_not_claim_reliable_single_loop_progress():
    state = {'resolved': {}, 'measurements': {}, 'reasons': []}
    execution_metrics.set_observer(lambda event, facts: observe_text_metrics(state, event, facts))
    for _ in range(2):
        TinyGenerator().generate(torch.zeros(1, 3, 2), do_sample=False, max_length=0)
    assert state['progress']['reliable'] is False
    assert 'multiple_text_generation_invocations' in state['reasons']
