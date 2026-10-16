from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.text_encoders.llama import BaseGenerate
from comfy.text_encoders.qwen35 import Qwen35
from comfy_extras.nodes_textgen import TextGenerate


class TinyGenerator(BaseGenerate):
    def __init__(self, stop=99):
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=8, stop_tokens=[stop]),
            graph_dynamic_vbar_blocks=False,
            embed_tokens=torch.nn.Embedding(100, 2),
            forward=lambda *args, **kwargs: (kwargs['embeds'], None, kwargs['past_key_values']),
        )
        self.steps = 0

    def init_kv_cache(self, batch, capacity, device, dtype):
        self.capacity = capacity
        return []

    def logits(self, x):
        return torch.zeros(1, 1, 100)

    def sample_token(self, *args, **kwargs):
        self.steps += 1
        return torch.tensor([[self.steps]])


@pytest.mark.parametrize('input_length', [1, 3, 7])
def test_auto_uses_expanded_input_and_context(input_length):
    model = TinyGenerator()
    ids = model.generate(torch.zeros(1, input_length, 2), do_sample=False, max_length=0)
    assert len(ids) == 8 - input_length
    assert model.capacity == 8


def test_auto_still_stops_at_eos():
    model = TinyGenerator(stop=2)
    assert model.generate(torch.zeros(1, 3, 2), do_sample=False, max_length=0) == [1, 2]


@pytest.mark.parametrize('input_length', [8, 9])
def test_auto_full_context_fails_before_cache_allocation(input_length):
    model = TinyGenerator()
    with pytest.raises(ValueError, match='no room for generation'):
        model.generate(torch.zeros(1, input_length, 2), do_sample=False, max_length=0)
    assert not hasattr(model, 'capacity')


def test_explicit_budget_unchanged():
    model = TinyGenerator()
    assert model.generate(torch.zeros(1, 3, 2), do_sample=False, max_length=2) == [1, 2]
    assert model.capacity == 5
    assert model.resolve_max_length(32769, 3) == 32769


def test_mtp_resolves_auto_before_allocating_cache():
    model = TinyGenerator()
    class CacheReached(Exception):
        pass
    def capture(batch, capacity, device, dtype):
        assert capacity == 8 + 7  # MTP's existing draft/rollback workspace.
        raise CacheReached
    model.init_kv_cache = capture
    with pytest.raises(CacheReached):
        Qwen35._generate_mtp(model, torch.zeros(1, 3, 2), 0, [99])


def test_node_accepts_auto_and_larger_explicit_budgets():
    length = next(i for i in TextGenerate.define_schema().inputs if i.id == 'max_length')
    assert length.min == 0
    assert length.max == 1048576


@pytest.mark.parametrize('decoded,prompt,expected', [
    ('<think>reason</think>answer', 'question', ('answer', 'reason')),
    ('reason</think>answer', 'question', ('answer', 'reason')),
    ('caption\n</think>\ncaption', 'question', ('caption', 'caption')),
    ('<think>unfinished', 'question', ('', 'unfinished')),
    ('unfinished', 'question<think>', ('', 'unfinished')),
    ('ordinary answer', 'question', ('ordinary answer', '')),
])
def test_reasoning_split_with_template_or_generated_opening(decoded, prompt, expected):
    clip = SimpleNamespace(tokenize=lambda *args, **kwargs: [],
                           generate=lambda *args, **kwargs: [],
                           decode=lambda ids: decoded)
    output = TextGenerate.execute(clip, prompt, 512, {'sampling_mode': 'off'}, thinking=False)
    assert output.result == expected
