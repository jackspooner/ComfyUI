import copy
import json
from types import SimpleNamespace

import pytest

from comfy.cli_args import args
args.cpu = True
from comfy.text_generation import read_history, format_qwen_chat, select_context, write_history
from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer, Qwen35TEModel
from comfy.text_encoders.moss_audio import MossAudioTokenizer, te
from comfy_extras.nodes_textgen import TextGenerate, TextGenerateLTX2Prompt


def tokenize(messages):
    # An exact synthetic tokenizer with measurable role/template overhead.
    return {'model': [[(i, 1.0) for i in range(2 + sum(len(m['content']) + 1 for m in messages))]]}


def transcript():
    return [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'aa'},
            {'role': 'assistant', 'content': 'bb'}, {'role': 'user', 'content': 'cc'},
            {'role': 'assistant', 'content': 'dd'}, {'role': 'user', 'content': 'new'}]


@pytest.mark.parametrize('bad', ['null', '[]', '{}', '{"version":true,"messages":[]}',
    '{"version":3,"messages":[]}', '{"version":1,"messages":[{"role":"tool","content":"x"}]}',
    '{"version":1,"messages":[{"role":"user","content":[]}]}',
    '{"version":1,"messages":[{"role":"user","content":"x","extra":1}]}',
    '{"version":1,"messages":[{"role":"user","content":"x"},{"role":"system","content":"x"}]}'])
def test_reject_invalid_history(bad):
    with pytest.raises(ValueError):
        read_history(bad, 'next')


def test_system_replacement_unicode_and_branch_isolation():
    source = transcript()[:-1]
    original = copy.deepcopy(source)
    encoded = write_history(source)
    branch = read_history(encoded, '你好', 'replacement')
    assert branch[0] == {'role': 'system', 'content': 'replacement'}
    assert branch[-1]['content'] == '你好'
    branch[1]['content'] = 'changed'
    assert source == original
    assert read_history(encoded, 'next')[0] == source[0]
    assert json.loads(write_history(branch))['messages'] == branch


def test_exact_fit_overflow_and_auto():
    messages = transcript()
    tokens, info = select_context(messages, tokenize, 40, 0, 20, 'error')
    assert info['input_tokens'] == 20
    assert info['output_allowance'] == 20
    assert info['excluded_message_ranges'] == []
    assert len(tokens['model'][0]) == 20
    with pytest.raises(ValueError, match='Context overflow'):
        select_context(messages, tokenize, 40, 39, 20, 'error')
    _, auto = select_context(messages, tokenize, 40, 21, 0, 'drop_oldest_turns')
    assert auto['output_allowance'] == 1
    assert auto['excluded_message_ranges'] == []


def test_remove_whole_turns_preserve_system_and_current_input():
    messages = transcript()
    original = copy.deepcopy(messages)
    observed = []
    def record(turns):
        observed.append(copy.deepcopy(turns))
        return tokenize(turns)
    _, info = select_context(messages, record, 40, 13, 5, 'drop_oldest_turns')
    assert info['excluded_message_ranges'] == [[1, 3], [3, 5]]
    assert info['input_tokens'] == 8
    assert observed[-1] == [messages[0], messages[-1]]
    assert messages == original
    with pytest.raises(ValueError, match='Context overflow'):
        select_context(messages, tokenize, 40, 8, 1, 'drop_oldest_turns')


@pytest.mark.parametrize('budget,maximum,policy', [(-1, 1, 'error'), (41, 1, 'error'), (True, 1, 'error'), (20, -1, 'error'), (20, 1, 'silent')])
def test_invalid_budget_before_tokenization(budget, maximum, policy):
    with pytest.raises(ValueError):
        select_context(transcript(), lambda _: pytest.fail('must validate first'), 40, budget, maximum, policy)


def test_unpaired_messages_are_not_silently_removed():
    messages = [{'role': 'assistant', 'content': 'long history'}, {'role': 'user', 'content': 'now'}]
    with pytest.raises(ValueError, match='Context overflow'):
        select_context(messages, tokenize, 10, 0, 1, 'drop_oldest_turns')


def test_no_approximate_media_counting():
    with pytest.raises(ValueError, match='media embeddings'):
        select_context(transcript(), lambda _: {'model': [[({'type': 'image'}, 1)]]}, 40, 0, 1, 'error')


def test_model_template_owners_and_context_limits():
    messages = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}, {'role': 'user', 'content': 'Again'}]
    fake = SimpleNamespace(tokenize_with_weights=lambda text, **kw: (text, kw))
    native = Qwen35ImageTokenizer()
    rows = native.tokenize_chat(messages, thinking=False)[native.clip]
    qwen = native.qwen35_2b.tokenizer.decode([t[0] for row in rows for t in row])
    assert qwen == format_qwen_chat(messages, False)
    assert qwen.count('<|im_start|>') == 4
    assert qwen.endswith('<think>\n\n</think>\n\n')
    moss, options = MossAudioTokenizer.tokenize_chat(fake, messages, thinking=True)
    assert moss.startswith('<|im_start|>system\nYou are a helpful assistant.')
    assert options['skip_template'] is True
    assert moss.endswith('<|im_start|>assistant\n')
    model = SimpleNamespace(transformer=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))))
    assert Qwen35TEModel.get_chat_context_length(SimpleNamespace(clip='qwen', qwen=model)) == 40960
    assert te().get_chat_context_length(SimpleNamespace(moss_audio=model)) == 40960


def test_real_qwen_tokenizer_counts_template_overhead_and_unicode():
    native = Qwen35ImageTokenizer()
    messages = [{'role': 'system', 'content': 'Remember facts.'}, {'role': 'user', 'content': '你好 🌍'}]
    tokens = native.tokenize_chat(messages, thinking=False)
    ids = native.qwen35_2b.tokenizer.encode(format_qwen_chat(messages, False), add_special_tokens=False)
    assert [item[0] for row in tokens['qwen35_2b'] for item in row] == ids
    _, info = select_context(messages, lambda turns: native.tokenize_chat(turns, thinking=False), len(ids) + 8, 0, 8, 'error')
    assert info['input_tokens'] == len(ids)
    assert info['input_tokens'] > len(native.qwen35_2b.tokenizer.encode(messages[-1]['content'], add_special_tokens=False))


class Clip:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def tokenize_chat(self, messages, thinking=False):
        self.active = copy.deepcopy(messages)
        return tokenize(messages)

    def get_chat_context_length(self):
        return 40

    def generate(self, tokens, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError('cancelled')
        return [101, 102, 103]

    def decode(self, ids):
        return '<think>reason</think>final'


def test_node_returns_complete_history_and_exact_diagnostics():
    clip = Clip()
    result = TextGenerate.execute(clip, 'new', 5, {'sampling_mode': 'off'}, history=write_history(transcript()[:-1]), context_budget=13, history_policy='drop_oldest_turns').result
    assert result[:2] == ('final', 'reason')
    history = json.loads(result[2])
    assert history['messages'] == [*transcript(), {'role': 'assistant', 'content': 'final'}]
    info = json.loads(result[3])
    assert info['generated_tokens'] == 3
    assert info['excluded_message_ranges'] == [[1, 3], [3, 5]]
    assert len(clip.active) == 2
    assert clip.calls[0]['max_length'] == 5
    # The returned full history can branch into another turn without mutation.
    before = result[2]
    again = TextGenerate.execute(Clip(), 'next', 1, {'sampling_mode': 'off'}, history=before)
    assert json.loads(again.result[2])['messages'][-2]['content'] == 'next'
    assert result[2] == before


def test_ordinary_templated_text_outputs_history_for_the_next_node():
    single = SimpleNamespace(tokenize=lambda *a, **kw: [], generate=lambda *a, **kw: [1], decode=lambda _: 'First answer')
    first = TextGenerate.execute(single, 'My code is BLUE', 64, {'sampling_mode': 'off'})
    assert first.result[:2] == ('First answer', '')
    assert first.result[3] == ''  # No counters are invented on the legacy path.
    second = TextGenerate.execute(Clip(), '?', 1, {'sampling_mode': 'off'}, history=first.result[2])
    assert json.loads(second.result[2])['messages'][:2] == [
        {'role': 'user', 'content': 'My code is BLUE'}, {'role': 'assistant', 'content': 'First answer'}]
    raw = TextGenerate.execute(single, 'raw prompt', 64, {'sampling_mode': 'off'}, use_default_template=False)
    assert raw.result[2:4] == ('', '')
    assert raw.result[4] == 'First answer'


@pytest.mark.parametrize('extra', [{'audio': object()}, {'video': object()}, {'use_default_template': False}])
def test_history_rejects_raw_or_media_before_generation(extra):
    clip = Clip()
    with pytest.raises(ValueError):
        TextGenerate.execute(clip, 'x', 1, {'sampling_mode': 'off'}, history=write_history([]), **extra)
    assert clip.calls == []


def test_failure_does_not_append_a_turn():
    history = write_history(transcript()[:-1])
    with pytest.raises(RuntimeError, match='cancelled'):
        TextGenerate.execute(Clip(fail=True), 'new', 1, {'sampling_mode': 'off'}, history=history)
    assert json.loads(history)['messages'] == transcript()[:-1]


def test_ltx_prompt_keeps_its_existing_contract():
    schema = TextGenerateLTX2Prompt.define_schema()
    assert len(schema.outputs) == 2
    assert not {'history', 'context_budget', 'history_policy'} & {i.id for i in schema.inputs}
