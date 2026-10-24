import json

import pytest

from comfy.cli_args import args
args.cpu = True
from comfy_extras.nodes_textgen import TextGenerate, TextGenerateLTX2Prompt


class Clip:
    def __init__(self, answer='{"ok": true}'):
        self.answer = answer
        self.tokenize_kwargs = None
        self.chat_messages = None
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_kwargs = kwargs
        return {'qwen': [[(1, 1.0), (2, 1.0)]]}

    def tokenize_chat(self, messages, thinking=False):
        self.chat_messages = messages
        return {'qwen': [[(1, 1.0), (2, 1.0)]]}

    def get_chat_context_length(self):
        return 64

    def generate(self, tokens, **kwargs):
        self.calls.append(kwargs)
        return [3, 4]

    def decode(self, ids):
        if ids == [3, 4]:
            return self.answer
        return ''


def run(clip, answer, schema='{"type":"object"}', **kwargs):
    clip.answer = answer
    return TextGenerate.execute(clip, 'question', 8, {'sampling_mode': 'off'}, auto_resize=False, json_schema=schema, **kwargs)


def test_json_schema_adds_instruction_and_returns_extracted_json():
    clip = Clip()
    output = run(clip, 'Here is the result:\n```json\n{"city":"Paris","temp_c":21}\n```')

    assert output.result[0] == '{"city":"Paris","temp_c":21}'
    assert output.result[4] == 'Here is the result:\n```json\n{"city":"Paris","temp_c":21}\n```'
    system = clip.tokenize_kwargs['system_prompt']
    assert system.startswith('Return exactly one JSON value')
    assert system.endswith('{"type":"object"}')
    history = json.loads(output.result[2])['messages']
    assert history[-1]['content'] == clip.answer.strip()
    assert all('JSON Schema' not in str(message['content']) for message in history)


def test_empty_schema_keeps_the_ordinary_output():
    clip = Clip()
    output = run(clip, 'plain answer', schema='')

    assert output.result[0] == 'plain answer'
    assert clip.tokenize_kwargs['system_prompt'] == ''


def test_schema_instruction_appends_to_an_existing_system_prompt():
    clip = Clip()
    run(clip, '{"ok":true}', system_prompt='Be terse.')

    system = clip.tokenize_kwargs['system_prompt']
    assert system.startswith('Be terse.\n\n')
    assert 'JSON Schema' in system


def test_history_preserves_its_system_message_and_does_not_mutate_the_input():
    clip = Clip()
    history = json.dumps({'version': 1, 'messages': [
        {'role': 'system', 'content': 'Remember BLUE.'},
        {'role': 'user', 'content': 'My code is BLUE.'},
        {'role': 'assistant', 'content': 'Understood.'},
    ]})
    output = TextGenerate.execute(clip, 'What is my code?', 8, {'sampling_mode': 'off'},
                                  history=history, context_budget=64, auto_resize=False,
                                  json_schema='{"type":"object"}')

    assert json.loads(history)['messages'][0]['content'] == 'Remember BLUE.'
    assert clip.chat_messages[0]['role'] == 'system'
    assert clip.chat_messages[0]['content'].startswith('Remember BLUE.\n\n')
    assert 'JSON Schema' in clip.chat_messages[0]['content']
    written = json.loads(output.result[2])['messages']
    assert written[0]['content'] == 'Remember BLUE.'
    assert 'JSON Schema' not in written[0]['content']
    assert written[-1] == {'role': 'assistant', 'content': '{"ok": true}'}

    next_clip = Clip()
    TextGenerate.execute(next_clip, 'Again', 8, {'sampling_mode': 'off'}, history=output.result[2],
                         context_budget=64, auto_resize=False, json_schema='{"type":"object"}')
    assert next_clip.chat_messages[0]['content'].count('Return exactly one JSON value') == 1


def test_thinking_is_split_before_json_extraction():
    clip = Clip()
    output = run(clip, '<think>reason</think>{"answer":42}')

    assert output.result[0] == '{"answer":42}'
    assert output.result[1] == 'reason'
    assert output.result[4] == '<think>reason</think>{"answer":42}'


@pytest.mark.parametrize('answer,expected', [
    ('{"a":1}', '{"a":1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('Here you go:\n{"a": 1}\nDone.', '{"a": 1}'),
    ('prefix [1,2] suffix', '[1,2]'),
    ('42', '42'),
])
def test_extracts_the_first_json_value(answer, expected):
    assert run(Clip(), answer).result[0] == expected


@pytest.mark.parametrize('answer', ['', 'I cannot comply.', '{"a": NaN}'])
def test_non_json_answer_fails(answer):
    with pytest.raises(ValueError, match='did not return a JSON value'):
        run(Clip(), answer)


def test_invalid_schema_fails_before_generation():
    clip = Clip()
    with pytest.raises(ValueError, match='json_schema is not valid JSON'):
        TextGenerate.execute(clip, 'question', 8, {'sampling_mode': 'off'}, auto_resize=False,
                             json_schema='{not json')
    assert clip.tokenize_kwargs is None
    assert clip.calls == []


def test_non_object_schema_fails_before_generation():
    clip = Clip()
    with pytest.raises(ValueError, match='json_schema must be a JSON object'):
        TextGenerate.execute(clip, 'question', 8, {'sampling_mode': 'off'}, auto_resize=False,
                             json_schema='[]')
    assert clip.tokenize_kwargs is None
    assert clip.calls == []


def test_json_schema_requires_the_default_template():
    clip = Clip()
    with pytest.raises(ValueError, match='use_default_template'):
        TextGenerate.execute(clip, 'question', 8, {'sampling_mode': 'off'}, auto_resize=False,
                             use_default_template=False, json_schema='{}')
    assert clip.tokenize_kwargs is None
    assert clip.calls == []


def test_ltx_prompt_node_does_not_expose_json_schema():
    assert 'json_schema' not in {item.id for item in TextGenerateLTX2Prompt.define_schema().inputs}
