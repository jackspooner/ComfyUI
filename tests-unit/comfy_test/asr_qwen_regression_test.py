"""CPU regressions for Qwen chunk boundaries, alignment and compute precision."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RotaryEmbedding

from comfy.cli_args import args
args.cpu = True
from comfy.asr import qwen
from comfy.asr.common import patcher


@pytest.fixture(autouse=True)
def execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.mark.parametrize('parts,expected', [
    (['Hello', 'world'], 'Hello world'),
    (['Hello.', 'World!'], 'Hello. World!'),
    (['Hello', ', world!'], 'Hello, world!'),
    (['(', 'hello', ')'], '(hello)'),
    (['well-', 'known'], 'well-known'),
    (['Hello ', 'world'], 'Hello world'),
    (['Hello', ' world'], 'Hello world'),
    (['Hello', '', 'world'], 'Hello world'),
    (['', ''], ''),
    (['你好', '世界'], '你好世界'),
    (['こんにちは', '世界'], 'こんにちは世界'),
    (['ｺﾝﾆﾁﾊ', 'ｾｶｲ'], 'ｺﾝﾆﾁﾊｾｶｲ'),
    (['你好。', '世界'], '你好。世界'),
    (['你好。', 'Hello'], '你好。Hello'),
    (['Hello', '世界'], 'Hello世界'),
    (['你好', 'world'], '你好world'),
    (['你好 world', 'again'], '你好 world again'),
    (['안녕', '세계'], '안녕 세계'),
    (['สวัสดี', 'โลก'], 'สวัสดีโลก'),
])
def test_join_chunks_preserves_script_and_punctuation_boundaries(parts, expected):
    chunks = [{'text': text, 'start': index, 'end': index + 1} for index, text in enumerate(parts)]
    before = [dict(chunk) for chunk in chunks]
    assert qwen.join_chunks(chunks) == expected
    assert chunks == before


def test_transcribe_joins_stripped_chunks_without_changing_records():
    model = SimpleNamespace(aligner=False)
    with (patch.object(qwen, 'waveform', return_value=torch.zeros(32000)),
          patch.object(qwen, 'split_audio', return_value=[(0, 16000), (16000, 32000)]),
          patch.object(qwen, 'embeddings', return_value=(None, None)),
          patch.object(qwen, 'decode', side_effect=['language English<asr_text> Hello ', 'language English<asr_text> world '])):
        result = qwen.transcribe(model, {})
    assert result['text'] == 'Hello world'
    assert result['chunks'] == [
        {'start': 0, 'end': 1, 'language': 'English', 'text': 'Hello'},
        {'start': 1, 'end': 2, 'language': 'English', 'text': 'world'},
    ]


@pytest.mark.parametrize('aligner', [False, True])
@pytest.mark.parametrize('bad_token', [None, '<|audio_pad|>', '<|audio_start|>', '<|audio_end|>',
                                      '<|im_start|>', '<|im_end|>', '<|endoftext|>', 'out-of-vocabulary', '<timestamp>'])
def test_load_checks_required_tokenizer_ids_before_network_construction(aligner, bad_token):
    required = {'<|audio_pad|>': 151676, '<|audio_start|>': 151669, '<|audio_end|>': 151670,
                '<|im_start|>': 151644, '<|im_end|>': 151645, '<|endoftext|>': 151643, '<timestamp>': 151705}
    thinker = {'audio_token_id': 151676, 'audio_start_token_id': 151669, 'audio_end_token_id': 151670,
               'text_config': {'vocab_size': 151706}}
    if aligner:
        thinker['classify_num'] = 5000
    config = {'thinker_config': thinker, 'timestamp_token_id': 151705}
    if bad_token == 'out-of-vocabulary':
        thinker['text_config']['vocab_size'] = thinker['audio_token_id']
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: SimpleNamespace(ids=[] if text == bad_token else [required[text]]))
    metadata = {'config': json.dumps(config), 'preprocessor_config': '{}'}
    state = {'tokenizer_json': torch.tensor(list(b'{}'), dtype=torch.uint8)}
    with (patch.object(qwen, 'package', return_value=(metadata, state)),
          patch.object(qwen.Tokenizer, 'from_str', return_value=tokenizer),
          patch.object(qwen, 'networks', return_value=('encoder', 'decoder')) as networks,
          patch.object(qwen, 'patcher', side_effect=lambda model, *args: model),
          patch.object(qwen.mm, 'get_torch_device', return_value=torch.device('cpu'))):
        if bad_token is None or (bad_token == '<timestamp>' and not aligner):
            model = qwen.load('fixture.safetensors')
            assert model.aligner == aligner
            networks.assert_called_once_with(config)
        else:
            with pytest.raises(ValueError, match='Unexpected native Qwen tokenizer ID'):
                qwen.load('fixture.safetensors')
            networks.assert_not_called()


@pytest.mark.parametrize('bins,offset,duration,expected,warn', [
    ([0, 10], 0, 1.01, (0, .8), False),
    ([12, 13], 0, 1.01, (.96, 1.01), True),
    ([13, 13], 0, 1.01, (1.01, 1.01), True),
    ([13, 13], 2, 1.01, (3.01, 3.01), True),
    ([13, 13], 2.0000625, 1.0100625, (3.01, 3.01), True),
    ([0, 13], .0005625, 1.0100625, (.001, 1.010625), True),
    ([14, 15], 0, 1.01, None, True),
    ([-2, -1], 0, 1.01, None, True),
])
def test_alignment_bounds_after_tolerance_offsets_and_rounding(bins, offset, duration, expected, warn):
    chunk = {'text': 'word', 'language': 'English', 'start': offset, 'end': offset + duration}
    transcript = {'text': 'word', 'chunks': [chunk], 'warnings': [{'code': 'existing', 'message': 'retained'}]}
    model = SimpleNamespace(aligner=True, config={'support_languages': ['English'], 'timestamp_segment_time': 80})
    with (patch.object(qwen, 'waveform', return_value=torch.zeros(round((offset + duration) * 16000))),
          patch.object(qwen, 'embeddings', return_value=(None, None)),
          patch.object(qwen, 'decode', return_value=torch.tensor(bins))):
        result = qwen.align(model, {}, transcript)
    assert result['text'] == 'word' and result['chunks'] == [chunk]
    assert transcript['warnings'] == [{'code': 'existing', 'message': 'retained'}]
    assert bool(result['warnings'][1:]) == warn
    assert all(warning['code'] == 'transcription_timestamps_incomplete' for warning in result['warnings'][1:])
    if expected is None:
        assert result['segments'] == []
    else:
        segment, = result['segments']
        assert (segment['start'], segment['end']) == pytest.approx(expected)
        assert offset <= segment['start'] <= segment['end'] <= offset + duration


@pytest.mark.parametrize('start,end,expected_samples,expected_interval', [
    (0, 2, (0, 16160), (0, 1.01)),
    (-1, 2, (0, 16160), (0, 1.01)),
    (.2, 2, (3200, 16160), (.2, 1.01)),
    (2, 3, None, None),
    (-2, -1, None, None),
    (.8, .2, None, None),
    (.2, .2, None, None),
])
def test_alignment_uses_only_actual_audio_for_external_chunk_bounds(start, end, expected_samples, expected_interval):
    value = torch.arange(16160, dtype=torch.float32)
    chunk = {'text': 'word', 'language': 'English', 'start': start, 'end': end}
    transcript = {'text': 'word', 'chunks': [chunk]}
    model = SimpleNamespace(aligner=True, config={'support_languages': ['English'], 'timestamp_segment_time': 80})
    bins = [0, 13] if start <= 0 else [0, 11]
    with (patch.object(qwen, 'waveform', return_value=value),
          patch.object(qwen, 'embeddings', return_value=(None, None)) as embeddings,
          patch.object(qwen, 'decode', return_value=torch.tensor(bins)) as decode):
        result = qwen.align(model, {}, transcript)
    assert result['text'] == 'word' and result['chunks'] == [chunk]
    assert result['warnings']
    assert all(warning['code'] == 'transcription_timestamps_incomplete' for warning in result['warnings'])
    if expected_samples is None:
        assert result['segments'] == []
        embeddings.assert_not_called()
        decode.assert_not_called()
    else:
        left, right = expected_samples
        torch.testing.assert_close(embeddings.call_args.args[1], value[left:right])
        segment, = result['segments']
        assert (segment['start'], segment['end']) == pytest.approx(expected_interval)
        assert 0 <= segment['start'] <= segment['end'] <= len(value) / 16000


def decoder_fixture(dtype):
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, vocab_size=128, max_position_embeddings=128,
                        rope_theta=1000000)
    config._attn_implementation = 'sdpa'
    torch.manual_seed(5)
    reference = Qwen3ForCausalLM(config).to(dtype).eval()
    # RoPE computes frequencies in FP32, then rounds factors to hidden dtype.
    reference.model.rotary_emb = Qwen3RotaryEmbedding(config)
    state = dict(reference.model.state_dict())
    state['lm_head.weight'] = reference.lm_head.weight
    return config, reference, state


@pytest.mark.parametrize('storage', [torch.bfloat16, torch.float16, torch.float32, 'mixed'])
def test_fixed_bf16_compute_is_independent_of_checkpoint_storage(storage):
    config, reference, state = decoder_fixture(torch.bfloat16)
    baseline = qwen.Decoder(config.to_dict(), 128, torch.bfloat16, 'cpu')
    patcher(baseline, state, torch.device('cpu'))
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    stored = {key: value.to(dtypes[index % 3] if storage == 'mixed' else storage)
              for index, (key, value) in enumerate(state.items())}
    # Exercise the storage dtype that originally selected the wrong RoPE math.
    if storage == 'mixed':
        stored['norm.weight'] = state['norm.weight'].float()
    native = qwen.Decoder(config.to_dict(), 128, torch.bfloat16, 'cpu')
    patcher(native, stored, torch.device('cpu'))
    ids = torch.arange(1, 21)[None]
    for actual, expected in zip(native.compute_freqs_cis(ids, 'cpu'), baseline.compute_freqs_cis(ids, 'cpu')):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual = native.lm_head(native(ids, dtype=torch.bfloat16)[0])
    expected = baseline.lm_head(baseline(ids, dtype=torch.bfloat16)[0])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, reference(ids).logits, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_decoder_compute_variants_match_reference_rope_and_cached_logits(dtype):
    config, reference, state = decoder_fixture(dtype)
    native = qwen.Decoder(config.to_dict(), 128, dtype, 'cpu')
    native.load_state_dict(state, strict=True)
    positions = torch.arange(20)[None]
    actual = native.compute_freqs_cis(positions, 'cpu')
    cos, sin = reference.model.rotary_emb(torch.empty(1, 20, 32, dtype=dtype), positions)
    expected = (cos[:, None], sin[:, None, :, :config.head_dim // 2], -sin[:, None, :, config.head_dim // 2:])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b.float(), atol=0, rtol=0)
    cache = native.init_kv_cache(1, 32, torch.device('cpu'), dtype)
    previous = None
    for ids in (torch.arange(1, 21)[None], torch.tensor([[21]])):
        hidden, _, cache = native(ids, dtype=dtype, past_key_values=cache)
        expected = reference(ids, past_key_values=previous, use_cache=True)
        previous = expected.past_key_values
        atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (2e-3, 1e-2)
        torch.testing.assert_close(native.lm_head(hidden), expected.logits, atol=atol, rtol=rtol)
