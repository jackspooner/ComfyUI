"""Pinned processor comparisons for native delay-family prompts and references."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen2TokenizerFast

from comfy.cli_args import args
args.cpu = True
from comfy import sd
import comfy.audio_codec as ac
from comfy.audio_codec import AudioCodes, AudioCodeSegment, MOSS_V1
from comfy.text_encoders.moss_speech import MossSpeechTokenizer, MossSpeechClipModel, encode_speech_reference
from comfy_extras.nodes_moss_speech import MossSpeechGenerate, MossDialogueGenerate


@pytest.fixture(params=[('delay', 'MOSS_DELAY_SOURCE', 32), ('acting', 'MOSS_ACTING_SOURCE', 32), ('ttsd', 'MOSS_TTSD_SOURCE', 16)])
def variant(request):
    name, variable, books = request.param
    if not os.environ.get(variable):
        pytest.skip(f'Requires pinned {variable} and the isolated Transformers 5 overlay')
    source = Path(os.environ[variable])
    manifest = json.loads((Path(__file__).parents[2] / f'scripts/moss_speech_{name}.json').read_text())
    files = ['configuration_moss_tts.py', 'processing_moss_tts.py', 'modeling_moss_tts.py', 'inference_utils.py']
    if name == 'acting':
        files.append('tts_robust_normalizer_single_script.py')
    for filename in files:
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == manifest['files'][filename]['sha256']
    package = types.ModuleType('moss_' + name + '_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    original = importlib.import_module(package.__name__ + '.modeling_moss_tts')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    config = original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True)
    processor = original.MossTTSDelayProcessor(tokenizer=tokenizer, model_config=config)
    raw = torch.frombuffer(bytearray((source / 'tokenizer.json').read_bytes()), dtype=torch.uint8)
    native = MossSpeechTokenizer(tokenizer_data=dict(tokenizer_json=raw, codebooks=books, dialogue=name == 'ttsd'))
    return processor, native, books, name


@pytest.mark.parametrize('with_references', [False, True])
def test_full_prompt_matches_pinned_processor(variant, with_references):
    processor, tokenizer, books, name = variant
    refs = [torch.arange(3 * books).reshape(3, books) % 1024, torch.arange(5 * books).reshape(5, books) % 1024] if with_references else None
    controls = dict(text=' [S1] Hello [laughs]!!! [S2] 你好。 ', instruction='Speak {text} in {language}; {tokens}',
                    language='en', tokens=71, quality='studio', sound_event='laughing', ambient_sound='quiet')
    expected = processor([[dict(role='user', reference=refs, **controls)]], mode='generation')['input_ids']
    payloads = [AudioCodes(MOSS_V1, (AudioCodeSegment(x.T[None].clone(), (len(x),)),)) for x in refs] if refs is not None else None
    tokens = tokenizer.tokenize_with_weights(references=payloads, normalize_text=name == 'acting', **controls)
    stage = MossSpeechClipModel.__new__(MossSpeechClipModel)
    captured = []
    def generate(ids, mask, length, seed, **sampling):
        captured.append(ids.clone())
        assert mask.all()
        return []
    stage.transformer = SimpleNamespace(codebooks=books, generate_delay=generate)
    stage.execution_device = torch.device('cpu')
    stage.generate_audio_codes(tokens, max_length=5)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_reference_gain_matches_upstream(variant):
    processor, _, _, _ = variant
    waveform = torch.stack([torch.linspace(-.02, .04, 3000), torch.linspace(-.8, .8, 3000)])[:, None]
    codec = SimpleNamespace(encode=Mock(return_value='codes'))
    assert encode_speech_reference(codec, {'waveform': waveform, 'sample_rate': 24000}) == 'codes'
    expected = torch.stack([processor.loudness_normalize(x.squeeze(0)) for x in waveform])[:, None]
    torch.testing.assert_close(codec.encode.call_args.args[0]['waveform'], expected, rtol=1e-6, atol=1e-7)
    assert codec.encode.call_args.args[1] == 32


@pytest.mark.parametrize('books,kind', [(32, sd.TEModel.MOSS_DELAY_8B), (16, sd.TEModel.MOSS_TTSD)])
def test_large_speech_detection_guards_each_key(books, kind):
    values = {'model.norm.weight': 4096, 'model.layers.35.self_attn.q_norm.weight': 128,
              'model.embed_tokens.weight': 155648, 'lm_heads.0.weight': 155648,
              'emb_ext.0.weight': 1025, f'emb_ext.{books-1}.weight': 1025, f'lm_heads.{books}.weight': 1025}
    weights = {k: torch.empty(v, device='meta') for k, v in values.items()}
    assert sd.detect_te_model(weights) == kind
    for key in weights.keys() - {'emb_ext.0.weight', 'lm_heads.0.weight'}:
        with pytest.raises(ValueError, match='Unsupported MOSS'):
            sd.detect_te_model({k: v for k, v in weights.items() if k != key})


def test_dialogue_schema_and_runtime_defaults_agree():
    inputs = MossDialogueGenerate.define_schema().inputs
    assert 'duration_tokens' not in [item.id for item in inputs]
    values = {item.id: item.default for item in inputs if hasattr(item, 'default')}
    assert values['audio_temperature'] == 1.1 and values['audio_top_k'] == 50
    assert values['max_length'] == 8192
    refs = next(x for x in MossSpeechGenerate.define_schema().inputs if x.id == 'references')
    assert refs.optional


@pytest.mark.parametrize('frames', [0, 2, 40])
def test_continuation_and_missing_speaker_match_processor(variant, frames):
    processor, tokenizer, books, name = variant
    prefix = torch.arange(frames * books).reshape(frames, books) % 1024
    reference = torch.arange(5 * books).reshape(5, books)
    text = '[S1] Reference. [S2] New dialogue.'
    user = processor.build_user_message(text=text, reference=[None, reference], instruction='')
    assistant = processor.build_assistant_message(audio_codes_list=[prefix])
    expected = processor([[user, assistant]], mode='continuation')['input_ids']
    def payload(codes):
        return AudioCodes(MOSS_V1, (AudioCodeSegment(codes.T[None].clone(), (len(codes),)),))
    tokens = tokenizer.tokenize_with_weights(text, references=[None, payload(reference)], continuation=payload(prefix), normalize_text=name == 'acting')
    stage = MossSpeechClipModel.__new__(MossSpeechClipModel)
    captured = []
    def generate(ids, mask, length, seed, **sampling):
        captured.append(ids.clone())
        assert mask.all()
        return []
    stage.transformer = SimpleNamespace(codebooks=books, generate_delay=generate)
    stage.execution_device = torch.device('cpu')
    stage.generate_audio_codes(tokens, max_length=5)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_sparse_speaker_references_keep_numbering():
    clip = SimpleNamespace(tokenize=Mock(return_value={}), generate_audio_codes=Mock(return_value='codes'))
    values = {item.id: item.default for item in MossSpeechGenerate.define_schema().inputs if hasattr(item, 'default')}
    values.update(clip=clip, text='[S2] Hello', references={'reference_1': 'speaker two'})
    MossSpeechGenerate.execute(**values)
    assert clip.tokenize.call_args.kwargs['references'] == (None, 'speaker two')


@pytest.mark.parametrize('prefix_length', [0, 3, 20])
def test_continuation_decode_matches_pinned_ratio(variant, prefix_length, monkeypatch):
    processor, _, books, _ = variant
    raw = torch.arange(12 * books).reshape(12, books)
    delayed = processor.apply_delay_pattern(raw, 1024)
    def decode_codes(codes):
        return torch.arange(codes.shape[-1] * 1920, dtype=torch.float32)[None, None]
    processor.decode_audio_codes = lambda codes: [decode_codes(item.T[None]) for item in codes]
    expected = processor._parse_audio_codes(prefix_length, delayed)
    stage = MossSpeechClipModel.__new__(MossSpeechClipModel)
    assistant = torch.cat((torch.zeros(len(delayed), 1, dtype=torch.int64), delayed), dim=1)
    stage.transformer = SimpleNamespace(codebooks=books, generate_delay=lambda *a, **k: [(prefix_length, assistant)])
    stage.execution_device = torch.device('cpu')
    generated = stage.generate_audio_codes({'moss_delay': [[(151644, 1)]]}, max_length=5)
    codec = ac.AudioCodec.__new__(ac.AudioCodec)
    codec.identity, codec.device = MOSS_V1, torch.device('cpu')
    codec.model = SimpleNamespace(decode=decode_codes)
    codec.load_model = Mock()
    monkeypatch.setattr(ac.comfy.ops, 'use_quantized_matmul', lambda *args: nullcontext())
    actual = codec.decode(generated)
    assert len(actual) == len(expected)
    for item, reference in zip(actual, expected):
        torch.testing.assert_close(item['waveform'], reference, rtol=0, atol=0)
