import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
import comfy.sd
from comfy.audio_codec import AudioCodes, MOSS_V1
from comfy.text_encoders.moss_speech import MossSpeechTokenizer, MossSpeechClipModel, voice_prompt, undelay_segments
from comfy_extras.nodes_moss_speech import MossVoiceGenerator


def test_detection_is_specific_and_guards_keys():
    weights = {key: torch.empty(size, device='meta') for key, size in {
        'model.norm.weight': 2048, 'model.layers.27.self_attn.q_norm.weight': 128,
        'model.embed_tokens.weight': 155648, 'lm_heads.0.weight': 155648,
        'emb_ext.0.weight': 1025, 'emb_ext.15.weight': 1025, 'lm_heads.16.weight': 1025,
    }.items()}
    assert comfy.sd.detect_te_model(weights) == comfy.sd.TEModel.MOSS_VOICEGENERATOR
    for key in weights.keys() - {'emb_ext.0.weight', 'lm_heads.0.weight'}:
        broken = {k: v for k, v in weights.items() if k != key}
        with pytest.raises(ValueError, match='Unsupported MOSS'):
            comfy.sd.detect_te_model(broken)
    assert comfy.sd.detect_te_model({'model.layers.0.post_attention_layernorm.weight': torch.empty(2048, device='meta'),
                                    'model.layers.0.self_attn.q_norm.weight': torch.empty(128, device='meta')}) == comfy.sd.TEModel.QWEN3_2B


def delay(codes):
    result = torch.full((codes.shape[0] + codes.shape[1] - 1, codes.shape[1]), 1024, dtype=torch.int64)
    for i in range(codes.shape[1]):
        result[i:i + codes.shape[0], i] = codes[:, i]
    return result


def test_de_delay_separates_segments_without_protocol_padding():
    first = torch.arange(48).reshape(3, 16)
    second = torch.arange(80).reshape(5, 16)
    raw = torch.cat([torch.full((1, 16), 1024), delay(first), torch.full((3, 16), 1024), delay(second), torch.full((1, 16), 1024)])
    segments = undelay_segments(raw)
    AudioCodes(MOSS_V1, tuple(segments)).validate(MOSS_V1)
    assert [s.lengths for s in segments] == [(3,), (5,)]
    assert torch.equal(segments[0].codes[0].T, first)
    assert torch.equal(segments[1].codes[0].T, second)
    raw.fill_(1000)
    assert torch.equal(segments[0].codes[0].T, first)


@pytest.mark.parametrize('steps,frames', [(0, 0), (4, 0), (15, 0), (16, 1), (17, 2)])
def test_short_budget_returns_only_complete_frames(steps, frames):
    raw = delay(torch.arange(80).reshape(5, 16))[:steps]
    segments = undelay_segments(raw)
    assert sum(s.lengths[0] for s in segments) == frames
    AudioCodes(MOSS_V1, tuple(segments)).validate(MOSS_V1)


def test_tokenizer_raw_template_and_embedded_roundtrip():
    source = os.environ.get('MOSS_VOICEGENERATOR_SOURCE')
    if not source:
        pytest.skip('Requires local pinned tokenizer JSON')
    raw = (Path(source) / 'tokenizer.json').read_bytes()
    tokenizer = MossSpeechTokenizer(tokenizer_data={'tokenizer_json': torch.frombuffer(bytearray(raw), dtype=torch.uint8)})
    tokens = tokenizer.tokenize_with_weights(' Hello [softly]!!! ', instruction=' [warm] voice!!! ', language='en')
    ids = [pair[0] for pair in tokens['moss_delay'][0]]
    assert tokenizer.decode(ids, skip_special_tokens=False) == voice_prompt(' Hello [softly]!!! ', ' [warm] voice!!! ', 'en')
    assert tokenizer.state_dict()['tokenizer_json'].numpy().tobytes() == raw
    with pytest.raises(ValueError, match='embedded tokenizer'):
        MossSpeechTokenizer()


def test_clip_delegation_loads_before_generation(monkeypatch):
    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reset_clip_options = Mock()
            self.set_clip_options = Mock()
            self.audio_generation_memory = Mock(return_value=123)
            self.generate_audio_codes = Mock(return_value=AudioCodes(MOSS_V1, ()))
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = Stage()
    clip.patcher = SimpleNamespace(load_device=torch.device('cpu'))
    load = Mock()
    monkeypatch.setattr(comfy.sd.model_management, 'load_models_gpu', load)
    tokens = {'moss_delay': [[(10, 1.)]]}
    result = clip.generate_audio_codes(tokens, max_length=37, seed=8, audio_top_k=4)
    assert isinstance(result, AudioCodes)
    load.assert_called_once_with([clip.patcher], memory_required=123)
    clip.cond_stage_model.audio_generation_memory.assert_called_once_with(tokens, 37)
    clip.cond_stage_model.generate_audio_codes.assert_called_once_with(tokens, max_length=37, seed=8, audio_top_k=4)
    clip.cond_stage_model = torch.nn.Module()
    with pytest.raises(ValueError, match='does not support'):
        clip.generate_audio_codes(tokens)


def test_node_preserves_values_and_returns_codes():
    clip = SimpleNamespace(tokenize=Mock(return_value={'moss_delay': []}), generate_audio_codes=Mock(return_value='codes'))
    result = MossVoiceGenerator.execute(clip, ' text ', ' [direction] ', '', 3, 4, 0., 1., 0, 1.1)
    assert result.result == ('codes',)
    clip.tokenize.assert_called_once_with(' text ', instruction=' [direction] ', language=None)
    assert clip.generate_audio_codes.call_args.kwargs['max_length'] == 4
    assert MossVoiceGenerator.define_schema().outputs[0].io_type == 'AUDIO_CODES'


def test_zero_budget_uses_longest_prompt():
    stage = MossSpeechClipModel.__new__(MossSpeechClipModel)
    stage.transformer = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=12)))
    assert stage.generation_length({'moss_delay': [[0] * 3, [0] * 5]}, 0) == 7
    assert stage.generation_length({'moss_delay': [[0] * 13]}, 0) == 0
    assert stage.generation_length({'moss_delay': [[0] * 3]}, 99) == 99
