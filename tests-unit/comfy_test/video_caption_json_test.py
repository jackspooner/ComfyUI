"""Native caption loops must reject illegal tokens and incomplete JSON."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Tokenizer

from comfy.video_caption import VideoCaptioner
from comfy.video_caption_internvideo3 import InternVideoCaptioner
from comfy.video_caption_json import prepare_json, json_result, select_token
from comfy.text_json import StructuredOutputError

pytest.importorskip('llguidance')
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.dumps({'type': 'object', 'properties': {'caption': {'type': 'string', 'minLength': 1}},
    'required': ['caption'], 'additionalProperties': False})
VOCAB = 151936
EOS = 151645


@pytest.fixture(scope='module')
def tokenizer():
    return Qwen2Tokenizer.from_pretrained(str(ROOT / 'comfy/text_encoders/qwen25_tokenizer'), local_files_only=True).backend_tokenizer


def prepared(tokenizer, family):
    return prepare_json(tokenizer, family, VOCAB, {EOS}, 'json_schema', SCHEMA)


class ScriptedLogits:
    """A model preferring premature EOS/control tokens over the desired answer."""
    def __init__(self, tokenizer, texts):
        self.rows = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
        self.step = 0

    def __call__(self, hidden):
        logits = torch.zeros((*hidden.shape[:-1], VOCAB))
        flat = logits.reshape(-1, VOCAB)
        flat[:, EOS] = 100
        flat[:, 151644] = 200  # im_start must never leak through the grammar
        for index, row in enumerate(self.rows):
            if self.step < len(row):
                flat[index, row[self.step]] = 10
        self.step += 1
        return logits


def test_salmonn_loop_masks_eos_and_finishes_on_last_budget_token(tokenizer):
    owner = object.__new__(VideoCaptioner)
    answer = '{"caption":"red square"}'
    logits = ScriptedLogits(tokenizer, [answer])
    owner.network = SimpleNamespace(config=SimpleNamespace(eos_token_id=EOS), dtype=torch.float32, logits=logits,
        model=SimpleNamespace(step=lambda hidden, *args, **kwargs: (hidden, [], {}),
            embed_tokens=lambda ids, **kwargs: torch.zeros(*ids.shape, 4)))
    attempt = prepared(tokenizer, 'salmonn2_pro_8b').new_attempt()
    ids = torch.tensor([[4, 5]])
    output = owner._salmonn(ids, torch.zeros(1, 2, 4), torch.zeros(3, 1, 2), 0, None, None, len(logits.rows[0]), attempt)
    result = json_result(attempt)
    assert result['text'] == answer
    assert result['structured_output']['finish_reason'] == 'grammar_complete'
    assert output == logits.rows[0]


def test_internvideo_loop_uses_schema_and_respects_budget(tokenizer, monkeypatch):
    import comfy.video_caption_internvideo3 as native
    owner = object.__new__(InternVideoCaptioner)
    owner.tokenizer, owner.load_device, owner.dtype = tokenizer, torch.device('cpu'), torch.float32
    owner.eos_tokens = {EOS, 151643}
    owner.patcher = object()
    owner.config = SimpleNamespace(video_token_id=151656, image_token_id=151655, vision_start_token_id=151652,
        vision_config=SimpleNamespace(spatial_merge_size=2))
    c = SimpleNamespace(vocab_size=VOCAB, max_position_embeddings=4096, kv_lora_rank_list=[6],
        qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4, num_attention_heads=2, hidden_size=4)
    answer = '{"caption":"blue"}'
    logits = ScriptedLogits(tokenizer, [answer])
    owner.network = SimpleNamespace(visual=lambda *args: torch.zeros(1, 4), model=SimpleNamespace(config=c,
        embed_tokens=lambda ids, **kwargs: torch.zeros(*ids.shape, 4),
        step=lambda hidden, *args: (hidden, []), lm_head=logits))
    monkeypatch.setattr(native.comfy.model_management, 'load_models_gpu', lambda *args, **kwargs: None)
    frames = torch.zeros(2, 32, 32, 3)
    result = owner.caption(frames, 2, None, 'Describe.', 2, 1024, len(logits.rows[0]), 1, 'json_schema', SCHEMA)
    assert result['text'] == answer
    assert result['structured'] == {'caption': 'blue'}
    logits.step = 0
    with pytest.raises(StructuredOutputError, match='incomplete'):
        owner.caption(frames, 2, None, 'Describe.', 2, 1024, 1, 1, 'json_schema', SCHEMA)


def test_pad_each_branch_has_private_matcher_and_finished_branches_stay_finished(tokenizer, monkeypatch):
    import comfy.video_caption as native
    class Session:
        def __init__(self, *args):
            self.start, self.end = torch.tensor([0]), torch.tensor([1])
            self.serial_evi_tgts, self.duration = [[]], 1
        def get_rope_index(self, ids, **kwargs):
            return torch.zeros(3, 1, ids.shape[1], dtype=torch.long), torch.tensor([[0]])
        def initialize(self, snapshots):
            pass
        def readout(self, *args):
            return [[len(self.serial_evi_tgts[0]), len(self.serial_evi_tgts[0])]]
        def _aggregate_interval_feature(self, *args):
            return torch.zeros(4)
        def _build_branch_prefix_visibility(self, ids, branches, prefix, device):
            return torch.ones(branches, prefix, dtype=torch.bool)
        def _branch_visibility_mask(self, branches, keys, prefix, dtype, device):
            return torch.zeros(1, 1, branches, keys)
    monkeypatch.setattr(native, 'PadSession', Session)
    owner = object.__new__(VideoCaptioner)
    owner.tokenizer = tokenizer
    config = SimpleNamespace(eos_token_id=EOS, evi_token_id=151660, evi_end_token_id=151661,
        shift_token_id=151662, audio_token_id=151663, video_token_id=151656, image_token_id=151655,
        vision_start_token_id=151652, vision_end_token_id=151653)
    scripted = ScriptedLogits(tokenizer, ['{"caption":"red"}', '{"caption":"a very bright blue square"}'])
    calls = [0]
    def step(hidden, positions, cache, **kwargs):
        calls[0] += 1
        size = hidden.shape[1] + (0 if cache is None else cache[0][2])
        return hidden, [(None, None, size)], {7: hidden}
    def logits(hidden):
        if calls[0] <= 3:
            value = torch.zeros(1, VOCAB)
            value[0, config.evi_token_id] = 1
            return value
        return scripted(hidden)
    owner.network = SimpleNamespace(config=config, dtype=torch.float32, logits=logits,
        model=SimpleNamespace(layers=[None] * 8, step=step, embed_tokens=lambda ids, **kwargs: torch.zeros(*ids.shape, 4)))
    result = owner._pad(torch.tensor([[4, 5]]), torch.zeros(1, 2, 4), torch.tensor([[2, 2, 2]]), 2,
        max(map(len, scripted.rows)), 2, 0, prepared(tokenizer, 'padcaptioner_3b'))
    assert [event['structured']['caption'] for event in result['events']] == ['red', 'a very bright blue square']
    assert all(not event['truncated'] and event['structured_output']['validated'] for event in result['events'])
    assert json.loads(result['text']) == result['structured']
    assert result['structured_output']['scope'] == 'event_caption'
    assert result['structured_output']['event_count'] == 2
    calls[0] = scripted.step = 0
    with pytest.raises(StructuredOutputError, match='incomplete'):
        owner._pad(torch.tensor([[4, 5]]), torch.zeros(1, 2, 4), torch.tensor([[2, 2, 2]]), 2, 1, 2, 0, prepared(tokenizer, 'padcaptioner_3b'))
    owner.network.logits = lambda hidden: torch.zeros(1, VOCAB)
    with pytest.raises(StructuredOutputError, match='planned no events'):
        owner._pad(torch.tensor([[4, 5]]), torch.zeros(1, 2, 4), torch.tensor([[2, 2, 2]]), 2, 32, 2, 0, prepared(tokenizer, 'padcaptioner_3b'))


def test_no_legal_logits_and_unsupported_formats_never_fall_back(tokenizer):
    attempt = prepared(tokenizer, 'salmonn2_pro_8b').new_attempt()
    with pytest.raises(ValueError, match='structured_output.logits'):
        select_token(torch.full((1, VOCAB), float('-inf')), attempt)
    with pytest.raises(StructuredOutputError, match='format'):
        prepare_json(tokenizer, 'salmonn2_pro_8b', VOCAB, {EOS}, 'json_object', '')
    with pytest.raises(StructuredOutputError, match='format'):
        prepare_json(tokenizer, 'salmonn2_pro_8b', VOCAB, {EOS}, 'text', SCHEMA)
    assert prepare_json(tokenizer, 'salmonn2_pro_8b', VOCAB, {EOS}, 'text', '') is None


@pytest.mark.parametrize('filename,family,stops', [
    ('video_salmonn_2_pro_8b_bf16.safetensors', 'salmonn2_pro_8b', {EOS}),
    ('padcaptioner_3b_bf16.safetensors', 'padcaptioner_3b', {EOS}),
    ('internvideo3_8b_instruct_bf16.safetensors', 'internvideo3_8b', {EOS, 151643})])
def test_actual_embedded_caption_tokenizers(filename, family, stops):
    directory = os.environ.get('VIDEO_CAPTION_JSON_FIXTURES')
    if not directory:
        pytest.skip('Set VIDEO_CAPTION_JSON_FIXTURES to existing converted packages')
    from safetensors import safe_open
    from tokenizers import Tokenizer
    with safe_open(str(Path(directory) / filename), framework='pt', device='cpu') as tensors:
        tokenizer = Tokenizer.from_str(tensors.get_tensor('tokenizer_json').numpy().tobytes().decode())
        vocab = tensors.get_slice('model.embed_tokens.weight').get_shape()[0]
    p = prepare_json(tokenizer, family, vocab, stops, 'json_schema', SCHEMA)
    attempt = p.new_attempt()
    for token in tokenizer.encode('{"caption":"中🙂"}', add_special_tokens=False).ids:
        assert attempt.allowed_mask(torch.zeros(1, vocab))[0, token]
        attempt.commit(token)
    assert json_result(attempt)['structured'] == {'caption': '中🙂'}
