"""Pinned Local v1.5 tokenizer and small-component comparisons, never full weights."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from transformers import Qwen2TokenizerFast

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.audio_codec import AudioCodes, AudioCodeSegment, MOSS_V1, MOSS_V2
from comfy.text_encoders.moss_speech import MossLocalV15Tokenizer, MossLocalV15ClipModel
from comfy.text_encoders.moss_tts_local_v15 import MossTTSLocalV15, NanoLocal, sample_v15_token
from scripts.convert_moss_local import expected_v15_shapes, mapped_v15_name, validate_v15_config


@pytest.fixture(scope='module')
def reference():
    selected = os.environ.get('MOSS_LOCAL_V15_REFERENCE')
    if not selected:
        pytest.skip('Requires pinned MOSS_LOCAL_V15_REFERENCE; full model weights are never loaded')
    source = Path(selected)
    manifest = json.loads((Path(__file__).parents[2] / 'scripts/moss_speech_local_v15.json').read_text())
    for name in ('configuration_moss_tts.py', 'modeling_moss_tts.py', 'processing_moss_tts.py', 'gpt2_decoder.py', 'qwen3_decoder.py'):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    package = types.ModuleType('moss_local_v15_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return (importlib.import_module(package.__name__+'.modeling_moss_tts'),
            importlib.import_module(package.__name__+'.processing_moss_tts'), source)


@pytest.fixture(autouse=True)
def deterministic():
    with torch.random.fork_rng(devices=[]), torch.set_grad_enabled(False):
        torch.manual_seed(7401)
        yield


def codes(value):
    return AudioCodes(MOSS_V2, (AudioCodeSegment(value.T[None].clone(), (len(value),)),))


@pytest.mark.parametrize('frames,continuation', [(None, False), (0, False), (3, False), (3, True)])
def test_exact_direct_prompt_pause_language_reference_and_continuation(reference, frames, continuation):
    model, processing, source = reference
    config = model.MossTTSLocalConfig.from_pretrained(str(source), local_files_only=True)
    processor = processing.MossTTSLocalProcessor(
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True), model_config=config)
    raw = torch.frombuffer(bytearray((source/'tokenizer.json').read_bytes()), dtype=torch.uint8)
    tokenizer = MossLocalV15Tokenizer(tokenizer_data=dict(tokenizer_json=raw, codebooks=12))
    values = torch.arange((frames or 0)*12).reshape(frames or 0, 12) % 1024
    controls = dict(text=' Keep {text} [pause 0.5s] 你好. ', language='English', tokens=73, instruction=' calm ')
    message = processor.build_user_message(reference=None if frames is None else values, **controls)
    conversation = [message]
    if continuation:
        conversation.append(processor.build_assistant_message(audio_codes_list=[values]))
    expected = processor([conversation], mode='continuation' if continuation else 'generation')['input_ids']
    prepared = tokenizer.tokenize_with_weights(**controls,
        references=None if frames is None else (codes(values),), continuation=codes(values) if continuation else None)
    owner = MossLocalV15ClipModel.__new__(MossLocalV15ClipModel)
    owner.transformer = types.SimpleNamespace(codebooks=12)
    owner.execution_device = torch.device('cpu')
    actual, mask = owner.prepare_inputs(prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert mask.all()
    wrong = AudioCodes(MOSS_V1, codes(values).segments)
    with pytest.raises(ValueError, match='identity'):
        tokenizer.tokenize_with_weights('text', references=(wrong,))


@pytest.mark.parametrize('bf16', [False, True])
def test_nanogpt2_prefix_and_static_cache_match(reference, bf16):
    model, _, _ = reference
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = model.GPT2Config(n_embd=32, n_inner=64, n_head=4, n_layer=1, n_positions=13,
        n_ctx=13, vocab_size=151936, position_embedding_type='rope', rope_base=1000000.,
        activation_function='silu', layer_norm_epsilon=1e-6, attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)
    original = model.MossTTSNanoGPT2Model(config, attn_implementation='eager').eval().to(dtype)
    original.wte = torch.nn.Identity()
    native = NanoLocal(32, 64, 4, dtype, 'cpu', manual_cast)
    native.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 8, 32).to(dtype)
    expected = original(inputs_embeds=x, position_ids=torch.arange(8)[None].expand(2, -1), use_cache=True)
    actual, cache = native(x)
    torch.testing.assert_close(actual, expected.last_hidden_state, rtol=.02 if bf16 else 1e-5, atol=.01 if bf16 else 2e-6)
    past = expected.past_key_values
    for position in range(8, 13):
        x = torch.randn(2, 1, 32).to(dtype)
        expected = original(inputs_embeds=x, position_ids=torch.full((2, 1), position), past_key_values=past, use_cache=True)
        past = expected.past_key_values
        actual, cache = native(x, cache)
        torch.testing.assert_close(actual, expected.last_hidden_state, rtol=.02 if bf16 else 1e-5, atol=.01 if bf16 else 2e-6)
    assert cache[0][2] == 13
    assert not any('cache' in name for name in vars(native))


@pytest.mark.parametrize('top_p,temperature,penalty', [(0., 1., 1.), (.8, 1.7, 1.2), (1., .7, .8)])
def test_sampler_exact_reference_and_private_rng(reference, top_p, temperature, penalty):
    model, _, _ = reference
    sampler = model.MossTTSLocalModel.__new__(model.MossTTSLocalModel)
    logits = torch.randn(2, 32)
    history = torch.tensor([[1, 1, 2, -1, 1024], [3, 4, 5, 5, 1024]])
    torch.manual_seed(91)
    expected = sampler._sample_next_token(logits, True, temperature, 9, top_p, history, penalty)
    before = torch.random.get_rng_state().clone()
    actual = sample_v15_token(logits, torch.Generator().manual_seed(91), temperature, top_p, 9, history, penalty)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), before)
    with pytest.raises(ValueError, match='positive'):
        sample_v15_token(logits, torch.Generator(), 0.)


def tiny_models(reference, dtype=torch.float32):
    model, _, _ = reference
    language = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        vocab_size=151936, max_position_embeddings=64, rope_theta=1000000., rms_norm_eps=1e-6)
    local = dict(n_embd=32, n_inner=64, n_head=4, n_layer=1, vocab_size=151936,
        position_embedding_type='rope', rope_base=1000000., activation_function='silu',
        layer_norm_epsilon=1e-6, attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)
    config = model.MossTTSLocalConfig(qwen3_config=language, gpt2_config=local,
        attn_implementation='eager', local_transformer_attn_implementation='eager')
    original = model.MossTTSLocalModel(config).eval().to(dtype)
    native = MossTTSLocalV15(dict(language={k: v for k, v in language.items() if k != 'head_dim'},
        local=dict(dim=32, inner=64, heads=4)), dtype, 'cpu', manual_cast)
    native.load_state_dict({mapped_v15_name(k): v for k, v in original.state_dict().items()}, strict=True)
    return original, native


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_global_cached_hidden_and_local_heads_with_padding(reference, dtype):
    original, native = tiny_models(reference, dtype)
    ids = torch.randint(0, 1025, (2, 7, 13))
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, :2] = False
    torch.testing.assert_close(native.combined_embeddings(ids), original._build_inputs_embeds(ids), rtol=0, atol=0)
    cache = native.model.init_kv_cache(2, 9, torch.device('cpu'), dtype)
    original_cache = None
    for start, end in [(0, 4), (4, 5), (5, 6), (6, 7)]:
        actual, cache = native.global_hidden(ids[:, start:end], mask[:, :end], cache)
        expected = original.transformer(inputs_embeds=original._build_inputs_embeds(ids[:, start:end]),
            attention_mask=mask[:, :end], past_key_values=original_cache, use_cache=True)
        original_cache = expected.past_key_values
        tolerance = dict(rtol=.03, atol=.02) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=3e-6)
        torch.testing.assert_close(actual, expected.last_hidden_state, **tolerance)
        local, local_cache = native.local_transformer(actual[:, -1:])
        expected_local, expected_cache = original._decode_local_hidden_states_with_cache(expected.last_hidden_state[:, -1:])
        for channel in range(12):
            torch.testing.assert_close(native.audio_lm_heads[channel](local[:, -1]),
                original.audio_lm_heads[channel](expected_local[:, -1]), **tolerance)
            torch.testing.assert_close(native.local_text_lm_head(local[:, -1]),
                original.local_text_lm_head(expected_local[:, -1]), **tolerance)
            if channel < 11:
                value = ids[:, end-1, channel+1].clamp_max(1023)
                local, local_cache = native.local_transformer(native.audio_embeddings[channel](value)[:, None], local_cache)
                expected_local, expected_cache = original._decode_local_hidden_states_with_cache(
                    original.audio_embeddings[channel](value)[:, None], expected_cache)


@pytest.mark.parametrize('budget', [0, 1, 5])
def test_generation_protocol_continuation_private_rng_and_cancellation(reference, budget):
    from unittest.mock import patch
    from comfy import model_management
    original, native = tiny_models(reference)
    # Exercise a finished row alongside a continuing child, then terminate both.
    # Audio logits/sampling still use the actual tiny pinned/native networks.
    class ScriptedText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = 0

        def forward(self, hidden):
            logits = torch.full((len(hidden), 2), -100.)
            logits[0, 1] = 100.
            logits[1, 1 if self.step == 3 else 0] = 100.
            self.step += 1
            return logits

    ids = torch.full((2, 5, 13), 1024, dtype=torch.int64)
    ids[..., 0] = torch.tensor([151644, 151669, 151656, 151669, 151656])
    ids[:, -1, 1:] = 19  # continuation must not enter generated repetition history
    mask = torch.ones(2, 5, dtype=torch.bool)
    controls = dict(text_temperature=.7, text_top_p=.8, text_top_k=1,
        audio_temperature=1.7, audio_top_p=.8, audio_top_k=25, audio_repetition_penalty=1.2)
    original.local_text_lm_head = ScriptedText()
    torch.manual_seed(7401)
    expected = original.generate(ids, mask, max_new_tokens=budget, **controls)
    before = torch.random.get_rng_state().clone()
    for _ in range(2):
        native.local_text_lm_head = ScriptedText()
        actual = native.generate_local(ids, mask, budget, 7401, **controls)
        for (prefix, value), (expected_prefix, expected_value) in zip(actual, expected):
            assert prefix == expected_prefix == 1
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
            assert value.device.type == 'cpu'
        assert torch.equal(torch.random.get_rng_state(), before)
    native.local_text_lm_head = ScriptedText()
    if budget:
        with patch.object(model_management, 'throw_exception_if_processing_interrupted',
                side_effect=model_management.InterruptProcessingException):
            with pytest.raises(model_management.InterruptProcessingException):
                native.generate_local(ids, mask, budget, 7401, **controls)
        native.local_text_lm_head = ScriptedText()
        recovered = native.generate_local(ids, mask, budget, 7401, **controls)
        assert torch.equal(recovered[1][1], expected[1][1])


def test_native_architecture_inventory_and_source_controls(reference):
    model, _, source = reference
    from safetensors import safe_open
    config = json.loads((source/'config.json').read_text())
    validate_v15_config(config)
    with safe_open(source/'model.safetensors', framework='pt', device='cpu') as saved:
        expected = expected_v15_shapes()
        assert set(saved.keys()) == set(expected)
        for key, shape in expected.items():
            assert list(saved.get_slice(key).get_shape()) == shape
    with torch.device('meta'):
        native = MossTTSLocalV15({}, torch.bfloat16, 'meta', manual_cast)
    actual = native.state_dict()
    from comfy.sd import detect_te_model, TEModel
    assert detect_te_model(actual) == TEModel.MOSS_LOCAL_V15
    assert set(actual) == {mapped_v15_name(key) for key in expected}
    assert len(actual) == 438
    for key, shape in expected.items():
        assert list(actual[mapped_v15_name(key)].shape) == shape
    assert native.text_lm_head.weight is native.model.embed_tokens.weight
    assert all(head.weight is embedding.weight for head, embedding in zip(native.audio_lm_heads, native.audio_embeddings))
