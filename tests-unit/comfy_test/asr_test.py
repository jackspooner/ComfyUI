"""Small CPU reference comparisons; no serving state, weights or GPU inference."""
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig, Qwen3Config
from transformers.generation import GenerationMixin
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration as WhisperReference
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder, Qwen3ASRMultiModalProjector
from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig

from comfy.cli_args import args
args.cpu = True
from comfy.asr import qwen, whisper
from comfy.asr.common import patcher
from comfy.ops import manual_cast


@pytest.fixture(autouse=True)
def execution():
    with torch.set_grad_enabled(False):
        yield


def whisper_fixture():
    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_source_positions=16, max_target_positions=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        decoder_start_token_id=1, begin_suppress_tokens=[], suppress_tokens=[])
    cfg._attn_implementation = 'sdpa'
    torch.manual_seed(3)
    ref = WhisperReference(cfg).float().eval()
    encoder, decoder = whisper.networks(cfg, torch.float32)
    encoder.load_state_dict(ref.model.encoder.state_dict(), strict=True)
    state = ref.model.decoder.state_dict()
    state['proj_out.weight'] = ref.proj_out.weight
    decoder.load_state_dict(state, strict=True)
    return cfg, ref, encoder, decoder


def test_whisper_encoder_decoder_cache_and_alignment_heads():
    cfg, ref, encoder, decoder = whisper_fixture()
    mel = torch.randn(2, 8, 32)
    actual, expected = encoder(mel), ref.model.encoder(mel).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
    a = decoder(ids, encoder_hidden_states=actual, alignment_heads=[(1, 0)])
    b = ref.model.decoder(ids, encoder_hidden_states=expected, use_cache=True)
    torch.testing.assert_close(a.last_hidden_state, b.last_hidden_state, atol=1e-6, rtol=1e-5)
    assert len(a.cross_attentions) == 1 and a.cross_attentions[0].shape == (2, 1, 3, 16)
    # Beam reordering must move both self and cross caches without sharing state.
    order = torch.tensor([1, 0])
    a.past_key_values.reorder_cache(order)
    b.past_key_values.reorder_cache(order)
    actual, expected = actual[order], expected[order]
    for token in (5, 7, 9):
        ids = torch.full((2, 1), token)
        a = decoder(ids, encoder_hidden_states=actual, past_key_values=a.past_key_values)
        b = ref.model.decoder(ids, encoder_hidden_states=expected, past_key_values=b.past_key_values, use_cache=True)
        torch.testing.assert_close(a.last_hidden_state, b.last_hidden_state, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('beams', [1, 3])
def test_whisper_upstream_generation_with_native_networks(beams):
    cfg, ref, encoder, decoder = whisper_fixture()
    model = whisper.Model(patcher(encoder, encoder.state_dict(), torch.device('cpu')),
                          patcher(decoder, decoder.state_dict(), torch.device('cpu')), cfg,
                          ref.generation_config.to_dict(), None, None, torch.float32)
    generation = whisper.Generation(model)
    features = torch.randn(1, 8, 32)
    with patch('comfy.asr.whisper.mm.load_models_gpu') as admission:
        actual = GenerationMixin.generate(generation, inputs=features, max_new_tokens=5, num_beams=beams)
    expected = GenerationMixin.generate(ref, inputs=features, max_new_tokens=5, num_beams=beams)
    assert actual.tolist() == expected.tolist()
    assert {call.args[0][0] for call in admission.call_args_list} == {model.encoder, model.decoder}


@pytest.mark.parametrize('frames', [32, 65])
def test_whisper_timestamp_and_longform_generation(frames):
    cfg, ref, encoder, decoder = whisper_fixture()
    config = {**ref.generation_config.to_dict(), 'no_timestamps_token_id': 16,
              'lang_to_id': {'<|en|>': 3}, 'task_to_id': {'transcribe': 4, 'translate': 5},
              'is_multilingual': True, 'prev_sot_token_id': 7, '_from_model_config': False}
    model = whisper.Model(patcher(encoder, encoder.state_dict(), torch.device('cpu')),
                          patcher(decoder, decoder.state_dict(), torch.device('cpu')), cfg,
                          config, None, None, torch.float32)
    native = whisper.Generation(model)
    from transformers import GenerationConfig
    ref.generation_config = GenerationConfig(**config)
    features = torch.randn(1, 8, frames)
    kwargs = dict(language='en', task='transcribe', num_beams=2, max_new_tokens=8,
                  return_timestamps=True, return_segments=True, return_dict_in_generate=True,
                  return_token_timestamps=False)
    with patch('comfy.asr.whisper.mm.load_models_gpu'):
        actual = native.generate(features, **kwargs)
    expected = ref.generate(features, **kwargs)
    assert actual['sequences'].tolist() == expected['sequences'].tolist()
    assert len(actual['segments']) == len(expected['segments'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_qwen_logits_and_cached_decode(dtype):
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=128, max_position_embeddings=128,
                      rope_theta=1000000)
    cfg._attn_implementation = 'sdpa'
    torch.manual_seed(5)
    ref = Qwen3ForCausalLM(cfg).to(dtype).eval()
    native = qwen.Decoder(cfg.to_dict(), 128, dtype, 'cpu')
    state = ref.model.state_dict()
    state['lm_head.weight'] = ref.lm_head.weight
    native.load_state_dict(state, strict=True)
    cache = native.init_kv_cache(1, 16, torch.device('cpu'), dtype)
    previous = None
    for ids in (torch.tensor([[1, 2, 3]]), torch.tensor([[4]]), torch.tensor([[7]])):
        hidden, _, cache = native(ids, past_key_values=cache)
        expected = ref(ids, past_key_values=previous, use_cache=True)
        previous = expected.past_key_values
        tolerance = (1e-6, 1e-5) if dtype == torch.float32 else (2e-3, 1e-2)
        torch.testing.assert_close(native.lm_head(hidden), expected.logits, atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.parametrize('length', [100, 231, 800, 901])
def test_qwen_audio_encoder(length):
    cfg = Qwen3ASREncoderConfig(d_model=16, encoder_layers=2, encoder_attention_heads=2,
        encoder_ffn_dim=64, num_mel_bins=8, downsample_hidden_size=4, output_dim=16,
        n_window=50, n_window_infer=800)
    cfg._attn_implementation = 'sdpa'
    ref = Qwen3ASREncoder(cfg).float().eval()
    projector = Qwen3ASRMultiModalProjector(SimpleNamespace(audio_config=cfg)).float().eval()
    native = qwen.Encoder({**cfg.to_dict(), 'conv_chunksize': 500}, 'cpu', torch.float32, manual_cast)
    # Transformers' integrated model separates the final projection from the tower.
    state = ref.state_dict()
    state.update({'proj' + key.removeprefix('linear_'): value for key, value in projector.state_dict().items()})
    native.load_state_dict(state, strict=True)
    features = torch.randn(8, length)
    padded = torch.nn.functional.pad(features, (0, -length % 100))[None]
    mask = torch.arange(padded.shape[-1])[None] < length
    expected = projector(ref(padded, input_features_mask=mask).last_hidden_state)
    torch.testing.assert_close(native(features), expected, atol=1e-5, rtol=1e-4)


def test_chunking_has_no_gaps_and_finds_low_amplitude_sample():
    value = torch.ones(32000)
    value[10000] = 0
    chunks = list(qwen.split_audio(value, 1))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(value)
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))
    assert torch.equal(torch.cat([value[start:end] for start, end in chunks]), value)


def test_qwen_language_controls():
    assert qwen.language_name('en') == 'English'
    assert qwen.language_name('auto') is None
    with pytest.raises(ValueError):
        qwen.language_name('not-a-language')


def test_whisper_word_alignment_uses_n_plus_one_boundaries():
    cfg, ref, encoder, decoder = whisper_fixture()
    cfg.eos_token_id = 60
    class Tokenizer:
        prefix_tokens = [1, 3, 4]
        def set_prefix_tokens(self, **kwargs):
            pass
    extractor = lambda *args, **kwargs: SimpleNamespace(input_features=torch.randn(1, 8, 32))
    model = whisper.Model(patcher(encoder, encoder.state_dict(), torch.device('cpu')),
                          patcher(decoder, decoder.state_dict(), torch.device('cpu')), cfg,
                          {'alignment_heads': [[0, 0], [1, 1]]}, Tokenizer(), extractor, torch.float32)
    with (patch('comfy.asr.whisper.mm.load_models_gpu'),
          patch('transformers.models.whisper.tokenization_whisper._combine_tokens_into_words',
                return_value=(['one', 'two'], [[6], [7, 8]], [[0], [1, 2]]))):
        words = whisper.align_words(model, torch.zeros(5120), [6, 7, 8])
    assert [word['word'] for word in words] == ['one', 'two']
    assert all(0 <= word['start'] <= word['end'] <= .32 for word in words)


def test_whisper_generation_failure_cleans_execution_prefetch():
    cfg, ref, encoder, decoder = whisper_fixture()
    extractor = lambda *args, **kwargs: SimpleNamespace(input_features=torch.zeros(1, 8, 32),
                                                      attention_mask=torch.ones(1, 32))
    model = whisper.Model(patcher(encoder, encoder.state_dict(), torch.device('cpu')),
                          patcher(decoder, decoder.state_dict(), torch.device('cpu')), cfg,
                          ref.generation_config.to_dict(), None, extractor, torch.float32)
    with (patch.object(whisper.Generation, 'generate', side_effect=RuntimeError('interrupted fixture')),
          patch('comfy.asr.whisper.prefetch.cleanup_prefetch_queues') as cleanup):
        with pytest.raises(RuntimeError, match='interrupted fixture'):
            whisper.transcribe_many(model, [{'waveform': torch.zeros(1, 1, 1600), 'sample_rate': 16000}])
    cleanup.assert_called_once()


def test_whisper_batches_independent_windows_of_one_recording():
    cfg, ref, encoder, decoder = whisper_fixture()
    extractor = lambda clips, **kwargs: SimpleNamespace(input_features=torch.zeros(len(clips), 8, 32),
                                                       attention_mask=torch.ones(len(clips), 32))
    model = whisper.Model(patcher(encoder, encoder.state_dict(), torch.device('cpu')),
                          patcher(decoder, decoder.state_dict(), torch.device('cpu')), cfg,
                          ref.generation_config.to_dict(), None, extractor, torch.float32)
    def generated(features, **kwargs):
        return {'segments': [[] for _ in features]}
    with (patch.object(whisper.Generation, 'generate', side_effect=generated) as generate,
          patch('comfy.asr.whisper.mm.load_models_gpu')):
        result = whisper.transcribe_many(model, [{'waveform': torch.zeros(1, 1, 15000), 'sample_rate': 16000}], batch_size=2)
    assert len(result) == 1
    assert [call.args[0].shape[0] for call in generate.call_args_list] == [2, 1]
