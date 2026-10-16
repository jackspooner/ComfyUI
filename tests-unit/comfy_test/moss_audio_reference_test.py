"""Opt-in parity tests: MOSS_REFERENCE_SOURCE must contain the pinned upstream src/.

Run in the isolated Transformers 4.57.1 reference environment. No model downloads.
"""
import os
from pathlib import Path
import sys

import pytest
import torch

reference_root = os.environ.get('MOSS_REFERENCE_SOURCE')
if not reference_root:
    pytest.skip('Set MOSS_REFERENCE_SOURCE to the pinned upstream checkout', allow_module_level=True)
sys.path.insert(0, reference_root)

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.moss_audio import MossAudioEncoder, AudioAdapter, MossAudioTokenizer, audio_features, audio_token_ids, AUDIO_TOKEN
from comfy.text_encoders.llama import Llama2_, Qwen3_8BConfig
from src.modeling_moss_audio import MossAudioEncoder as ReferenceEncoder, MossAudioModel
from src.configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig
from src.processing_moss_audio import MossAudioProcessor, MelConfig
from transformers import Qwen3Config, Qwen2Tokenizer, Qwen2TokenizerFast


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.fixture
def encoders():
    torch.manual_seed(42)
    cfg = MossAudioEncoderConfig(d_model=16, output_dim=16, encoder_attention_heads=2, encoder_ffn_dim=32,
                                encoder_layers=25, downsample_hidden_size=2)
    reference = ReferenceEncoder(cfg).float().eval()
    native = MossAudioEncoder(dtype=torch.float32, operations=manual_cast, d_model=16, n_heads=2, d_ff=32,
                              n_layers=25, conv_channels=2)
    native.load_state_dict(reference.state_dict())
    return native, reference


@pytest.mark.parametrize('length', [1,199,200,399,400,401,411,12000])
def test_audio_encoder_and_deepstack(encoders, length):
    native, reference = encoders
    features = torch.randn(128, length)
    expected = reference(features)
    for actual, wanted in zip(native(features), [expected.last_hidden_state[0], *[x[0] for x in expected.hidden_states]]):
        torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-4)


def test_prefill_and_cached_decode_with_deepstack():
    torch.manual_seed(42)
    audio = MossAudioEncoderConfig(d_model=16, output_dim=16, encoder_attention_heads=2, encoder_ffn_dim=32,
                                  encoder_layers=3, downsample_hidden_size=2, deepstack_encoder_layer_indexes=[0,1,2])
    language = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, rope_theta=1000000)
    reference = MossAudioModel(MossAudioConfig(audio_config=audio, language_config=language, adapter_hidden_size=32)).float().eval()
    config = Qwen3_8BConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2)
    config.head_dim = 8
    native = Llama2_(config, dtype=torch.float32, ops=manual_cast)
    state = dict(reference.language_model.state_dict())
    state['lm_head.weight'] = reference.lm_head.weight
    native.load_state_dict(state)
    mel = torch.randn(1,128,401)
    audio_outputs = reference.audio_encoder(mel)
    projected = reference.audio_adapter(audio_outputs.last_hidden_state)
    deepstack = [adapter(value)[0] for adapter,value in zip(reference.deepstack_audio_merger_list,audio_outputs.hidden_states)]
    ids = torch.randint(0,128,(1,60))
    mask = torch.zeros_like(ids,dtype=torch.bool)
    mask[:,3:3+projected.shape[1]] = True
    embeds = native.embed_tokens(ids)
    embeds[mask] = projected[0]
    hidden, _, cache = native(None, embeds=embeds, past_key_values=native.init_kv_cache(1,70,torch.device('cpu'),torch.float32), visual_pos_masks=mask, deepstack_embeds=deepstack)
    expected = reference(input_ids=ids,audio_data=mel,audio_input_mask=mask,use_cache=True)
    torch.testing.assert_close(native.lm_head(hidden),expected.logits,atol=1e-5,rtol=1e-4)
    for token in (2,8,15):
        ids = torch.tensor([[token]])
        hidden, _, cache = native(None, embeds=native.embed_tokens(ids), past_key_values=cache)
        expected = reference(input_ids=ids,past_key_values=expected.past_key_values,use_cache=True)
        torch.testing.assert_close(native.lm_head(hidden),expected.logits,atol=1e-5,rtol=1e-4)


@pytest.fixture
def processors():
    path = os.environ.get('MOSS_REFERENCE_MODEL')
    if not path:
        pytest.skip('Set MOSS_REFERENCE_MODEL to local tokenizer files')
    slow = Qwen2Tokenizer.from_pretrained(path,local_files_only=True)
    fast = Qwen2TokenizerFast.from_pretrained(path,local_files_only=True)
    assert slow.get_vocab() == fast.get_vocab()
    data = fast.backend_tokenizer.to_str().encode()
    native = MossAudioTokenizer(tokenizer_data={'tokenizer_json':torch.frombuffer(bytearray(data),dtype=torch.uint8)})
    return native,MossAudioProcessor(slow,mel_config=MelConfig(mel_dtype=torch.float32),enable_time_marker=True)


@pytest.mark.parametrize('samples',[16000,64001,480160,1920000])
def test_preprocessing_and_expanded_tokens(processors,samples):
    native, reference = processors
    waveform = torch.randn(samples)*.1
    actual = audio_features({'waveform':waveform[None,None], 'sample_rate':16000})
    expected = reference._extract_mel(waveform)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
    prompt = 'Describe foreground/background sounds and their order. café 音楽 (test:1.2)'
    tokens = native.tokenize_with_weights(prompt,thinking=True,audio={'waveform':waveform[None,None],'sample_rate':16000})['moss_audio'][0]
    expanded = []
    for token,_ in tokens:
        expanded.extend(audio_token_ids((actual.shape[1]+7)//8) if isinstance(token,dict) else [token])
    assert expanded == reference(text=prompt,audios=[waveform])['input_ids'][0].tolist()
    assert native.tokenize_with_weights(prompt,thinking=True)['moss_audio'][0] == [(i,1.) for i in reference(text=prompt)['input_ids'][0].tolist()]


def test_raw_and_system_prompts(processors):
    native,_ = processors
    raw='<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\nDescribe.<|im_end|>\n<|im_start|>assistant\n'
    result=native.tokenize_with_weights(raw,audio={},skip_template=True,thinking=False)['moss_audio'][0]
    assert sum(isinstance(t,dict) for t,_ in result)==1
    with pytest.raises(ValueError,match='placeholder'):
        native.tokenize_with_weights('raw prompt',audio={},skip_template=True)
    tokens=native.tokenize_with_weights('Hi',system_prompt='Only describe sounds.',thinking=False)['moss_audio'][0]
    text=native.decode([t for t,_ in tokens],skip_special_tokens=False)
    assert 'Only describe sounds.' in text
    assert text.endswith('<think>\n\n</think>\n\n')
