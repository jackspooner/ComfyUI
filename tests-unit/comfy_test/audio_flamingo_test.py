import os
from pathlib import Path

import pytest
import torch
from transformers import AudioFlamingo3EncoderConfig, Qwen2Config, Qwen2TokenizerFast, AutoProcessor
from transformers.models.audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3Encoder as ReferenceEncoder
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.audio_flamingo import AudioFlamingoEncoder, AudioFlamingoConfig, AudioFlamingoTokenizer, audio_features
from comfy.text_encoders.llama import Llama2_


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.mark.parametrize('length', [3, 7, 17, 31, 32])
def test_encoder_mask_pool_and_positions(length):
    torch.manual_seed(9)
    config = AudioFlamingo3EncoderConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=2, max_source_positions=16, num_mel_bins=8)
    config._attn_implementation = 'sdpa'
    reference = ReferenceEncoder(config).float().eval()
    native = AudioFlamingoEncoder(n_mels=8, n_ctx=16, n_state=16, n_head=2, n_layer=2,
                                  dtype=torch.float32, device='cpu', operations=manual_cast).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    features = torch.randn(1,8,32)
    mask = torch.arange(32)[None] < length
    expected = reference(features, mask).last_hidden_state[:, :((length + 1)//2)//2]
    torch.testing.assert_close(native(features,length),expected,atol=1e-5,rtol=1e-4)


def test_native_qwen_prefill_and_cached_decode():
    torch.manual_seed(10)
    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2,vocab_size=128,rope_theta=1000000)
    config._attn_implementation = 'sdpa'
    reference = Qwen2ForCausalLM(config).float().eval()
    cfg = AudioFlamingoConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                             num_attention_heads=4,num_key_value_heads=2,vocab_size=128)
    cfg.head_dim = 8
    native = Llama2_(cfg,dtype=torch.float32,device='cpu',ops=manual_cast).eval()
    state = reference.model.state_dict();state['lm_head.weight'] = reference.lm_head.weight
    native.load_state_dict(state,strict=True)
    ids = torch.randint(0,128,(1,7))
    expected = reference(ids,use_cache=True)
    hidden,_,cache = native(None,embeds=native.embed_tokens(ids),past_key_values=native.init_kv_cache(1,16,torch.device('cpu'),torch.float32))
    torch.testing.assert_close(native.lm_head(hidden),expected.logits,atol=1e-5,rtol=1e-4)
    for token in [3,5,7]:
        ids = torch.tensor([[token]])
        expected = reference(ids,past_key_values=expected.past_key_values,use_cache=True)
        hidden,_,cache = native(None,embeds=native.embed_tokens(ids),past_key_values=cache)
        torch.testing.assert_close(native.lm_head(hidden),expected.logits,atol=1e-5,rtol=1e-4)


def test_full_checkpoint_detection_guards():
    from comfy.sd import detect_te_model, TEModel
    signature = {'audio_tower.conv1.weight':1280,'audio_tower.embed_positions.weight':1500,
                 'multi_modal_projector.linear_1.weight':3584,'model.embed_tokens.weight':151672,
                 'model.layers.27.self_attn.q_proj.weight':3584,'model.lm_head.weight':151672}
    state = {k:torch.empty(v,1,device='meta') for k,v in signature.items()}
    state['model.layers.0.self_attn.k_proj.bias'] = torch.empty(512, device='meta')
    assert detect_te_model(state) == TEModel.AUDIO_FLAMINGO
    assert TEModel.AUDIO_FLAMINGO != TEModel.AUK
    for key in signature:
        assert detect_te_model({k:v for k,v in state.items() if k != key}) != TEModel.AUDIO_FLAMINGO


@pytest.mark.parametrize('samples', [481, 16000, 480001])
def test_offline_processor_parity(samples):
    path = os.environ.get('AUDIO_FLAMINGO_REFERENCE_MODEL')
    if not path:
        pytest.skip('Set AUDIO_FLAMINGO_REFERENCE_MODEL to the already provisioned HF snapshot')
    reference = AutoProcessor.from_pretrained(path,local_files_only=True)
    tokenizer = Qwen2TokenizerFast.from_pretrained(path,local_files_only=True)
    native = AudioFlamingoTokenizer(tokenizer_data={'tokenizer_json':torch.frombuffer(bytearray(tokenizer.backend_tokenizer.to_str().encode()),dtype=torch.uint8)})
    waveform = torch.randn(1,1,samples)
    audio = {'waveform':waveform,'sample_rate':16000}
    inputs = reference(text='<sound>Describe the audio.',audio=[waveform[0,0].numpy()],sampling_rate=16000,return_tensors='pt')
    chunks = audio_features(audio)
    torch.testing.assert_close(torch.cat([x for x,_ in chunks]),inputs.input_features,atol=1e-6,rtol=1e-6)
    lengths = [(n+1)//2//2 for _,n in chunks]
    assert sum(lengths) == int((inputs.input_ids==151669).sum())
    actual = native.tokenize_with_weights('Describe the audio.',audio=audio)['audio_flamingo'][0]
    ids = []
    for value,_ in actual:
        ids.extend([151669]*sum(lengths) if isinstance(value,dict) else [value])
    template = reference.apply_chat_template([{'role':'user','content':[{'type':'audio','audio':waveform[0,0].numpy()},{'type':'text','text':'Describe the audio.'}]}],tokenize=False,add_generation_prompt=True)
    expected = reference(text=template,audio=[waveform[0,0].numpy()],sampling_rate=16000,return_tensors='pt').input_ids[0].tolist()
    assert ids == expected
