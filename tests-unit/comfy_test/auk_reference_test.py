"""Opt-in upstream comparisons; uses local pinned sources and never downloads.

AUK_REFERENCE_SOURCE: AuK checkout at 9481474f107ee4d8dec959b3aed05b0627cebbf8.
AUK_REFERENCE_WEIGHTS: optional complete AuK snapshot, for the full VAE comparison.
Run with Transformers 4.57.1 and the upstream torchdiffeq/x-transformers dependencies.
"""
import json
import os
from pathlib import Path
import sys

import pytest
import torch

source = os.environ.get('AUK_REFERENCE_SOURCE')
if source is None:
    pytest.skip('Set AUK_REFERENCE_SOURCE to the pinned upstream checkout', allow_module_level=True)
sys.path.insert(0, str(Path(source) / 'src'))

from comfy.cli_args import args
args.cpu = True
from comfy import ops
from comfy.ldm.auk.model import AuK
from comfy.ldm.auk.vae import AuKVAE
from comfy.text_encoders.auk import OmniAudioEncoder, AuKTokenizer, audio_features
from comfy.text_encoders.llama import Llama2_, Qwen25_3BConfig
from scripts.convert_auk import load_complete, fold_weight_norm
from auk.model.flux2_edit import Flux2Edit
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig
from transformers import Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniTextConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder, Qwen2_5OmniThinkerTextModel


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.mark.parametrize('reference_frames', [0, 5])
def test_transformer_and_four_euler_steps(reference_frames):
    torch.manual_seed(17)
    original = Flux2Edit(dim=32, heads=2, dim_head=16, ff_mult=2, latent_dim=64, text_hidden_dim=16,
                        num_layers=2, num_single_layers=2, dropout=0, attn_mask_enabled=True).eval()
    # Upstream initializes training output/modulation to zero; exercise nonzero paths.
    for name, value in original.named_parameters():
        if '.linear.' in name or name.startswith('proj_out.'):
            torch.nn.init.normal_(value, std=.05)
    # Published Flash frequencies have a BF16 round trip despite FP32 storage.
    original.rotary_embed.inv_freq.copy_(original.rotary_embed.inv_freq.bfloat16().float())
    native = AuK(dim=32, heads=2, head_dim=16, ff_mult=2, latent_dim=64, text_hidden_dim=16,
                 num_layers=2, num_single_layers=2, operations=ops.manual_cast)
    native.load_state_dict(original.state_dict(), strict=True)
    initial_noise = torch.randn(1, 10, 64)
    original_x, native_x = initial_noise.clone(), initial_noise.transpose(1, 2).clone()
    context = torch.randn(1, 7, 16)
    reference = torch.randn(1, reference_frames, 64) if reference_frames else None
    reference_mask = torch.ones(1, reference_frames, dtype=torch.bool) if reference_frames else None
    if reference_mask is not None:
        reference_mask[:, -1] = False
    mask = torch.tensor([[True] * 6 + [False]])
    projected = native.preprocess_text(context)
    sigmas = torch.tensor([1., .9238795042037964, .7071067690849304, .3826833963394165, 0.])
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        expected = original(original_x, text=context, time=(1-sigma).reshape(1), ref=reference,
                            ref_mask=reference_mask, c_mask=mask)
        actual = -native(native_x, sigma.reshape(1), projected, attention_mask=mask.float(),
                         reference=None if reference is None else reference.transpose(1,2),
                         reference_mask=None if reference_mask is None else reference_mask.float())
        torch.testing.assert_close(actual.transpose(1,2), expected, atol=1e-5, rtol=1e-4)
        original_x += expected * (sigma-next_sigma)
        native_x += actual * (sigma-next_sigma)
        torch.testing.assert_close(native_x.transpose(1,2), original_x, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('length', [100, 200, 201, 499])
def test_qwen_audio_chunk_boundaries(length):
    torch.manual_seed(42)
    config = Qwen2_5OmniAudioEncoderConfig(d_model=32, encoder_attention_heads=4, encoder_layers=2,
                                         encoder_ffn_dim=128, output_dim=32)
    config._attn_implementation = 'sdpa'
    original = Qwen2_5OmniAudioEncoder(config).eval()
    native = OmniAudioEncoder(dim=32, heads=4, layers=2, output_dim=32, operations=ops.manual_cast)
    native.load_state_dict({k:v for k,v in original.state_dict().items() if not k.startswith('audio_bos_eos_token.')}, strict=True)
    features = torch.randn(128, length)
    expected = original(features, feature_lens=torch.tensor([length]), aftercnn_lens=torch.tensor([(length+1)//2])).last_hidden_state
    torch.testing.assert_close(native(features), expected, atol=1e-6, rtol=1e-5)


def test_qwen_text_hidden_layers():
    torch.manual_seed(9)
    settings = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                    num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=1e6)
    cfg = Qwen2_5OmniTextConfig(**settings, rope_scaling={'mrope_section':[1,1,2], 'rope_type':'default'})
    cfg._attn_implementation = 'sdpa'
    original = Qwen2_5OmniThinkerTextModel(cfg).eval()
    config = Qwen25_3BConfig(**settings)
    config.head_dim = 8
    native = Llama2_(config, dtype=torch.float32, device='cpu', ops=ops.manual_cast)
    native.load_state_dict(original.state_dict(), strict=True)
    tokens = torch.randint(0,128,(1,15))
    expected = original(tokens, output_hidden_states=True).hidden_states
    _, actual = native(tokens, intermediate_output='all', final_layer_norm_intermediate=False, dtype=torch.float32)
    for index, value in enumerate(expected):
        torch.testing.assert_close(actual[:, index], value, atol=1e-5, rtol=1e-4)


def test_full_vae_checkpoint():
    root = os.environ.get('AUK_REFERENCE_WEIGHTS')
    if root is None:
        pytest.skip('Set AUK_REFERENCE_WEIGHTS to test the complete VAE checkpoint')
    weights = load_complete(Path(root) / 'vae.safetensors')
    original = BigVGANFlowVAE(BigVGANFlowVAEConfig()).eval()
    original.load_state_dict(weights, strict=True)
    native = AuKVAE()
    native.load_state_dict(fold_weight_norm(weights)[0], strict=True)
    waveform = torch.randn(1,1,4800) * .05
    torch.testing.assert_close(native.audio_encoder(waveform), original.audio_encoder(waveform), atol=0, rtol=0)
    torch.manual_seed(42)
    expected, _ = original.encoding_and_normalization(waveform)
    torch.manual_seed(42)
    actual = native.encode(waveform)
    torch.testing.assert_close(actual.transpose(1,2), expected, atol=0, rtol=0)
    expected_audio = original.inference_from_latents(original.denormalize(expected).transpose(1,2))
    torch.testing.assert_close(native.decode(actual), expected_audio, atol=0, rtol=0)


@pytest.mark.parametrize('with_audio', [False, True])
def test_tokenizer_and_audio_features(with_audio):
    root = os.environ.get('AUK_QWEN_SOURCE')
    if root is None:
        pytest.skip('Set AUK_QWEN_SOURCE to test the original tokenizer/preprocessor')
    path = Path(root)
    template = json.loads((path/'chat_template.json').read_text())['chat_template']
    data = {'tokenizer_json':torch.tensor(list((path/'tokenizer.json').read_bytes()), dtype=torch.uint8),
            'auk_config_json':torch.tensor(list(json.dumps({'chat_template':template}).encode()), dtype=torch.uint8)}
    native = AuKTokenizer(tokenizer_data=data)
    original = Qwen2_5OmniProcessor.from_pretrained(path, local_files_only=True)
    instruction = 'Say the following with the same voice: "Hello there."'
    waveform = torch.randn(1,1,32101)*.05
    audio = {'waveform':waveform,'sample_rate':16000} if with_audio else None
    content = [{'type':'text','text':instruction if with_audio else instruction+'|<no_prompt_audio>|'}]
    if with_audio:
        content.append({'type':'audio','audio':'local-reference'})
    text = original.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
    inputs = original(text=[text], audio=[waveform[0,0].numpy()] if with_audio else None, padding=True, return_tensors='pt')
    tokens = native.tokenize_with_weights(instruction, audio=audio)['auk'][0]
    actual = []
    for token,_ in tokens:
        if isinstance(token,dict):
            features = audio_features(audio)
            count = inputs.feature_attention_mask.sum().item()
            torch.testing.assert_close(features, inputs.input_features[0,:,:count], atol=0, rtol=0)
            actual.extend([original.tokenizer.convert_tokens_to_ids('<|AUDIO|>')] * (((count+1)//2)//2))
        else:
            actual.append(token)
    assert actual == inputs.input_ids[0].tolist()


def test_bf16_transformer_keeps_float32_time_precision():
    torch.manual_seed(14)
    original = Flux2Edit(dim=32, heads=2, dim_head=16, ff_mult=2, latent_dim=64, text_hidden_dim=16,
                        num_layers=1, num_single_layers=1, dropout=0, attn_mask_enabled=True).eval()
    for name, value in original.named_parameters():
        if '.linear.' in name or name.startswith('proj_out.'):
            torch.nn.init.normal_(value, std=.05)
    native = AuK(dim=32, heads=2, head_dim=16, ff_mult=2, text_hidden_dim=16, num_layers=1,
                 num_single_layers=1, operations=ops.manual_cast, dtype=torch.bfloat16)
    native.load_state_dict(original.state_dict(), strict=True)
    context = torch.randn(1, 7, 16)
    x = torch.randn(1, 64, 10)
    sigma = torch.tensor([.9238795042037964])
    actual = native(x.bfloat16(), sigma, native.preprocess_text(context.bfloat16()))
    assert actual.dtype == torch.bfloat16
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = original(x.transpose(1,2), text=context, time=1-sigma)
    # FP32 source norms under autocast and BF16 native weights have different rounding.
    torch.testing.assert_close(-actual.transpose(1,2).float(), expected.float(), atol=.015, rtol=.03)
