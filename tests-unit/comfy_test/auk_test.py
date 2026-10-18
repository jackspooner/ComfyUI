import json
import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from comfy.cli_args import args
args.cpu = True
from comfy import model_detection
from comfy.model_sampling import ModelSamplingDiscreteFlow, CONST
from comfy.text_encoders import auk_prompt
from comfy_extras.nodes_auk import EmptyAuKLatentAudio, AuKPromptHelper
from comfy_extras.nodes_audio import vae_decode_audio
from scripts import convert_auk as converter


def test_conversion_payload_and_no_overwrite(tmp_path):
    values = {'weight': torch.randn(4, 8), 'bias': torch.randn(4)}
    output = tmp_path / 'model.safetensors'
    receipt = converter.write_component(values, output, {'format': 'pt'})
    assert receipt['bytes'] == output.stat().st_size
    assert receipt['sha256'] == converter.sha256(output)
    for key, value in load_file(output).items():
        assert torch.equal(value, values[key])
    with pytest.raises(FileExistsError):
        converter.write_component(values, output, {})


def test_weight_norm_accounting():
    values = {'conv.weight_v': torch.randn(4, 3, 5), 'conv.weight_g': torch.rand(4, 1, 1),
              'conv.bias': torch.randn(4), 'flow.unused': torch.randn(2)}
    result, actions = converter.fold_weight_norm(values)
    assert actions.keys() == values.keys()
    assert actions['flow.unused']['action'] == 'excluded'
    expected = values['conv.weight_v'] * values['conv.weight_g'] / values['conv.weight_v'].norm(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(result['conv.weight'], expected)
    with pytest.raises(ValueError, match='magnitude'):
        converter.fold_weight_norm({'conv.weight_v': values['conv.weight_v']})


@pytest.mark.parametrize('fault', ['pointer', 'truncated'])
def test_incomplete_source_rejected(tmp_path, fault):
    path = tmp_path / 'weights.safetensors'
    save_file({'weight': torch.ones(3)}, path)
    if fault == 'pointer':
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 999\n')
    else:
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(Exception):
        converter.load_complete(path)


def test_required_inventory():
    expected = {'weight': torch.empty(2, 3)}
    for invalid in ({}, {'other': torch.ones(2, 3)}, {'weight': torch.ones(2, 4)}):
        with pytest.raises(ValueError):
            converter.validate_inventory(invalid, expected, 'test')


@pytest.mark.parametrize('seconds', [.02, .021, 1.001, 5.0])
def test_latent_frame_rounding(seconds):
    latent = EmptyAuKLatentAudio.execute(seconds, 2).result[0]
    assert latent['samples'].shape == (2, 64, math.ceil(seconds * 50))
    assert latent['sample_rate'] == 24000


def test_euler_velocity_convention():
    class Sampling(ModelSamplingDiscreteFlow, CONST):
        pass
    class Config:
        sampling_settings = {'shift': 1., 'multiplier': 1.}
    sampling = Sampling(Config())
    sigmas = torch.tensor([1., .9238795042037964, .7071067690849304, .3826833963394165, 0.])
    native = torch.randn(1, 64, 5)
    upstream = native.clone()
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        t = sampling.timestep(sigma)
        torch.testing.assert_close(t, sigma)
        velocity = upstream * .03 + (1 - sigma)
        denoised = sampling.calculate_denoised(sigma.reshape(1), -velocity, native)
        native += (native - denoised) / sigma * (next_sigma - sigma)
        upstream += velocity * (sigma - next_sigma)
        torch.testing.assert_close(native, upstream)


def test_audio_decode_amplitude_compatibility():
    class VAE:
        audio_sample_rate = 24000
        def decode(self, latent):
            return torch.tensor([[[-.8], [.8]]])
    samples = {'samples': torch.zeros(1, 64, 1)}
    raw = vae_decode_audio(VAE(), samples, normalize=False)
    default = vae_decode_audio(VAE(), samples)
    assert raw['waveform'].abs().max() == .8
    assert default['waveform'].abs().max() < raw['waveform'].abs().max()
    assert raw['sample_rate'] == 24000


def synthesis(text='Hello, world!'):
    return {'task_type': 'instruct_tts', 'language': 'en', 'text_language': 'en',
            'params_extracted': {'text': text, 'style_desc': 'a warm voice'}}


def test_helper_preserves_speech_and_duration_override():
    result = synthesis()
    result['timing_ratio'] = 'malformed but unused'
    instruction, seconds = auk_prompt.prepare(result, 'Say "Hello, world!" in a warm voice', explicit_duration=2.31)
    assert '"Hello, world!"' in instruction
    assert seconds == 2.31
    with pytest.raises(ValueError, match='changed'):
        auk_prompt.prepare(synthesis('Hello world.'), 'Say "Hello, world!"')


def test_helper_reference_duration_rules():
    result = {'task_type':'speed_edit', 'language':'en', 'params_extracted':{'speed_multiplier':2}}
    assert auk_prompt.prepare(result, 'Twice as fast', 10)[1] == 5
    with pytest.raises(ValueError, match='reference audio'):
        auk_prompt.prepare(result, 'Twice as fast')
    clone = {'task_type':'zero_shot_tts', 'text_language':'en', 'params_extracted':{'text':'Hello there, world.'}}
    assert auk_prompt.prepare(clone, 'Hello there, world.', 3, 'Hello there, world.')[1] == pytest.approx(3, abs=.011)


@pytest.mark.parametrize('text', ['not json', '[]', '{"task_type":"unknown","params_extracted":{}}', '{"task_type":[],"params_extracted":{}}'])
def test_helper_malformed_output(text):
    with pytest.raises(ValueError, match='AuK Prompt Helper'):
        auk_prompt.parse_preparation(text)


def test_helper_native_generation_and_transcript_preserved():
    class Clip:
        def tokenize(self, prompt, **kwargs):
            payload = json.loads(prompt)
            assert payload['reference_transcript'] == '  Exact transcript.\n'
            assert kwargs['audio']['sample_rate'] == 16000
            return prompt
        def generate(self, tokens, **kwargs):
            assert not kwargs['do_sample']
            return [1]
        def decode(self, tokens):
            return json.dumps(synthesis())
    instruction, duration = AuKPromptHelper.execute(Clip(), 'Hello, world!', 512,
        reference_audio={'waveform':torch.zeros(1,1,16000),'sample_rate':16000},
        reference_transcript='  Exact transcript.\n', duration=7).result
    assert duration == 7 and 'Hello, world!' in instruction


def test_detector_requires_complete_auk_signature():
    signature = {'audio_embed.linear.weight': 1536, 'txt_proj.weight': 1536,
                 'transformer_blocks.9.attn.to_qkv_c.weight': 4608,
                 'single_transformer_blocks.19.attn.to_qkv.weight': 4608, 'proj_out.weight': 64}
    weights = {key: torch.empty(size, 1, device='meta') for key, size in signature.items()}
    weights['txt_norm.weight'] = torch.empty(1536, device='meta')
    assert model_detection.detect_unet_config(weights, '') == {'audio_model': 'auk_flash'}
    for missing in signature:
        assert model_detection.detect_unet_config({k:v for k,v in weights.items() if k != missing}, '') is None
    qwen = {'txt_norm.weight': torch.empty(3072, device='meta'),
            'img_in.weight': torch.empty(3072, 64, device='meta')}
    assert model_detection.detect_unet_config(qwen, '')['image_model'] == 'qwen_image'


def test_helper_malformed_numeric_timing():
    result = synthesis()
    result['timing_ratio'] = {'bad': 'number'}
    with pytest.raises(ValueError, match='AuK Prompt Helper.*timing ratio'):
        auk_prompt.prepare(result, 'Hello, world!')


@pytest.mark.parametrize('field,value', [('operation_subtype', []), ('style_desc', []), ('style_desc', {})])
def test_helper_malformed_parameter_types(field, value):
    result = synthesis()
    if field == 'operation_subtype':
        result[field] = value
    else:
        result['params_extracted'][field] = value
    with pytest.raises(ValueError, match='AuK Prompt Helper'):
        auk_prompt.prepare(result, 'Hello, world!')
