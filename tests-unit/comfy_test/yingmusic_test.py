from functools import partial
import json
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from comfy.cli_args import args
args.cpu = True
from comfy import model_detection, ops, sd
from comfy.audio_encoders import audio_encoders
from comfy.audio_encoders.yingmusic import prepare_melody
from comfy.ldm.yingmusic.model import Attention, GRN, YingMusic
from comfy_extras.nodes_multigpu import SelectVAEDeviceNode
from comfy_extras.nodes_yingmusic import YingMusicPackageLoader, prepare_audio


def initialize(model):
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    return model


@pytest.mark.parametrize("length", [1, 7, 29])
def test_interleaved_rotary_attention(length):
    torch.manual_seed(17)
    attention = initialize(Attention(64, 2, torch.float32, "cpu", ops.manual_cast))
    x = torch.randn(1, length, 64)
    frequencies = torch.outer(torch.arange(length), 10000 ** (-torch.arange(0, 32, 2) / 32))
    cos, sin = frequencies.cos(), frequencies.sin()
    matrix = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, length, 16, 2, 2)
    def rotate(value):
        value = value.reshape(1, length, 2, 32).transpose(1, 2)
        pairs = value.reshape(1, 2, length, 16, 2)
        first, second = pairs.unbind(-1)
        return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)
    q, k = rotate(attention.to_q(x)), rotate(attention.to_k(x))
    v = attention.to_v(x).reshape(1, length, 2, 32).transpose(1, 2)
    reference = attention.to_out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, length, 64))
    torch.testing.assert_close(attention(x, matrix), reference, atol=2e-6, rtol=2e-5)


def test_melody_probabilities_dropout_and_reference_mask():
    logits = torch.zeros(1, 30, 128)
    actual = prepare_melody(logits, 30, 4)
    assert torch.count_nonzero(actual[:, :4]) == 0
    assert torch.count_nonzero(actual[:, ::10]) == 0
    assert torch.all(actual[:, 4:10] == .5)
    assert torch.count_nonzero(logits) == 0


def test_null_conditioning_drops_midi_bias_after_projection():
    torch.manual_seed(4)
    model = initialize(YingMusic(dim=64, depth=1, heads=2, text_dim=32, text_depth=1,
                                 dtype=torch.float32, device="cpu", operations=ops.manual_cast))
    tokens, midi, reference = torch.randint(0, 373, (1, 5)), torch.randn(1, 5, 128), torch.randn(1, 64, 5)
    context = model.preprocess_conditioning(tokens, midi, reference, unconditioned=True)
    assert torch.count_nonzero(context[..., :64]) == 0
    assert torch.count_nonzero(context[..., -128:]) == 0
    assert torch.count_nonzero(context[..., 64:-128]) > 0


def test_grn_preserves_checkpoint_affine_terms():
    model = initialize(GRN(8, torch.float32, "cpu"))
    x = torch.randn(2, 5, 8)
    gx = torch.norm(x, p=2, dim=1, keepdim=True)
    expected = model.gamma * (x * gx / (gx.mean(-1, keepdim=True) + 1e-6)) + model.beta + x
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("shift", [.25, .5, 1, 2])
def test_flow_time_reversal(shift):
    time = torch.linspace(0, 1, 9)
    time = shift * time / (1 + (shift - 1) * time)
    sigma = 1 - time
    initial = torch.randn(1, 64, 7)
    velocity = torch.randn_like(initial)
    upstream, native = initial.clone(), initial.clone()
    for step in range(8):
        upstream += (time[step + 1] - time[step]) * velocity
        native += (sigma[step + 1] - sigma[step]) * -velocity
    torch.testing.assert_close(native, upstream)


def test_detection_requires_complete_specific_signature():
    shapes = {"text_embed_p.text_embed.weight": (374, 512), "input_embed_with_midi.midi_proj.weight": (128, 128),
              "input_embed_with_midi.proj.weight": (1024, 768), "transformer_blocks.21.attn_norm.linear.weight": (6144, 1024),
              "time_embed.time_mlp.0.weight": (1024, 256), "proj_out.weight": (64, 1024)}
    sd = {key: torch.empty(shape, device="meta") for key, shape in shapes.items()}
    assert model_detection.detect_unet_config(sd, "")["audio_model"] == "yingmusic"
    for key in sd:
        assert model_detection.detect_unet_config({name: value for name, value in sd.items() if name != key}, "") is None


def test_reference_channel_and_silence_preparation():
    source = {"waveform": torch.ones(1, 1, 4410), "sample_rate": 44100}
    result = prepare_audio(source, .5)
    assert result.shape == (1, 2, 26460)
    assert torch.all(result[..., :4410] == 1)
    assert torch.count_nonzero(result[..., 4410:]) == 0
    assert source["waveform"].shape == (1, 1, 4410)


@pytest.fixture
def singer_package(tmp_path, monkeypatch):
    tiny_vae = partial(sd.AudioOobleckVAE, channels=2, c_mults=[1], strides=[2])
    monkeypatch.setattr(sd, 'AudioOobleckVAE', tiny_vae)
    state = initialize(tiny_vae()).state_dict()
    identity = {
        'source_sha256': 'b32dc225a6cfff37f2252b7d3b60cbf536f656de349eba6d5b773aa5f5b780e4',
        'implementation_revision': 'baa409c2e7e5e775f09b4e92a220808f4827d2cc',
    }
    manifest = {**identity, 'files': {}}
    for kind in ('diffusion_models', 'vae', 'audio_encoders'):
        path = tmp_path/kind/'yingmusic_singer_plus.safetensors'
        path.parent.mkdir()
        tensors = state if kind == 'vae' else {'weight': torch.ones(1)}
        save_file(tensors, str(path), metadata={**identity, 'source': 'ASLP-lab/YingMusic-Singer-Plus'})
        manifest['files'][str(path.relative_to(tmp_path))] = {'tensors': len(tensors)}
    (tmp_path/'yingmusic_singer_plus.manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(sd, 'load_diffusion_model', Mock(return_value=object()))
    monkeypatch.setattr(audio_encoders, 'load_audio_encoder_from_sd', Mock(return_value=object()))
    monkeypatch.setattr(sd.model_management, 'get_torch_device', lambda: torch.device('cuda:0'))
    monkeypatch.setattr(sd.model_management, 'vae_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(sd.model_management, 'vae_dtype', lambda *args: torch.float32)
    monkeypatch.setattr(sd.model_management, 'unload_model_and_clones', Mock())
    return tmp_path, state


@pytest.mark.parametrize('device', ['auto', 'cpu', 'cuda:0'])
def test_package_vae_device_selection_reconstructs_pristine_weights(singer_package, monkeypatch, device):
    root, state = singer_package
    model, vae, encoder, lyrics = YingMusicPackageLoader().load(str(root), device)
    assert model is sd.load_diffusion_model.return_value
    assert encoder is audio_encoders.load_audio_encoder_from_sd.return_value
    assert lyrics == str(root/'text_encoders/yingmusic_singer_plus')
    selected = None if device == 'auto' else torch.device(device)
    original_device = selected or torch.device('cpu')
    assert vae.device == vae.patcher.load_device == original_device
    assert not vae.crop_input  # Converted YingMusic metadata disables generic VAE cropping.
    assert all(value.device.type == 'cpu' for value in vae.first_stage_model.parameters())
    original = vae.first_stage_model
    key = next(iter(state))
    original.state_dict()[key].add_(7)
    target = torch.device('cuda:1' if device == 'cuda:0' else 'cuda:0')
    monkeypatch.setattr(sd.model_management, 'resolve_gpu_device_option', lambda value: target if value == 'gpu:fixture' else None)
    retargeted = SelectVAEDeviceNode.execute(vae, 'gpu:fixture').result[0]
    assert retargeted.device == retargeted.patcher.load_device == target
    assert retargeted.first_stage_model is retargeted.patcher.model
    assert retargeted.first_stage_model is not original
    for name, value in retargeted.first_stage_model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        assert value.device.type == 'cpu'
    restored = SelectVAEDeviceNode.execute(retargeted, 'default').result[0]
    assert restored.device == restored.patcher.load_device == original_device
    assert restored.first_stage_model is not retargeted.first_stage_model
    assert restored.first_stage_model is restored.patcher.model
    assert vae.device == original_device
    assert not restored.crop_input
    assert sd.model_management.unload_model_and_clones.call_count == 2


def test_package_loader_rejects_invalid_vae_before_encoder(singer_package):
    root, state = singer_package
    path = root/'vae/yingmusic_singer_plus.safetensors'
    save_file({f'unknown.{index}': value for index, value in enumerate(state.values())}, str(path),
              metadata={'source_sha256': 'b32dc225a6cfff37f2252b7d3b60cbf536f656de349eba6d5b773aa5f5b780e4'})
    with pytest.raises(RuntimeError, match='VAE is invalid'):
        YingMusicPackageLoader().load(str(root), 'cpu')
    audio_encoders.load_audio_encoder_from_sd.assert_not_called()
