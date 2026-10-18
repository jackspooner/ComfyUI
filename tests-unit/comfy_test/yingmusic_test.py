import pytest
import torch
from torch.nn import functional as F

from comfy.cli_args import args
args.cpu = True
from comfy import model_detection, ops
from comfy.audio_encoders.yingmusic import prepare_melody
from comfy.ldm.yingmusic.model import Attention, GRN, YingMusic
from comfy_extras.nodes_yingmusic import prepare_audio


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
