import numpy as np
import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy import ops
from comfy.ldm.rvc.model import RelativeAttention, Synthesizer
from comfy.rvc.checkpoint import fold_weights, voice_config
from comfy.rvc.runtime import coarse_f0, retrieve


@pytest.mark.parametrize("length", [1, 7, 35])
def test_relative_keys_and_values_match_direct_offsets(length):
    torch.manual_seed(4)
    module = RelativeAttention(12, 3, ops.manual_cast)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    x = torch.randn(1, 12, length)
    q, k, v = [p(x).reshape(1, 3, 4, length).transpose(2, 3) for p in (module.conv_q, module.conv_k, module.conv_v)]
    offsets = torch.arange(length)[None, :] - torch.arange(length)[:, None]
    valid = offsets.abs() <= 10
    rel_k = module.emb_rel_k[0][(offsets + 10).clamp(0, 20)] * valid[..., None]
    rel_v = module.emb_rel_v[0][(offsets + 10).clamp(0, 20)] * valid[..., None]
    scores = (q @ k.transpose(-1, -2) + torch.einsum("bhid,ijd->bhij", q, rel_k)) / 2
    weights = scores.softmax(-1)
    expected = weights @ v + torch.einsum("bhij,ijd->bhid", weights, rel_v)
    expected = module.conv_o(expected.transpose(2, 3).reshape(1, 12, length))
    torch.testing.assert_close(module(x), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("modern", [False, True])
def test_fold_half_weights_matches_upstream_float_inference(modern):
    v = torch.randn(7, 4, 3).half()
    g = torch.randn(7, 1, 1).half()
    prefix = "layer.parametrizations.weight.original" if modern else "layer.weight_"
    keys = [prefix + x for x in (("0", "1") if modern else ("g", "v"))]
    state, _ = fold_weights({keys[0]: g, keys[1]: v})
    assert state["layer.weight"].dtype == torch.float32
    torch.testing.assert_close(state["layer.weight"], torch._weight_norm(v.float(), g.float(), 0), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incomplete"):
        fold_weights({keys[0]: g})


def test_pitch_bins_and_octave_shift():
    f0 = torch.tensor([[0., 50., 220., 440., 1100.]])
    bins = coarse_f0(f0)
    assert bins.tolist()[0][0:2] == [1, 1]
    assert bins[0, -1] == 255
    assert bins[0, 2] < bins[0, 3]
    assert torch.equal(coarse_f0(f0 * 2)[:, 2:3], bins[:, 3:4])


def test_retrieval_exact_neighbors_stays_finite():
    class Index:
        def search(self, features, count):
            return np.zeros((len(features), count), dtype=np.float32), np.tile(np.arange(count), (len(features), 1))
    features = torch.randn(1, 3, 4)
    vectors = np.arange(32, dtype=np.float32).reshape(8, 4)
    actual = retrieve(features, Index(), vectors, .75)
    expected = features * .25 + torch.from_numpy(vectors.mean(0))[None, None] * .75
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("version,use_f0", [("v1", True), ("v2", True), ("v2", False)])
def test_inference_only_export_shapes(version, use_f0):
    config = [1025, 32, 8, 8, 16, 2, 1, 3, 0, "1", [3], [[1, 3, 5]], [10, 8, 2, 2], 32, [16, 16, 4, 4], 1, 8, 32000]
    model = Synthesizer(config, version, use_f0, ops.manual_cast)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.02)
    features = torch.randn(1, 12, 256 if version == "v1" else 768)
    pitch, f0 = torch.full((1, 12), 100), torch.full((1, 12), 220.)
    result = model(features, pitch if use_f0 else None, f0 if use_f0 else None, torch.tensor([0]), torch.Generator().manual_seed(8), {})
    assert result.shape == (1, 1, 3840)
    assert torch.isfinite(result).all()
    assert "enc_q" not in model._modules


def test_export_rejects_training_and_other_vocoders():
    with pytest.raises(ValueError, match="inference export"):
        voice_config({"model": {}})
    checkpoint = {"weight": {}, "config": [0] * 18, "version": "v2", "vocoder": "RefineGAN"}
    with pytest.raises(ValueError, match="HiFi-GAN"):
        voice_config(checkpoint)
