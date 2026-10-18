import json

import pytest
import torch
from torch.nn import functional as F

from comfy.cli_args import args
args.cpu = True
from comfy import ops
from comfy.ldm.soulx_singer.model import AdaptiveRMSNorm, Attention, Singer, coarse_f0
from comfy.ldm.soulx_singer.vocoder import Vocos
from comfy.soulx_singer.metadata import align_lyrics, metadata_from_rows, preprocess, read_metadata, segment_metadata
from comfy.soulx_singer.runtime import fit_f0, sample_flow, whisper_features
from comfy.text_encoders.llama import precompute_freqs_cis


def initialize(model):
    torch.manual_seed(23)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.05)
    return model


@pytest.mark.parametrize("length", [1, 7, 83])
def test_split_half_noncausal_attention(length):
    attn = initialize(Attention(64, 2, torch.float32, "cpu", ops.manual_cast))
    x = torch.randn(1, length, 64)
    cos, sin, _ = precompute_freqs_cis(32, torch.arange(length)[None], 10000., device="cpu")
    half_cos = cos[..., :16]
    matrix = torch.stack((half_cos, -sin, sin, half_cos), dim=-1).unflatten(-1, (2, 2))
    def rotated(projection):
        a, b = projection(x).reshape(1, length, 2, 32).transpose(1, 2).chunk(2, dim=-1)
        return torch.cat((a * half_cos - b * sin, b * half_cos + a * sin), dim=-1)
    q, k = rotated(attn.q_proj), rotated(attn.k_proj)
    v = attn.v_proj(x).reshape(1, length, 2, 32).transpose(1, 2)
    expected = attn.o_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, length, 64))
    torch.testing.assert_close(attn(x, matrix, {}), expected, atol=2e-6, rtol=2e-5)


def test_adaptive_norm_matches_released_variance_math():
    norm = initialize(AdaptiveRMSNorm(64, torch.float32, "cpu", ops.manual_cast))
    x, time = torch.randn(1, 13, 64), torch.randn(1, 64)
    expected = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * norm.to_weight(time)[:, None]
    torch.testing.assert_close(norm(x, time), expected, atol=2e-6, rtol=2e-5)


def test_pitch_bins_voicing_and_shift():
    f0 = torch.tensor([[0., 32.7031956625, 440., 880., 1e5]])
    expected = torch.tensor([[0, 1, 226, 286, 360]])
    assert torch.equal(coarse_f0(f0), expected)
    assert torch.equal(coarse_f0(f0, -12), torch.tensor([[0, 1, 166, 226, 300]]))


def test_preprocessor_matches_released_index_allocation():
    phones = ["<PAD>", "<SP>", "<BOW>", "<EOW>", "<SEP>", "en_L", "en_IH1", "en_V"]
    meta = {"phoneme": "<SP> en_L-IH1-V <SP> <SP>", "duration": ".1 .5 .1 .1", "note_pitch": "0 65 0 0", "note_type": "1 2 1 1"}
    actual = preprocess(meta, phones)
    assert len(actual["mel2note"]) == 40
    assert len(actual["phoneme"]) == 13
    assert actual["mel2note"][0] == 1
    assert actual["mel2note"][5] == 4
    assert actual["mel2note"][29] == 9
    assert actual["mel2note"][-1] == 12


def test_replacement_uses_syllable_count_and_keeps_f0():
    f0 = [220.] * 100
    pitch = {"f0": f0, "duration": 2.}
    result = align_lyrics(pitch, [{"start": 0, "end": 2, "text": "old"}], "hello world",
                          lambda w: {"hello": ["HH", "AH0", "L", "OW1"], "world": ["W", "ER1", "L", "D"]}[w])
    assert result["f0"] == f0
    assert sum(result["duration"]) == pytest.approx(2.)
    assert len([r for r in result["rows"] if r.get("text")]) == 3
    assert len({(r["word"], r["syllable"]) for r in result["rows"]}) == 3
    assert result["alignment_status"].startswith("estimated")


def test_invalid_and_overlapping_metadata_fails_at_boundary():
    with pytest.raises(ValueError, match="overlap"):
        metadata_from_rows([{"start": 0, "end": 1, "phoneme": "<SP>", "note_pitch": 0},
                            {"start": .5, "end": 2, "phoneme": "<SP>", "note_pitch": 0}], [0.] * 100, 2.)
    with pytest.raises(ValueError, match="phrases"):
        align_lyrics({"f0": [220.] * 100}, [{"start": 0, "end": 2}], "one|two", lambda w: ["AA1"])


def test_sampler_matches_midpoint_cfg_rescale_and_prompt_removal():
    model = initialize(Singer(dim=32, depth=1, heads=2, encoder_dim=16, encoder_depth=1,
                             dtype=torch.float32, device="cpu", operations=ops.manual_cast))
    cond, prompt = torch.randn(1, 9, 16), torch.randn(1, 3, 128)
    seed, steps, cfg = 5, 3, 2.
    x = torch.randn(1, 6, 128, generator=torch.Generator().manual_seed(seed))
    conditioning = model.prepare_conditioning(cond)
    unconditional = model.diff_estimator.prepare_conditioning(torch.zeros(1, 6, 32))
    for i in range(steps):
        t = torch.tensor([(i + .5) / steps])
        positive = model(torch.cat((prompt, x), 1), t, conditioning)[:, 3:]
        negative = model(x, t, unconditional)
        guided = positive + cfg * (positive - negative)
        x = x + (.75 * guided * positive.std() / guided.std() + .25 * guided) / steps
    actual = sample_flow(model, prompt, cond, 6, steps, cfg, seed)
    torch.testing.assert_close(actual, x, atol=2e-6, rtol=2e-5)


def test_vocos_same_padding_retains_exact_hop_count():
    model = initialize(Vocos(dim=16, intermediate=32, depth=2, n_fft=1920, hop=480,
                             dtype=torch.float32, device="cpu", operations=ops.manual_cast))
    model.head.istft.window.copy_(torch.hann_window(1920))
    result = model(torch.randn(1, 128, 5))
    assert result.shape == (1, 1, 2400)
    assert torch.isfinite(result).all()


def test_whisper_short_audio_pads_features_after_log_not_waveform():
    features, valid = whisper_features(torch.randn(1, 24000))
    assert features.shape == (1, 80, 3000)
    assert valid == 50
    assert torch.count_nonzero(features[..., 100:]) == 0


def test_short_contour_pads_unvoiced_tail_without_stretching():
    assert fit_f0([0, 220, 230], 5) == [0, 220, 230, 0, 0]
    assert fit_f0([0, 220, 230], 2) == [0, 220]


def test_vibrato_does_not_fragment_syllable_phonemes():
    f0 = [220. if i % 8 < 4 else 247. for i in range(100)]
    aligned = align_lyrics({"f0": f0, "duration": 2.}, [{"start": 0, "end": 2}], "love",
                           lambda word: ["L", "AH1", "V"])
    assert len(aligned["rows"]) == 1
    assert aligned["phoneme"] == ["en_L-AH1-V"]
    assert aligned["f0"] == f0


def test_segmented_metadata_keeps_global_placement_and_local_edit_times():
    data = metadata_from_rows([{"start": 1., "end": 61., "phoneme": "en_AH1", "note_pitch": 60}], [220.] * 3100, 62.)
    segments = segment_metadata(data)
    restored = read_metadata(json.dumps(segments))
    assert len(restored) == 3
    assert restored[-1]["time"][1] == 62000
    for item in restored:
        assert sum(item["duration"]) == pytest.approx((item["time"][1] - item["time"][0]) / 1000)
        assert item["rows"][0]["start"] == 0
