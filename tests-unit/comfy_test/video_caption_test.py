"""CPU component and preprocessing checks; full checkpoint/queue acceptance is separate."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import comfy.ops
from comfy.video_caption import CaptionDecoder
from comfy.video_caption_audio import QueryBlock
from comfy.video_caption_grounding import PadGrounding, SalmonnPositions
from comfy.video_caption_media import prepare_video, media_prompt, prepare_audio
from comfy.text_encoders.llama import Llama2Config
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel


def test_qformer_matches_independent_query_attention_math():
    torch.manual_seed(17)
    config = dict(hidden_size=16, encoder_width=8, num_attention_heads=4, intermediate_size=24, layer_norm_eps=1e-12)
    block = QueryBlock(config, torch.float32, 'cpu', torch.nn)
    hidden, source = torch.randn(3, 1, 16), torch.randn(3, 5, 8)
    expected = hidden
    for attention, inputs in [(block.attention, hidden), (block.crossattention, source)]:
        q = attention.self.query(expected).reshape(3, 1, 4, 4).transpose(1, 2)
        k = attention.self.key(inputs).reshape(3, -1, 4, 4).transpose(1, 2)
        v = attention.self.value(inputs).reshape(3, -1, 4, 4).transpose(1, 2)
        attended = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(3, 1, 16)
        expected = attention.output.LayerNorm(attention.output.dense(attended) + expected)
    expected = block.output_query.LayerNorm(block.output_query.dense(F.gelu(block.intermediate_query.dense(expected))) + expected)
    torch.testing.assert_close(block(hidden, source), expected, atol=2e-6, rtol=2e-6)


def test_decoder_cached_steps_match_full_causal_pass():
    with torch.set_grad_enabled(False):
        torch.manual_seed(3)
        config = Llama2Config(vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                              num_attention_heads=2, num_key_value_heads=1)
        config.head_dim = 8
        decoder = CaptionDecoder(config, device='cpu', dtype=torch.float32, ops=torch.nn)
        for name, parameter in decoder.named_parameters():
            if 'norm' in name:
                torch.nn.init.ones_(parameter)
        hidden = torch.randn(1, 5, 16)
        full, _, _ = decoder.step(hidden.clone(), torch.arange(5)[None], None)
        cache, parts = None, []
        for index in range(5):
            output, cache, _ = decoder.step(hidden[:, index:index+1].clone(), torch.tensor([[index]]), cache)
            parts.append(output)
        torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=2e-6, rtol=2e-6)
        assert cache[0][2] == 5


def test_pad_branch_masks_isolate_histories_and_restrict_intervals():
    owner = PadGrounding()
    owner._branch_prefix_visible = torch.tensor([[True, False], [False, True]])
    mask = owner._branch_visibility_mask(2, 8, 2, torch.float32, 'cpu')[0, 0]
    assert torch.equal(mask == 0, torch.tensor([[True, False, True, False, True, False, True, False],
                                               [False, True, False, True, False, True, False, True]]))


def test_multimodal_decoder_positions_use_native_single_batch_layout():
    torch.manual_seed(9)
    config = Llama2Config(vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1)
    config.head_dim, config.rope_dims, config.interleaved_mrope = 8, [2, 1, 1], True
    decoder = CaptionDecoder(config, device='cpu', dtype=torch.float32, ops=torch.nn)
    for name, parameter in decoder.named_parameters():
        if 'norm' in name:
            torch.nn.init.ones_(parameter)
    positions = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]])
    hidden = torch.randn(1, 4, 16)
    with torch.inference_mode():
        upstream, _, _ = decoder.step(hidden.clone(), positions[:, None], None)
        native, _, _ = decoder.step(hidden.clone(), positions, None)
    torch.testing.assert_close(upstream, native, atol=0, rtol=0)


def test_temporal_patching_preserves_both_frames_and_last_odd_frame():
    frames = torch.stack([torch.zeros(28, 28, 3), torch.ones(28, 28, 3), torch.full((28, 28, 3), 0.5)])
    patches, grid, timestamps = prepare_video(frames, 2, 2, 14, 784)
    assert grid.tolist() == [[2, 2, 2]]
    assert timestamps == [0.25, 1.0]
    unpacked = patches.reshape(2, 1, 1, 2, 2, 3, 2, 14, 14)
    assert not torch.equal(unpacked[0, ..., 0, :, :], unpacked[0, ..., 1, :, :])
    torch.testing.assert_close(unpacked[1, ..., 0, :, :], unpacked[1, ..., 1, :, :])


@pytest.mark.parametrize('patch', [14, 16])
def test_temporal_preprocessing_matches_official_processors(patch):
    qwen2 = pytest.importorskip('transformers.models.qwen2_vl.video_processing_qwen2_vl')
    qwen3 = pytest.importorskip('transformers.models.qwen3_vl.video_processing_qwen3_vl')
    frames = torch.rand(4, 56, 84, 3)
    cls = qwen2.Qwen2VLVideoProcessor if patch == 14 else qwen3.Qwen3VLVideoProcessor
    processor = cls(patch_size=patch, temporal_patch_size=2, do_sample_frames=False, do_rescale=False,
                    size={'shortest_edge': 784, 'longest_edge': 61250})
    expected = processor(videos=[frames.permute(0, 3, 1, 2)], return_tensors='pt')
    actual, grid, _ = prepare_video(frames, 2, 2, patch, 61250)
    torch.testing.assert_close(grid, expected.video_grid_thw, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected.pixel_values_videos, rtol=0, atol=0)


def test_audio_padding_and_chunk_count():
    audio = dict(waveform=torch.zeros(1, 2, 8000), sample_rate=16000)
    assert prepare_audio(audio).shape == (1, 128, 3000)
    assert prepare_audio(None) is None


def test_embedded_tokenizer_prompt_keeps_audio_and_video_counts():
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.add_special_tokens([AddedToken(value, special=True) for value in
        ['<|audio_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>', '<|im_start|>', '<|im_end|>']])
    ids, counts = media_prompt(tokenizer, 'padcaptioner_3b', 'Describe.', torch.tensor([[3, 2, 2]]), [0.25, 1.25, 2.25], 60, 2)
    assert counts == [2, 2, 56]
    assert (ids == tokenizer.token_to_id('<|video_pad|>')).sum() == 3
    assert (ids == tokenizer.token_to_id('<|audio_pad|>')).sum() == 60


def test_grounding_keeps_largest_mass_and_snaps_only_stronger_edges():
    owner = PadGrounding()
    owner.config = SimpleNamespace(grounding_threshold=0.7, grounding_merge_gap=0, grounding_snap_window=1)
    assert owner.find_and_merge(torch.tensor([[0.8, 0.8, 0, 0.9]])) == [[0, 1]]
    assert owner._snap_readout([[1, 2]], torch.tensor([2., 1., 0., 0.]), torch.tensor([0., 0., 1., 2.])) == [[0, 3]]
