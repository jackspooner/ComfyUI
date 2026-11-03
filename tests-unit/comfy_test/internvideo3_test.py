"""CPU cache/layout checks, including opt-in comparison with pinned upstream code."""
import ast
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import comfy.ops
from comfy.video_caption import namespace
from comfy.video_caption_internvideo3 import InternVideoDecoder, LatentAttention, apply_interleaved_rope, internvideo_prompt, rotary_frequencies


def config():
    return namespace(dict(hidden_size=16, intermediate_size=24, vocab_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4, attention_bias=True, attention_dropout=0,
        q_lora_rank=None, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4, qk_head_dim=8,
        kv_lora_rank_list=[6, 6], rope_theta=5000000, rope_scaling={'mrope_section': [0, 1, 1]},
        rms_norm_eps=1e-6, hidden_act='silu', _attn_implementation='eager'))


def initialize(module):
    for name, parameter in module.named_parameters():
        nn.init.ones_(parameter) if 'norm' in name else nn.init.normal_(parameter, std=0.1)
    return module


@torch.inference_mode()
def test_latent_decode_matches_full_and_chunked_prefill_without_expanded_cache():
    torch.manual_seed(12)
    model = initialize(InternVideoDecoder(config(), torch.float32, 'cpu', comfy.ops.manual_cast))
    hidden = torch.randn(1, 7, 16)
    positions = torch.arange(7)[None, None].expand(3, 1, -1).clone()
    positions[1] *= 2
    positions[2] *= 3
    full, _ = model.step(hidden.clone(), positions, None)
    first, cache = model.step(hidden[:, :3].clone(), positions[:, :, :3], None)
    second, cache = model.step(hidden[:, 3:6].clone(), positions[:, :, 3:6], cache)
    last, cache = model.step(hidden[:, 6:].clone(), positions[:, :, 6:], cache)
    torch.testing.assert_close(torch.cat((first, second, last), dim=1), full, atol=3e-6, rtol=3e-6)
    assert cache[0][0].shape == (1, 7, 6)
    assert cache[0][1].shape == (1, 1, 7, 4)


def test_interleaved_rotary_matches_transformers_deepseek_reference():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
    positions = torch.arange(5)[None, None].expand(3, 1, -1)
    cos, sin = rotary_frequencies(positions, 128, 5000000, [24, 20, 20])
    q, k = torch.randn(1, 2, 5, 128), torch.randn(1, 1, 5, 128)
    expected_q, expected_k = apply_rotary_pos_emb_interleave(q, k, torch.cat((cos, cos), -1), torch.cat((sin, sin), -1))
    torch.testing.assert_close(apply_interleaved_rope(q, cos, sin), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(apply_interleaved_rope(k, cos, sin), expected_k, rtol=0, atol=0)


def test_prompt_preserves_pinned_template_without_audio_or_default_system():
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|video_pad|>', '<|audio_pad|>'])
    ids = internvideo_prompt(tokenizer, 'Describe.', torch.tensor([[2, 4, 6]]), [0.25, 1.25])
    assert ids[0, 0] == tokenizer.token_to_id('<|im_start|>')
    assert (ids == tokenizer.token_to_id('<|video_pad|>')).sum() == 12
    assert (ids == tokenizer.token_to_id('<|vision_start|>')).sum() == 2
    assert not (ids == tokenizer.token_to_id('<|audio_pad|>')).any()


@torch.inference_mode()
def test_attention_matches_pinned_upstream_prefill_and_absorbed_decode():
    source = os.environ.get('INTERNVIDEO3_REFERENCE')
    if not source:
        pytest.skip('Set INTERNVIDEO3_REFERENCE to the pinned local modeling_internvideo3.py')
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
    # Compile only the inspected attention class, without importing upstream
    # module registration, network hooks, dependencies or model loaders.
    tree = ast.parse(Path(source).read_text())
    selected = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'InternVideo3TextAttentionMLA')
    def reference_attention(module, q, k, v, mask, scaling, **kwargs):
        scores = q @ k.transpose(-1, -2) * scaling
        if mask is not None:
            scores = scores + mask
        output = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ v
        return output.transpose(1, 2), None
    env = dict(torch=torch, nn=nn, F=F, apply_rotary_pos_emb_interleave=apply_rotary_pos_emb_interleave,
        eager_attention_forward=reference_attention, deprecate_kwarg=lambda *args, **kwargs: lambda function: function)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), source, 'exec'), env)
    c = config()
    reference = env['InternVideo3TextAttentionMLA'](c, 0).eval()
    native = LatentAttention(c, 0, torch.float32, 'cpu', comfy.ops.manual_cast)
    native.load_state_dict(reference.state_dict(), strict=True)
    class Cache:
        def __init__(self):
            self.rotary = self.latent = None
        def update(self, rotary, latent, *args):
            self.rotary = rotary if self.rotary is None else torch.cat((self.rotary, rotary), dim=2)
            self.latent = latent if self.latent is None else torch.cat((self.latent, latent), dim=2)
            return self.rotary, self.latent
    upstream_cache, native_cache = Cache(), None
    for start, count in [(0, 4), (4, 1), (5, 1)]:
        hidden = torch.randn(1, count, 16)
        positions = torch.arange(start, start + count)[None, None].expand(3, 1, -1)
        cos, sin = rotary_frequencies(positions, 4, c.rope_theta, c.rope_scaling.mrope_section)
        mask = torch.triu(torch.full((1, 1, count, count), float('-inf')), 1) if count > 1 else None
        expected, _ = reference(hidden, (torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)), mask, past_key_values=upstream_cache)
        actual, native_cache = native(hidden, (cos, sin), native_cache, mask)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@torch.inference_mode()
def test_native_vision_matches_pinned_upstream_on_temporal_nonsquare_grid():
    source = os.environ.get('INTERNVIDEO3_REFERENCE')
    if not source:
        pytest.skip('Set INTERNVIDEO3_REFERENCE to the pinned local modeling_internvideo3.py')
    from transformers.activations import ACT2FN
    from comfy.text_encoders.qwen35 import Qwen35VisionModel
    names = {'InternVideo3VisionMLP', 'InternVideo3VisionPatchEmbed', 'InternVideo3VisionRotaryEmbedding',
        'InternVideo3VisionPatchMerger', 'rotate_half', 'apply_rotary_pos_emb_vision', 'repeat_kv',
        'eager_attention_forward', 'InternVideo3VisionAttention', 'InternVideo3VisionBlock', 'InternVideo3VisionModel'}
    tree = ast.parse(Path(source).read_text())
    selected = [item for item in tree.body if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in names]
    class ReferenceBase(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
    env = dict(torch=torch, nn=nn, F=F, ACT2FN=ACT2FN, GradientCheckpointingLayer=nn.Module, InternVideo3PreTrainedModel=ReferenceBase)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), source, 'exec'), env)
    c = dict(hidden_size=32, intermediate_size=48, depth=2, num_heads=4, patch_size=2,
        temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=16,
        out_hidden_size=16, deepstack_visual_indexes=[], hidden_act='gelu_pytorch_tanh', _attn_implementation='eager')
    reference = env['InternVideo3VisionModel'](namespace(c)).eval()
    native = Qwen35VisionModel(c, dtype=torch.float32, device='cpu', ops=comfy.ops.manual_cast)
    native.load_state_dict(reference.state_dict(), strict=True)
    grid = torch.tensor([[2, 4, 6]])
    patches = torch.randn(48, 24)
    expected, deepstack = reference(patches, grid)
    assert not deepstack
    torch.testing.assert_close(native(patches, grid), expected, atol=2e-6, rtol=2e-6)
