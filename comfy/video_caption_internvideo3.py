# SPDX-License-Identifier: Apache-2.0
# Adapted from InternVideo3 by Ziang Yan et al., checkpoint/code c4602918.
"""Native InternVideo3 M²LA, with latent-only request-local KV caches."""
import json

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.storage
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import MLP
from comfy.text_encoders.qwen35 import Qwen35VisionModel
from comfy.video_caption import namespace
from comfy.video_caption_grounding import SalmonnPositions
from comfy.video_caption_media import prepare_video


def rotary_frequencies(positions, dimension, theta, sections):
    inv = theta ** (-torch.arange(0, dimension, 2, device=positions.device, dtype=torch.float32) / dimension)
    angles = positions.float()[..., None] * inv
    mixed = angles[0].clone()
    for axis in (1, 2):
        selection = slice(axis, sections[axis] * 3, 3)
        mixed[..., selection] = angles[axis, ..., selection]
    return mixed.cos(), mixed.sin()


def apply_interleaved_rope(value, cos, sin):
    # Upstream DeepSeek layout rotates even/odd input pairs into split halves.
    even, odd = value[..., 0::2], value[..., 1::2]
    cos, sin = cos[:, None].to(value), sin[:, None].to(value)
    return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)


class LatentAttention(nn.Module):
    def __init__(self, config, index, dtype, device, operations):
        super().__init__()
        self.heads = config.num_attention_heads
        self.rank = config.kv_lora_rank_list[index]
        self.rope_dim, self.nope_dim, self.value_dim = config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim
        self.qk_dim = self.rope_dim + self.nope_dim
        self.q_proj = operations.Linear(config.hidden_size, self.heads * self.qk_dim, bias=config.attention_bias, dtype=dtype, device=device)
        self.kv_a_proj_with_mqa = operations.Linear(config.hidden_size, self.rank + self.rope_dim, bias=config.attention_bias, dtype=dtype, device=device)
        self.kv_b_proj = operations.Linear(self.rank, self.heads * (self.nope_dim + self.value_dim), bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(self.heads * self.value_dim, config.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden, frequencies, cache, mask):
        batch, count, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, count, self.heads, self.qk_dim).transpose(1, 2)
        q_plain, q_rope = query.split((self.nope_dim, self.rope_dim), dim=-1)
        latent, k_rope = self.kv_a_proj_with_mqa(hidden).split((self.rank, self.rope_dim), dim=-1)
        q_rope = apply_interleaved_rope(q_rope, *frequencies)
        k_rope = apply_interleaved_rope(k_rope[:, None], *frequencies)
        if cache is not None:
            latent = torch.cat((cache[0], latent), dim=1)
            k_rope = torch.cat((cache[1], k_rope), dim=2)
        next_cache = (latent, k_rope)
        if count == 1:
            # Absorb the up-projection into the query and output, retaining only
            # rank + rotary dimensions per cached token, rather than expanded KV.
            with comfy.ops.CastBiasWeightContext(self.kv_b_proj, hidden, offloadable=True) as (weight, _):
                weight = weight.view(self.heads, self.nope_dim + self.value_dim, self.rank)
                absorbed = torch.einsum('bhsd,hdc->bhsc', q_plain, weight[:, :self.nope_dim])
                scores = (torch.einsum('bhsc,btc->bhst', absorbed, latent) + q_rope @ k_rope.transpose(-1, -2)) * self.qk_dim ** -0.5
                if mask is not None:
                    scores = scores + mask
                attended = torch.einsum('bhst,btc->bhsc', scores.softmax(-1), latent)
                attended = torch.einsum('bhsc,hdc->bhsd', attended, weight[:, self.nope_dim:])
        else:
            expanded = self.kv_b_proj(latent).view(batch, -1, self.heads, self.nope_dim + self.value_dim).transpose(1, 2)
            keys, values = expanded.split((self.nope_dim, self.value_dim), dim=-1)
            keys = torch.cat((keys, k_rope.expand(-1, self.heads, -1, -1)), dim=-1)
            query = torch.cat((q_plain, q_rope), dim=-1)
            # Equal head widths preserve native attention backends; zero padding
            # leaves the actual value result unchanged.
            values = F.pad(values, (0, self.qk_dim - self.value_dim))
            attention = optimized_attention_for_device(hidden.device, mask=mask is not None, small_input=True)
            attended = attention(query, keys, values, self.heads, mask=mask, skip_reshape=True, skip_output_reshape=True)[..., :self.value_dim]
        return self.o_proj(attended.transpose(1, 2).reshape(batch, count, -1)), next_cache


class InternVideoLayer(nn.Module):
    def __init__(self, config, index, dtype, device, operations):
        super().__init__()
        self.input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = LatentAttention(config, index, dtype, device, operations)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=operations)

    def forward(self, hidden, frequencies, cache, mask):
        attended, cache = self.self_attn(self.input_layernorm(hidden), frequencies, cache, mask)
        hidden = hidden + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), cache


class InternVideoDecoder(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.config = config
        config.mlp_activation = config.hidden_act
        self.embed_tokens = operations.Embedding(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([InternVideoLayer(config, i, dtype, device, operations) for i in range(config.num_hidden_layers)])
        self.norm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.lm_head = operations.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)

    def step(self, hidden, positions, cache):
        past = 0 if cache is None else cache[0][0].shape[1]
        count = hidden.shape[1]
        mask = None
        if count > 1:
            future = torch.arange(past + count, device=hidden.device)[None] > torch.arange(past, past + count, device=hidden.device)[:, None]
            mask = hidden.new_zeros((1, 1, count, past + count)).masked_fill(future, torch.finfo(hidden.dtype).min)
        frequencies = rotary_frequencies(positions, self.config.qk_rope_head_dim, self.config.rope_theta, self.config.rope_scaling.mrope_section)
        next_cache = []
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), hidden.device, {'prefetch_dynamic_vbars': True})
        for i, layer in enumerate(self.layers):
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden, current = layer(hidden, frequencies, None if cache is None else cache[i], mask)
                next_cache.append(current)
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='block')
        return self.norm(hidden), next_cache


class InternVideoNetwork(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.model = InternVideoDecoder(namespace(config['text_config']), dtype, device, operations)
        self.visual = Qwen35VisionModel(config['vision_config'], dtype=dtype, device=device, ops=operations)

    def get_dynamic_vram__units(self):
        return [*self.model.layers, *self.visual.blocks], []


def internvideo_prompt(tokenizer, prompt, grid, timestamps):
    _, height, width = grid[0].tolist()
    media = ''.join(f'<{time:.1f} seconds><|vision_start|>' + '<|video_pad|>' * (height * width // 4) + '<|vision_end|>' for time in timestamps)
    # The pinned template has neither a default system message nor an inserted
    # newline between video and user text.
    text = '<|im_start|>user\n' + media + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids])


class InternVideoCaptioner:
    family = 'internvideo3_8b'
    supports_audio = False

    def __init__(self, state_dict, metadata):
        if metadata.get('conversion_version') != '1' or metadata.get('architecture') != self.family:
            raise ValueError('Use convert_video_captioner.py --family internvideo3_8b')
        self.load_device = comfy.model_management.text_encoder_device()
        offload = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.bfloat16 if comfy.model_management.supports_dtype(self.load_device, torch.bfloat16) else torch.float32
        self.tokenizer = Tokenizer.from_str(state_dict.pop('tokenizer_json').numpy().tobytes().decode())
        config = json.loads(metadata['captioner_config'])
        generation = json.loads(json.loads(metadata['captioner_assets'])['generation_config.json'])
        self.eos_tokens = frozenset(generation['eos_token_id'])
        self.config = namespace(config)
        self.network = InternVideoNetwork(config, self.dtype, offload, comfy.ops.manual_cast).eval()
        self.network.load_state_dict(state_dict, strict=True, assign=True)
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.network, load_device=self.load_device, offload_device=offload,
            fast_disk=comfy.storage.state_dict_fast_disk(state_dict))

    def caption(self, images, source_fps, audio, prompt, fps, max_pixels, max_tokens, max_events):
        if audio is not None:
            raise ValueError('InternVideo3 has no audio encoder; use visual frames only')
        patches, grid, timestamps = prepare_video(images, source_fps, fps, 16, max_pixels)
        ids = internvideo_prompt(self.tokenizer, prompt, grid, timestamps)
        config = self.network.model.config
        total = ids.shape[1] + max_tokens
        if total > config.max_position_embeddings:
            raise ValueError('InternVideo3 context exceeded; reduce frames, resolution or output tokens')
        memory = sum(rank + config.qk_rope_head_dim for rank in config.kv_lora_rank_list) * total * 2
        # Prefill temporarily expands one layer's KV; it is not retained in cache.
        memory += ids.shape[1] * config.num_attention_heads * (2 * config.qk_nope_head_dim + config.qk_rope_head_dim + config.v_head_dim) * 2
        memory += max(ids.shape[1] * config.hidden_size * 64, int(grid[0, 1:].prod()) ** 2 * 4)
        comfy.model_management.load_models_gpu([self.patcher], memory_required=memory)
        ids = ids.to(self.load_device)
        hidden = self.network.model.embed_tokens(ids, out_dtype=self.dtype)
        visual = self.network.visual(patches.to(self.load_device, dtype=self.dtype), grid.to(self.load_device))
        hidden[ids == self.config.video_token_id] = visual.to(hidden)
        position_owner = SalmonnPositions()
        position_owner.config = self.config
        positions, delta = position_owner.get_rope_index(ids, video_grid_thw=grid)
        cache, output = None, []
        for _ in range(max_tokens):
            comfy.model_management.throw_exception_if_processing_interrupted()
            hidden, cache = self.network.model.step(hidden, positions, cache)
            token = int(self.network.model.lm_head(hidden[:, -1:]).argmax(-1).item())
            if token in self.eos_tokens:
                break
            output.append(token)
            hidden = self.network.model.embed_tokens(torch.tensor([[token]], device=self.load_device), out_dtype=self.dtype)
            positions = torch.full((3, 1, 1), ids.shape[1] + len(output) - 1 + int(delta[0]), device=self.load_device, dtype=torch.long)
        return dict(text=self.tokenizer.decode(output, skip_special_tokens=True).strip(), events=[], generated_tokens=len(output))
