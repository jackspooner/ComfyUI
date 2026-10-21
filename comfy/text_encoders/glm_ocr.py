"""Native Z.ai GLM-OCR. Architecture follows Transformers' Apache-2.0 GLM-OCR implementation."""
import math
from .chat import chat_segments, encode_chat
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
from comfy import sd1_clip
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from .llama import BaseLlama, BaseGenerate
from .moss_audio import EmbeddedTokenizer


TEXT_CONFIG = dict(hidden_size=1536, intermediate_size=4608, num_hidden_layers=16, num_attention_heads=16,
                   num_key_value_heads=8, head_dim=128, vocab_size=59392, rms_norm_eps=1e-5,
                   max_position_embeddings=131072, rope_theta=10000, mrope_section=[16, 24, 24],
                   stop_tokens=[59246, 59253])
VISION_CONFIG = dict(hidden_size=1024, intermediate_size=4096, depth=24, num_heads=16,
                     out_hidden_size=1536, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
                     rms_norm_eps=1e-5, rope_theta=10000)
IMAGE_MARKER = '<|begin_of_image|><|image|><|end_of_image|>'


def image_geometry(height, width):
    # Reduce only above one megapixel. Pad right/bottom to the native merge grid; never enlarge pixels.
    scale = min(1.0, math.sqrt(1_000_000 / (height * width)))
    h, w = max(1, int(height * scale)), max(1, int(width * scale))
    return h, w, ((h + 27) // 28) * 28, ((w + 27) // 28) * 28


def process_image(image):
    h, w, padded_h, padded_w = image_geometry(image.shape[1], image.shape[2])
    x = image[..., :3].movedim(-1, 1)
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
    x = F.pad(x, (0, padded_w - w, 0, padded_h - h), value=1.0)
    mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    x = (x - mean) / std
    gh, gw = padded_h // 14, padded_w // 14
    x = x.reshape(1, 3, gh // 2, 2, 14, gw // 2, 2, 14).permute(0, 2, 5, 3, 6, 1, 4, 7)
    patches = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(-1, 3, 2, 14, 14)
    return patches, (gh, gw)


def rotation_matrix(angles):
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(*angles.shape, 2, 2)


def block_loop(layers, x, call):
    queue = comfy.model_prefetch.make_prefetch_queue(list(layers), x.device, {'prefetch_dynamic_vbars': True})
    for i, layer in enumerate(layers):
        comfy.model_management.throw_exception_if_processing_interrupted()
        def core():
            nonlocal x
            x = call(layer, x, i)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
    return x


class VisionAttention(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        dim, heads = config['hidden_size'], config['num_heads']
        self.heads, self.head_dim = heads, dim // heads
        self.comfy_attention = ComfyAttention()
        self.qkv = operations.Linear(dim, dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=config['rms_norm_eps'], device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=config['rms_norm_eps'], device=device, dtype=dtype)

    def forward(self, x, rope):
        q, k, v = self.qkv(x).reshape(1, x.shape[0], 3, self.heads, self.head_dim).unbind(2)
        with comfy.ops.CastBiasWeightContext(self.q_norm, q, offloadable=True) as (qw, _):
            with comfy.ops.CastBiasWeightContext(self.k_norm, k, offloadable=True) as (kw, _):
                q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope, qw, kw, self.q_norm.eps)
        q, k, v = [AttentionTensorContainer(t.transpose(1, 2)) for t in (q, k, v)]
        return self.proj(optimized_attention(q, k, v, self.heads, skip_reshape=True, preferred_attention=self.comfy_attention).squeeze(0))


class VisionMLP(nn.Module):
    def __init__(self, dim, intermediate, operations, device, dtype, bias):
        super().__init__()
        self.gate_proj = operations.Linear(dim, intermediate, bias=bias, device=device, dtype=dtype)
        self.up_proj = operations.Linear(dim, intermediate, bias=bias, device=device, dtype=dtype)
        self.down_proj = operations.Linear(intermediate, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.norm1 = operations.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'], device=device, dtype=dtype)
        self.norm2 = operations.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'], device=device, dtype=dtype)
        self.attn = VisionAttention(config, operations, device, dtype)
        self.mlp = VisionMLP(config['hidden_size'], config['intermediate_size'], operations, device, dtype, True)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class VisionMerger(VisionMLP):
    def __init__(self, dim, operations, device, dtype):
        super().__init__(dim, dim * 3, operations, device, dtype, False)
        self.proj = operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.post_projection_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x):
        return super().forward(F.gelu(self.post_projection_norm(self.proj(x))))


class VisionModel(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = operations.Conv3d(3, config['hidden_size'], (2, 14, 14), stride=(2, 14, 14), device=device, dtype=dtype)
        self.blocks = nn.ModuleList([VisionBlock(config, operations, device, dtype) for _ in range(config['depth'])])
        self.post_layernorm = operations.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'], device=device, dtype=dtype)
        self.downsample = operations.Conv2d(config['hidden_size'], config['out_hidden_size'], 2, stride=2, device=device, dtype=dtype)
        self.merger = VisionMerger(config['out_hidden_size'], operations, device, dtype)

    def forward(self, patches, grid):
        gh, gw = grid
        x = self.patch_embed.proj(patches).flatten(1)
        h, w = torch.meshgrid(torch.arange(gh, device=x.device), torch.arange(gw, device=x.device), indexing='ij')
        positions = torch.stack([p.reshape(gh // 2, 2, gw // 2, 2).transpose(1, 2).flatten() for p in (h, w)], dim=-1)
        head_dim = self.config['hidden_size'] // self.config['num_heads']
        inv = self.config['rope_theta'] ** (-torch.arange(0, head_dim // 2, 2, device=x.device, dtype=torch.float32) / (head_dim // 2))
        angles = (positions[..., None] * inv).flatten(1)
        rope = rotation_matrix(angles).unsqueeze(0).unsqueeze(2)
        x = block_loop(self.blocks, x, lambda layer, value, _: layer(value, rope))
        x = self.post_layernorm(x).reshape(-1, 2, 2, self.config['hidden_size']).permute(0, 3, 1, 2)
        return self.merger(self.downsample(x).flatten(1))


class TextAttention(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.comfy_attention = ComfyAttention()
        for name, size in [('q_proj', self.heads * self.head_dim), ('k_proj', self.kv_heads * self.head_dim), ('v_proj', self.kv_heads * self.head_dim)]:
            setattr(self, name, operations.Linear(config.hidden_size, size, bias=False, device=device, dtype=dtype))
        self.o_proj = operations.Linear(self.heads * self.head_dim, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, rope, mask, cache):
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = comfy.quant_ops.ck.apply_rope(q, k, rope)
        if cache is not None:
            key, value, index = cache
            key[:, :, index:index + n].copy_(k)
            value[:, :, index:index + n].copy_(v)
            k, v = key[:, :, :index + n], value[:, :, :index + n]
            cache = (key, value, index + n)
        q, k, v = [AttentionTensorContainer(t) for t in (q, k, v)]
        x = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, enable_gqa=self.kv_heads != self.heads, preferred_attention=self.comfy_attention)
        return self.o_proj(x), cache


class TextMLP(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.gate_up_proj = operations.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TextBlock(nn.Module):
    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.self_attn = TextAttention(config, operations, device, dtype)
        self.mlp = TextMLP(config, operations, device, dtype)
        for name in ('input_layernorm', 'post_attention_layernorm', 'post_self_attn_layernorm', 'post_mlp_layernorm'):
            setattr(self, name, operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype))

    def forward(self, x, rope, mask, cache):
        attended, cache = self.self_attn(self.input_layernorm(x), rope, mask, cache)
        x = x + self.post_self_attn_layernorm(attended)
        return x + self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x))), cache


class TextModel(nn.Module):
    graph_dynamic_vbar_blocks = False

    def __init__(self, config, operations, device, dtype):
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.embed_tokens = operations.Embedding(self.config.vocab_size, self.config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TextBlock(self.config, operations, device, dtype) for _ in range(self.config.num_hidden_layers)])
        self.norm = operations.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, device=device, dtype=dtype)
        self.lm_head = operations.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, device=device, dtype=dtype)

    def init_kv_cache(self, batch, capacity, device, dtype):
        shape = (batch, self.config.num_key_value_heads, capacity, self.config.head_dim)
        return [(torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype), 0) for _ in self.layers]

    def forward(self, x, attention_mask=None, embeds=None, past_key_values=None, position_ids=None, **kwargs):
        x = self.embed_tokens(x) if embeds is None else embeds
        n = x.shape[1]
        past = past_key_values[0][2] if past_key_values else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + n, device=x.device).unsqueeze(0)
        positions = position_ids.expand(3, -1)
        inv = self.config.rope_theta ** (-torch.arange(0, self.config.head_dim, 2, device=x.device, dtype=torch.float32) / self.config.head_dim)
        angles = positions[..., None].float() * inv
        angles = torch.cat([part[i % 3] for i, part in enumerate(angles.split(self.config.mrope_section, dim=-1))], dim=-1)
        # Upstream rounds cos/sin to compute dtype before text rotation.
        rope = rotation_matrix(angles).to(x.dtype).float().unsqueeze(0).unsqueeze(0)
        mask = None
        if n > 1:
            mask = x.new_full((n, past + n), torch.finfo(x.dtype).min).triu_(past + 1)
        caches = list(past_key_values) if past_key_values is not None else [None] * len(self.layers)
        def call(layer, value, i):
            value, caches[i] = layer(value, rope, mask, caches[i])
            return value
        x = block_loop(self.layers, x, call)
        return self.norm(x), None, caches


class GlmOCR(BaseLlama, BaseGenerate, nn.Module):
    def image_token_length(self, embed):
        _, _, h, w = image_geometry(*embed['data'].shape[1:3])
        return h * w // (28 * 28)

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        text = {**TEXT_CONFIG, **config_dict.get('text', {})}
        vision = {**VISION_CONFIG, **config_dict.get('vision', {})}
        self.model = TextModel(text, operations, device, dtype)
        self.visual = VisionModel(vision, operations, device, dtype)
        self.num_layers, self.dtype = text['num_hidden_layers'], dtype

    def get_dynamic_vram__units(self):
        return [*self.visual.blocks, *self.model.layers], []

    def preprocess_embed(self, embed, device):
        if embed['type'] != 'image':
            return None, None
        patches, grid = process_image(embed['data'])
        return self.visual(patches.to(device=device, dtype=self.dtype), grid), {'grid': grid}

    def image_positions(self, embeds, info):
        positions, offset, cursor = [], 0, 0
        for entry in info:
            start, size = entry['index'], entry['size']
            length = start - cursor
            positions.append(torch.arange(offset, offset + length, device=embeds.device).expand(3, -1))
            offset += length
            h, w = [value // 2 for value in entry['extra']['grid']]
            y, x = torch.meshgrid(torch.arange(h, device=embeds.device), torch.arange(w, device=embeds.device), indexing='ij')
            positions.append(torch.stack((torch.full_like(x, offset), y + offset, x + offset)).flatten(1))
            offset += max(h, w)
            cursor = start + size
        positions.append(torch.arange(offset, offset + embeds.shape[1] - cursor, device=embeds.device).expand(3, -1))
        return torch.cat(positions, dim=1)


class GlmOCRClipModel(sd1_clip.SDClipModel):
    def __init__(self, device='cpu', layer='last', layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, dtype=dtype, textmodel_json_config={},
                         special_tokens={'pad': 59246}, model_class=GlmOCR, layer_norm_hidden_state=False, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0, mtp=True):
        if isinstance(tokens, dict):
            tokens = tokens['glm_ocr']
        embeds, _, _, info = self.process_tokens([[t[0] for t in row] for row in tokens], self.execution_device)
        max_length = self.transformer.resolve_max_length(max_length, embeds.shape[1])
        if embeds.shape[1] + max_length > self.transformer.model.config.max_position_embeddings:
            raise ValueError('GLM-OCR input and output budget exceed its 131072-token context.')
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                                         execution_dtype=self.transformer.dtype, presence_penalty=presence_penalty,
                                         position_ids=self.transformer.image_positions(embeds, info))


class GlmOCRSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data.get('tokenizer_json')
        super().__init__(self.tokenizer_json, embedding_directory=embedding_directory, embedding_size=1536,
                         embedding_key='glm_ocr', tokenizer_class=EmbeddedTokenizer, has_start_token=False, has_end_token=False,
                         pad_with_end=False, pad_to_max_length=False, max_length=99999999, min_length=1,
                         pad_token=59246, tokenizer_data=tokenizer_data, disable_weights=True)

    def state_dict(self):
        return {'tokenizer_json': self.tokenizer_json}


class GlmOCRTokenizer(sd1_clip.SD1Tokenizer):
    def tokenize_chat(self, messages, thinking=False):
        segments = chat_segments(messages, '<|{role}|>\n', '', '<|assistant|>\n',
                                 '<|begin_of_image|>', '<|end_of_image|>', prefix='[gMASK]<sop>')
        return {'glm_ocr': encode_chat(segments, lambda text: self.glm_ocr.tokenizer.encode(text, add_special_tokens=False))}

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name='glm_ocr', tokenizer=GlmOCRSDTokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, skip_template=False, system_prompt='', **kwargs):
        if image is not None and image.shape[0] != 1:
            raise ValueError('GLM-OCR currently accepts one image per invocation; submit independent crops separately.')
        if not skip_template:
            system = '<|system|>\n' + system_prompt if system_prompt else ''
            text = '[gMASK]<sop>' + system + '<|user|>\n' + (IMAGE_MARKER if image is not None else '') + text + '<|assistant|>\n'
        if text.count(IMAGE_MARKER) != int(image is not None):
            raise ValueError('GLM-OCR needs one native image marker for its supplied image, and none for text only.')
        tokenizer = self.glm_ocr.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [{'type': 'image', 'data': image} if token == 59280 and image is not None else token for token in ids]
        return {'glm_ocr': [[(t, 1.0, i + 1) if return_word_ids else (t, 1.0) for i, t in enumerate(ids)]]}


class GlmOCRTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name='glm_ocr', clip_model=GlmOCRClipModel, model_options=model_options)

    def generation_memory_estimation_function(self, tokens, max_length, device):
        rows = tokens['glm_ocr']
        total, patches = len(rows[0]), 0
        for token, *_ in rows[0]:
            if isinstance(token, dict) and token['type'] == 'image':
                image = token['data']
                _, _, h, w = image_geometry(image.shape[1], image.shape[2])
                image_patches = h * w // (14 * 14)
                patches = max(patches, image_patches)
                total += image_patches // 4 - 1
        config = self.glm_ocr.transformer.model.config
        budget = self.glm_ocr.transformer.resolve_max_length(max_length, total)
        if total + budget > config.max_position_embeddings:
            raise ValueError('GLM-OCR input and output budget exceed its 131072-token context.')
        # KV includes all layers and the full requested budget. Workspace also covers an
        # unfused attention backend's FP32 score matrix and prefill/vision activations.
        size = comfy.model_management.dtype_size(self.glm_ocr.transformer.dtype)
        kv = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * (total + budget) * size
        workspace = max(16 * patches * patches * 4, config.num_attention_heads * total * total * 4,
                        32 * patches * VISION_CONFIG['intermediate_size'] * size,
                        32 * total * config.intermediate_size * size)
        return kv + workspace + 256 * 1024 * 1024


def te(dtype_llama):
    class GlmOCRTEModel_(GlmOCRTEModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            super().__init__(device=device, dtype=dtype_llama, model_options=model_options)
    return GlmOCRTEModel_
