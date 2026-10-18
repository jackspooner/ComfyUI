"""Native Qwen3-Omni Thinker; architecture follows Qwen/Transformers (Apache-2.0)."""
from dataclasses import dataclass, field, fields
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, WhisperFeatureExtractor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy import sd1_clip
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention as attention
from .llama import Attention, BaseLlama, BaseGenerate, Llama2_, Qwen3VL_8BConfig, TransformerBlock, apply_rope, moe_experts_forward
from .qwen35 import Qwen35VisionPatchEmbed, Qwen35VisionPatchMerger
from .qwen3vl import Qwen3VLDeepstackMerger


ASSET_KEYS = ('tokenizer_json', 'qwen3_omni_config_json', 'qwen3_omni_processor_json',
              'qwen3_omni_tokenizer_config_json', 'qwen3_omni_chat_template', 'qwen3_omni_quant_config_json')
IMAGE_TOKEN, AUDIO_TOKEN = 151655, 151675


def asset_bytes(value):
    return value.numpy().tobytes() if isinstance(value, torch.Tensor) else value


@dataclass
class OmniConfig(Qwen3VL_8BConfig):
    vocab_size: int = 152064
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rope_dims: list = field(default_factory=lambda: [24, 20, 20])
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    fixed_kv: bool = False
    graph_dynamic_vbar_blocks = False
    prefetch_dynamic_vbars = True
    stop_tokens: list = field(default_factory=lambda: [151645])


class TextAttention(Attention):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.comfy_attention = ComfyAttention()

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        def attend(q, k, v, heads, **kwargs):
            return attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                             heads, preferred_attention=self.comfy_attention, **kwargs)
        return super().forward(hidden_states, attention_mask, freqs_cis, attend, past_key_value)


class Experts(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.gate_up_proj = operations.MoEExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size * 2,
                                                 bias=False, device=device, dtype=dtype)
        self.down_proj = operations.MoEExperts(config.num_experts, config.moe_intermediate_size, config.hidden_size,
                                              bias=False, device=device, dtype=dtype)


class SparseMLP(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gate = operations.Linear(config.hidden_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.experts = Experts(config, device, dtype, operations)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probabilities = self.gate(x).float().softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)
        return moe_experts_forward(x, indices, weights, self.num_experts, self.experts.gate_up_proj,
                                   self.experts.down_proj, comfy.ops.INPUT_ACT_EAGER['swiglu']).reshape(shape)


class TextBlock(TransformerBlock):
    def __init__(self, config, device=None, dtype=None, operations=None):
        nn.Module.__init__(self)
        self.self_attn = TextAttention(config, device=device, dtype=dtype, ops=operations)
        self.mlp = SparseMLP(config, device, dtype, operations)
        self.input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)


class TextModel(Llama2_):
    def __init__(self, config, device, dtype, operations):
        nn.Module.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size
        self.fixed_kv = False
        self.graph_dynamic_vbar_blocks = False
        self.prefetch_dynamic_vbars = True
        self.embed_tokens = operations.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TextBlock(config, device, dtype, operations) for _ in range(config.num_hidden_layers)])
        self.norm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.lm_head = operations.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def get_dynamic_vram__units(self):
        return list(self.layers), []

    def compute_freqs_cis(self, position_ids, device):
        # Upstream rounds sin/cos to the compute dtype before rotating Q/K.
        return tuple(value.to(self.norm.weight.dtype).float() for value in super().compute_freqs_cis(position_ids, device))


class AudioAttention(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.heads = config['encoder_attention_heads']
        self.comfy_attention = ComfyAttention()
        dim = config['d_model']
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, operations.Linear(dim, dim, bias=True, device=device, dtype=dtype))

    def forward(self, x, lengths):
        q, k, v = [projection(x).reshape(x.shape[0], self.heads, -1).transpose(0, 1).unsqueeze(0)
                   for projection in (self.q_proj, self.k_proj, self.v_proj)]
        outputs = []
        for query, key, value in zip(q.split(lengths, dim=2), k.split(lengths, dim=2), v.split(lengths, dim=2)):
            outputs.append(attention(AttentionTensorContainer(query), AttentionTensorContainer(key),
                                     AttentionTensorContainer(value), self.heads, skip_reshape=True,
                                     preferred_attention=self.comfy_attention))
        return self.out_proj(torch.cat(outputs, dim=1).squeeze(0))


class AudioBlock(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        dim, intermediate = config['d_model'], config['encoder_ffn_dim']
        self.self_attn = AudioAttention(config, device, dtype, operations)
        self.self_attn_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.fc1 = operations.Linear(dim, intermediate, device=device, dtype=dtype)
        self.fc2 = operations.Linear(intermediate, dim, device=device, dtype=dtype)

    def forward(self, x, lengths):
        x = x + self.self_attn(self.self_attn_layer_norm(x), lengths)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


def audio_output_length(frames, chunk_size=100):
    return frames // chunk_size * ((chunk_size + 7) // 8) + (frames % chunk_size + 7) // 8


class AudioEncoder(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        dim, channels = config['d_model'], config['downsample_hidden_size']
        self.chunk_size = config['n_window'] * 2
        self.window_ratio = config['n_window_infer'] // self.chunk_size
        self.conv_chunksize = config['conv_chunksize']
        self.conv2d1 = operations.Conv2d(1, channels, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d2 = operations.Conv2d(channels, channels, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d3 = operations.Conv2d(channels, channels, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv_out = operations.Linear(channels * ((config['num_mel_bins'] + 7) // 8), dim, bias=False, device=device, dtype=dtype)
        self.layers = nn.ModuleList([AudioBlock(config, device, dtype, operations) for _ in range(config['encoder_layers'])])
        self.ln_post = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.proj1 = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.proj2 = operations.Linear(dim, config['output_dim'], device=device, dtype=dtype)

    def forward(self, features):
        chunks = features.T.split(self.chunk_size)
        lengths = [(chunk.shape[0] + 7) // 8 for chunk in chunks]
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
        encoded = []
        for chunk in padded.split(self.conv_chunksize):
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            encoded.append(x)
        x = torch.cat(encoded)
        b, channels, frequency, time = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(b, time, channels * frequency))
        half = x.shape[-1] // 2
        inv = torch.exp(-math.log(10000.0) / (half - 1) * torch.arange(half, device=x.device, dtype=torch.float32))
        phase = torch.arange(time, device=x.device, dtype=torch.float32)[:, None] * inv
        x = x + torch.cat((phase.sin(), phase.cos()), dim=-1).to(x.dtype)
        x = torch.cat([value[:length] for value, length in zip(x.unbind(), lengths)])
        window = max(lengths) * self.window_ratio
        windows = [min(window, x.shape[0] - start) for start in range(0, x.shape[0], window)]
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {'prefetch_dynamic_vbars': True})
        for layer in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, windows)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return self.proj2(F.gelu(self.proj1(self.ln_post(x))))


class VisionAttention(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.heads = config['num_heads']
        dim = config['hidden_size']
        self.comfy_attention = ComfyAttention()
        self.qkv = operations.Linear(dim, dim * 3, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, lengths, rope):
        q, k, v = self.qkv(x).reshape(x.shape[0], 3, self.heads, -1).unbind(1)
        q, k = apply_rope(q, k, rope)
        outputs = []
        for query, key, value in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
            query, key, value = [AttentionTensorContainer(t.transpose(0, 1).unsqueeze(0)) for t in (query, key, value)]
            outputs.append(attention(query, key, value, self.heads, skip_reshape=True, preferred_attention=self.comfy_attention))
        return self.proj(torch.cat(outputs, dim=1).squeeze(0))


class VisionMLP(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        dim, intermediate = config['hidden_size'], config['intermediate_size']
        self.padding = -intermediate % 32
        self.linear_fc1 = operations.Linear(dim, intermediate, device=device, dtype=dtype)
        self.linear_fc2 = operations.Linear(intermediate + self.padding, dim, device=device, dtype=dtype)

    def forward(self, x):
        x = F.gelu(self.linear_fc1(x), approximate='tanh')
        return self.linear_fc2(F.pad(x, (0, self.padding)))


class VisionBlock(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.norm1 = operations.LayerNorm(config['hidden_size'], eps=1e-6, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(config['hidden_size'], eps=1e-6, device=device, dtype=dtype)
        self.attn = VisionAttention(config, device, dtype, operations)
        self.mlp = VisionMLP(config, device, dtype, operations)

    def forward(self, x, lengths, rope):
        x = x + self.attn(self.norm1(x), lengths, rope)
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.merge_size = config['spatial_merge_size']
        self.grid_side = math.isqrt(config['num_position_embeddings'])
        self.head_dim = config['hidden_size'] // config['num_heads']
        self.patch_embed = Qwen35VisionPatchEmbed(config, device=device, dtype=dtype, ops=operations)
        self.pos_embed = operations.Embedding(config['num_position_embeddings'], config['hidden_size'], device=device, dtype=dtype)
        self.blocks = nn.ModuleList([VisionBlock(config, device, dtype, operations) for _ in range(config['depth'])])
        self.merger = Qwen35VisionPatchMerger(config['hidden_size'], self.merge_size, config['out_hidden_size'],
                                            device=device, dtype=dtype, ops=operations)
        self.deepstack_visual_indexes = config['deepstack_visual_indexes']
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLDeepstackMerger(config['hidden_size'], self.merge_size, config['out_hidden_size'],
                                  device=device, dtype=dtype, ops=operations) for _ in self.deepstack_visual_indexes])

    def positions(self, grids, device):
        positions, rotary, lengths = [], [], []
        inv = 10000.0 ** (-torch.arange(0, self.head_dim // 2, 2, device=device, dtype=torch.float32) / (self.head_dim // 2))
        merge = self.merge_size
        for frames, height, width in grids:
            h = torch.linspace(0, self.grid_side - 1, height, device=device)
            w = torch.linspace(0, self.grid_side - 1, width, device=device)
            hf, wf = h.long(), w.long()
            hc, wc = (hf + 1).clamp_max(self.grid_side - 1), (wf + 1).clamp_max(self.grid_side - 1)
            dh, dw = h - hf, w - wf
            indices = torch.stack([(r[:, None] * self.grid_side + c[None, :]).flatten()
                                   for r, c in ((hf, wf), (hf, wc), (hc, wf), (hc, wc))], dim=1)
            weights = torch.stack([(r[:, None] * c[None, :]).flatten()
                                   for r, c in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))], dim=1)
            pos = (self.pos_embed(indices) * weights.unsqueeze(-1)).sum(dim=1)
            pos = pos.reshape(height // merge, merge, width // merge, merge, -1).permute(0, 2, 1, 3, 4).flatten(0, 3)
            positions.append(pos.repeat(frames, 1))
            rows = torch.arange(height, device=device)[:, None].expand(height, width)
            cols = torch.arange(width, device=device)[None, :].expand(height, width)
            coordinates = torch.stack((rows, cols), dim=-1).reshape(height // merge, merge, width // merge, merge, 2)
            coordinates = coordinates.permute(0, 2, 1, 3, 4).flatten(0, 3).repeat(frames, 1)
            rotary.append((coordinates.float().unsqueeze(-1) * inv).flatten(1))
            lengths.extend([height * width] * frames)
        angles = torch.cat(rotary)
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        sin = angles.sin()
        half = sin.shape[-1] // 2
        return torch.cat(positions), (angles.cos(), sin[..., :half], -sin[..., half:]), lengths

    def forward(self, pixels, grids):
        x = self.patch_embed(pixels)
        positions, rope, lengths = self.positions(grids, x.device)
        x = x + positions.to(x.dtype)
        deepstack = []
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, {'prefetch_dynamic_vbars': True})
        for index, block in enumerate(self.blocks):
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = block(x, lengths, rope)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope='block')
            if index in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)](x))
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return self.merger(x), deepstack


def media_positions(info, length, device):
    positions = torch.empty((3, length), device=device, dtype=torch.float32)
    cursor, next_position = 0, 0
    for image in sorted((entry for entry in info if entry['type'] == 'image'), key=lambda entry: entry['index']):
        start, size = image['index'], image['size']
        positions[:, cursor:start] = torch.arange(next_position, next_position + start - cursor, device=device)
        next_position += start - cursor
        frames, height, width = image['extra']['grid']
        height, width = height // 2, width // 2
        positions[0, start:start + size] = next_position
        positions[1, start:start + size] = torch.arange(height, device=device).repeat_interleave(width).repeat(frames) + next_position
        positions[2, start:start + size] = torch.arange(width, device=device).repeat(height * frames) + next_position
        next_position += max(height, width)
        cursor = start + size
    positions[:, cursor:] = torch.arange(next_position, next_position + length - cursor, device=device)
    return positions


def audio_features(audio, config):
    waveform, sample_rate = audio['waveform'], audio['sample_rate']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or min(waveform.shape[1:]) == 0:
        raise ValueError('Qwen3-Omni requires one nonempty AUDIO clip [1, channels, samples].')
    if not isinstance(sample_rate, int) or sample_rate <= 0 or not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError('Qwen3-Omni requires a positive sample rate and finite floating-point audio.')
    waveform = comfy.audio.resample(waveform.float().mean(dim=1), sample_rate, config['sampling_rate'])
    hop, minimum = config['hop_length'], config['n_fft'] // 2 + 1
    size = max(minimum, waveform.shape[-1])
    # A complete hop also keeps upstream feature/mask lengths equal for short clips.
    size = (size + hop - 1) // hop * hop
    waveform = F.pad(waveform, (0, size - waveform.shape[-1]))
    extractor = WhisperFeatureExtractor.from_dict(config)
    return extractor(waveform[0].detach().cpu().numpy(), sampling_rate=config['sampling_rate'],
                     padding=True, truncation=False, return_tensors='pt').input_features[0]


class Qwen3Omni(BaseLlama, BaseGenerate, nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        source = config_dict['thinker_config']
        text = source['text_config']
        config = OmniConfig(**{item.name: text[item.name] for item in fields(OmniConfig) if item.name in text})
        self.num_layers = config.num_hidden_layers
        self.model = TextModel(config, device, dtype, operations)
        self.audio_tower = AudioEncoder(source['audio_config'], device, dtype, operations)
        self.visual = VisionEncoder(source['vision_config'], device, dtype, operations)
        self.processor_config = config_dict['processor_config']
        self.dtype = dtype

    def get_dynamic_vram__units(self):
        return [self.model.lm_head, *self.visual.blocks, *self.audio_tower.layers, *self.model.layers], [self.model.embed_tokens]

    def preprocess_embed(self, embed, device):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if embed['type'] == 'audio':
            features = audio_features(embed['data'], self.processor_config['feature_extractor'])
            return self.audio_tower(features.to(device=device, dtype=self.dtype)), None
        if embed['type'] == 'image':
            image = embed['data'][0]
            if image.shape[-1] == 1:
                image = image.expand(*image.shape[:-1], 3)
            processor = Qwen2VLImageProcessor.from_dict(self.processor_config['image_processor'])
            processed = processor(images=image[..., :3].movedim(-1, 0), do_rescale=False, return_tensors='pt')
            grids = processed.image_grid_thw.tolist()
            merged, deepstack = self.visual(processed.pixel_values.to(device=device, dtype=self.dtype), grids)
            return merged, {'grid': grids[0], 'deepstack': deepstack}
        return None, None

    def image_inputs(self, embeds, info):
        positions = media_positions(info, embeds.shape[1], embeds.device)
        mask, deepstack = None, None
        for image in (entry for entry in info if entry['type'] == 'image'):
            if mask is None:
                mask = torch.zeros(embeds.shape[:2], device=embeds.device, dtype=torch.bool)
                deepstack = [[] for _ in image['extra']['deepstack']]
            mask[0, image['index']:image['index'] + image['size']] = True
            for parts, value in zip(deepstack, image['extra']['deepstack']):
                parts.append(value)
        return positions, mask, [torch.cat(parts) for parts in deepstack] if deepstack is not None else None


class OmniClipModel(sd1_clip.SDClipModel):
    def __init__(self, device='cpu', layer='last', layer_idx=-1, dtype=None, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, dtype=dtype,
                         textmodel_json_config=model_options['qwen3_omni_config'], special_tokens={'pad': 151643},
                         layer_norm_hidden_state=False, model_class=Qwen3Omni, enable_attention_masks=True,
                         return_attention_masks=True, model_options=model_options)

    def load_sd(self, sd):
        required = set(dict(self.transformer.named_parameters()))
        projections = {}
        for name, module in self.transformer.named_modules():
            if isinstance(module, comfy.ops.MixedPrecisionOp):
                required.add(name + '.weight')
                descriptor = sd.get(name + '.comfy_quant')
                if descriptor is not None:
                    descriptor = json.loads(asset_bytes(descriptor))
                    if not isinstance(descriptor, dict) or descriptor.get('format') not in ('nvfp4', 'float8_e4m3fn'):
                        raise ValueError(f'Unsupported Qwen3-Omni projection format: {name}; expected NVFP4, FP8 E4M3 or dense floating weights.')
                    required.add(name + '.weight_scale')
                    if descriptor['format'] == 'nvfp4':
                        required.add(name + '.weight_scale_2')
                projections[name] = module, descriptor
        missing = sorted(required - set(sd))
        if missing:
            raise ValueError(f'Incomplete Qwen3-Omni artifact: {len(missing)} required tensors missing, including {missing[:8]}. Run the verified converter again.')
        for name, (module, descriptor) in projections.items():
            weight = sd[name + '.weight']
            shape = module._orig_shape
            if descriptor is None:
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32) or tuple(weight.shape) != shape:
                    raise ValueError(f'Invalid Qwen3-Omni dense projection: {name}; expected floating weights with shape {shape}. Packed weights require a quantization descriptor.')
                if any(name + suffix in sd for suffix in ('.weight_scale', '.weight_scale_2')):
                    raise ValueError(f'Qwen3-Omni projection scales without a quantization descriptor: {name}.')
                continue
            leading, rows, columns = shape[:-2], shape[-2], shape[-1]
            if descriptor['format'] == 'float8_e4m3fn':
                scale = sd[name + '.weight_scale']
                if (weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != shape
                        or scale.dtype != torch.float32 or tuple(scale.shape) != leading
                        or descriptor.get('num_experts') != (leading[0] if leading else None)
                        or name + '.weight_scale_2' in sd):
                    raise ValueError(f'Invalid Qwen3-Omni FP8 projection layout or scales: {name}.')
                continue
            blocks, scale = sd[name + '.weight_scale'], sd[name + '.weight_scale_2']
            block_shape = (*leading, (rows + 127) // 128 * 128, (columns // 16 + 3) // 4 * 4)
            if (columns % 32 or weight.dtype != torch.uint8 or tuple(weight.shape) != (*shape[:-1], columns // 2)
                    or blocks.dtype != torch.float8_e4m3fn or tuple(blocks.shape) != block_shape
                    or scale.dtype not in (torch.bfloat16, torch.float16, torch.float32) or tuple(scale.shape) != leading
                    or descriptor.get('num_experts') != (leading[0] if leading else None)):
                raise ValueError(f'Invalid Qwen3-Omni NVFP4 projection layout or scales: {name}. Run the verified converter again.')
        return super().load_sd(sd)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                 presence_penalty=0.0, mtp=True):
        if isinstance(tokens, dict):
            tokens = tokens['qwen3_omni']
        embeds, _, _, info = self.process_tokens([[value[0] for value in row] for row in tokens], self.execution_device)
        budget = self.transformer.resolve_max_length(max_length, embeds.shape[1])
        limit = self.transformer.model.config.max_position_embeddings
        if embeds.shape[1] + budget > limit:
            raise ValueError(f'Qwen3-Omni needs {embeds.shape[1] + budget} context tokens after media expansion; its context holds {limit}. Shorten the input or output budget.')
        positions, mask, deepstack = self.transformer.image_inputs(embeds, info)
        return self.transformer.generate(embeds, do_sample, budget, temperature, top_k, top_p, min_p,
                                         repetition_penalty, seed, presence_penalty=presence_penalty,
                                         position_ids=positions, visual_pos_masks=mask, deepstack_embeds=deepstack)


class EmbeddedOmniTokenizer(PreTrainedTokenizerFast):
    @classmethod
    def from_pretrained(cls, assets, **kwargs):
        config = json.loads(asset_bytes(assets['qwen3_omni_tokenizer_config_json']))
        return cls(tokenizer_object=Tokenizer.from_str(asset_bytes(assets['tokenizer_json']).decode()),
                   eos_token=config['eos_token'], pad_token=config['pad_token'],
                   chat_template=asset_bytes(assets['qwen3_omni_chat_template']).decode())


class OmniSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.assets = {key: tokenizer_data[key] for key in ASSET_KEYS}
        super().__init__(self.assets, embedding_directory=embedding_directory, embedding_size=2048,
                         embedding_key='qwen3_omni', tokenizer_class=EmbeddedOmniTokenizer, has_start_token=False,
                         has_end_token=False, pad_with_end=False, pad_to_max_length=False, max_length=99999999,
                         min_length=1, pad_token=151643, tokenizer_data=tokenizer_data, disable_weights=True)

    def state_dict(self):
        return self.assets.copy()


class Qwen3OmniTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name='qwen3_omni', tokenizer=OmniSDTokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, images=None, audio=None,
                              thinking=False, skip_template=False, system_prompt='', llama_template=None, **kwargs):
        if images is None:
            images = [] if image is None else [image[index:index + 1] for index in range(image.shape[0])]
        tokenizer = self.qwen3_omni.tokenizer
        if skip_template or text.startswith('<|im_start|>'):
            formatted = text
        elif llama_template is not None:
            formatted = llama_template.format(text)
        else:
            content = [{'type': 'image'} for _ in images]
            if audio is not None:
                content.append({'type': 'audio'})
            content.append({'type': 'text', 'text': text})
            messages = [{'role': 'system', 'content': system_prompt}] if system_prompt else []
            messages.append({'role': 'user', 'content': content})
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        if ids.count(IMAGE_TOKEN) != len(images) or ids.count(AUDIO_TOKEN) != int(audio is not None):
            raise ValueError('Qwen3-Omni needs one image placeholder per supplied image and one audio placeholder per supplied clip. Use the default template or provide matching raw placeholders.')
        image_index = 0
        for index, token in enumerate(ids):
            if token == IMAGE_TOKEN:
                ids[index] = {'type': 'image', 'data': images[image_index]}
                image_index += 1
            elif token == AUDIO_TOKEN:
                ids[index] = {'type': 'audio', 'data': audio}
        return {'qwen3_omni': [[(token, 1.0, index + 1) if return_word_ids else (token, 1.0)
                               for index, token in enumerate(ids)]]}


def te(config, dtype_llama=None, llama_quantization_metadata=None):
    class Qwen3OmniTEModel(sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            dtype = dtype_llama or dtype
            options = {**model_options, 'qwen3_omni_config': config}
            if options.get('custom_operations') is None:
                options['custom_operations'] = comfy.ops.mixed_precision_ops(llama_quantization_metadata or {}, dtype, full_precision_mm=True, nvfp4_input_passes=3)
            super().__init__(device=device, dtype=dtype, name='qwen3_omni', clip_model=OmniClipModel, model_options=options)

        def generation_memory_estimation_function(self, tokens, max_length, device):
            model = self.qwen3_omni.transformer
            cfg = model.model.config
            processor = model.processor_config
            total, largest_image, audio_tokens = len(tokens['qwen3_omni'][0]), 0, 0
            for token, *_ in tokens['qwen3_omni'][0]:
                if not isinstance(token, dict):
                    continue
                if token['type'] == 'image':
                    image, image_config = token['data'], processor['image_processor']
                    patch, merge = image_config['patch_size'], image_config['merge_size']
                    h, w = smart_resize(image.shape[1], image.shape[2], factor=patch * merge,
                                        min_pixels=image_config['size']['shortest_edge'],
                                        max_pixels=image_config['size']['longest_edge'])
                    patches = h * w // (patch * patch)
                    total += patches // (merge * merge) - 1
                    largest_image = max(largest_image, patches)
                elif token['type'] == 'audio':
                    audio, feature_config = token['data'], processor['feature_extractor']
                    rate = audio['sample_rate']
                    if not isinstance(rate, int) or rate <= 0:
                        raise ValueError('Qwen3-Omni requires a positive audio sample rate.')
                    # Match the resampling owner's float32 output-length rounding.
                    samples = math.ceil(float(np.float32(audio['waveform'].shape[-1] * feature_config['sampling_rate'] / rate)))
                    hop = feature_config['hop_length']
                    frames = (max(feature_config['n_fft'] // 2 + 1, samples) + hop - 1) // hop
                    audio_tokens = audio_output_length(frames, model.audio_tower.chunk_size)
                    total += audio_tokens - 1
            budget = model.resolve_max_length(max_length, total)
            if total + budget > cfg.max_position_embeddings:
                raise ValueError(f'Qwen3-Omni needs {total + budget} context tokens after media expansion; its context holds {cfg.max_position_embeddings}.')
            size = comfy.model_management.dtype_size(model.dtype)
            kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * (total + budget) * size
            workspace = max(cfg.num_attention_heads * total * total * 4,
                            model.visual.blocks[0].attn.heads * largest_image * largest_image * 4,
                            32 * total * cfg.hidden_size * size,
                            8 * audio_tokens * model.audio_tower.proj1.in_features * size)
            return kv + workspace + 256 * 1024 * 1024
    return Qwen3OmniTEModel
