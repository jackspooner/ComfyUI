"""Native Whisper/Q-Former used by video-SALMONN and PadCaptioner."""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.audio_encoders.whisper import AudioEncoder
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class QueryAttention(nn.Module):
    def __init__(self, width, source_width, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.query = operations.Linear(width, width, dtype=dtype, device=device)
        self.key = operations.Linear(source_width, width, dtype=dtype, device=device)
        self.value = operations.Linear(source_width, width, dtype=dtype, device=device)

    def forward(self, query, source):
        q = AttentionTensorContainer(self.query(query))
        k = AttentionTensorContainer(self.key(source))
        v = AttentionTensorContainer(self.value(source))
        return optimized_attention(q, k, v, self.heads, preferred_attention=self.comfy_attention)


class QueryOutput(nn.Module):
    def __init__(self, input_width, width, eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_width, width, dtype=dtype, device=device)
        self.LayerNorm = operations.LayerNorm(width, eps=eps, dtype=dtype, device=device)

    def forward(self, hidden, residual):
        return self.LayerNorm(self.dense(hidden) + residual)


class QueryBlock(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        width = config['hidden_size']
        for name, source_width in [('attention', width), ('crossattention', config['encoder_width'])]:
            attention = nn.Module()
            attention.self = QueryAttention(width, source_width, config['num_attention_heads'], dtype, device, operations)
            attention.output = QueryOutput(width, width, config['layer_norm_eps'], dtype, device, operations)
            self.add_module(name, attention)
        self.intermediate_query = nn.Module()
        self.intermediate_query.dense = operations.Linear(width, config['intermediate_size'], dtype=dtype, device=device)
        self.output_query = QueryOutput(config['intermediate_size'], width, config['layer_norm_eps'], dtype, device, operations)

    def forward(self, hidden, source):
        hidden = self.attention.output(self.attention.self(hidden, hidden), hidden)
        hidden = self.crossattention.output(self.crossattention.self(hidden, source), hidden)
        return self.output_query(F.gelu(self.intermediate_query.dense(hidden)), hidden)


class CaptionAudioEncoder(AudioEncoder):
    def __init__(self, config, dtype, device, operations):
        super().__init__(n_mels=config['num_mel_bins'], n_ctx=config['max_source_positions'],
                         n_state=config['d_model'], n_head=config['encoder_attention_heads'],
                         n_layer=config['encoder_layers'], dtype=dtype, device=device, operations=operations)
        q = config['qformer_config']
        self.window = round(1500 * config['second_per_window'] / 30)
        self.stride = round(1500 * config['second_stride'] / 30)
        self.q_tokens = nn.Parameter(torch.empty(1, q['query_length'], q['hidden_size'], dtype=dtype, device=device))
        self.qformer = nn.Module()
        self.qformer.bert = nn.Module()
        bert = self.qformer.bert
        bert.embeddings = nn.Module()
        bert.embeddings.LayerNorm = operations.LayerNorm(q['hidden_size'], eps=q['layer_norm_eps'], dtype=dtype, device=device)
        bert.encoder = nn.Module()
        bert.encoder.layer = nn.ModuleList([QueryBlock(q, dtype, device, operations) for _ in range(q['num_hidden_layers'])])
        self.audio_proj = operations.Linear(q['hidden_size'], q['out_hidden_size'], dtype=dtype, device=device)

    def forward(self, features):
        hidden = F.gelu(self.conv2(F.gelu(self.conv1(features)))).transpose(1, 2)
        hidden = hidden + comfy.ops.cast_to_input(self.embed_positions.weight, hidden)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), hidden.device, {'prefetch_dynamic_vbars': True})
        for layer in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden = layer(hidden)
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='block')
        hidden = self.layer_norm(hidden)
        batch, _, width = hidden.shape
        source = hidden.unfold(1, self.window, self.stride).permute(0, 1, 3, 2).reshape(-1, self.window, width)
        query = comfy.ops.cast_to_input(self.q_tokens, source).expand(source.shape[0], -1, -1)
        query = self.qformer.bert.embeddings.LayerNorm(query)
        for layer in self.qformer.bert.encoder.layer:
            comfy.model_management.throw_exception_if_processing_interrupted()
            query = layer(query, source)
        return self.audio_proj(query).reshape(batch, -1, self.audio_proj.out_features)
