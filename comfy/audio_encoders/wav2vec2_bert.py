# Copyright 2023 The HuggingFace Inc. team. Apache-2.0.
# Adapted from transformers.models.wav2vec2_bert for native ComfyUI operations.
"""W2V-BERT 2.0 conformer with Shaw relative-key attention and causal convolution."""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device


class FeatureProjection(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim = config['feature_projection_input_dim']
        self.layer_norm = operations.LayerNorm(dim, eps=config['layer_norm_eps'], dtype=dtype, device=device)
        self.projection = operations.Linear(dim, config['hidden_size'], dtype=dtype, device=device)

    def forward(self, x):
        return self.projection(self.layer_norm(x))


class FeedForward(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim, inner = config['hidden_size'], config['intermediate_size']
        self.intermediate_dense = operations.Linear(dim, inner, dtype=dtype, device=device)
        self.output_dense = operations.Linear(inner, dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.output_dense(F.silu(self.intermediate_dense(x)))


class Convolution(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim, kernel = config['hidden_size'], config['conv_depthwise_kernel_size']
        self.layer_norm = operations.LayerNorm(dim, eps=config['layer_norm_eps'], dtype=dtype, device=device)
        self.pointwise_conv1 = operations.Conv1d(dim, dim*2, 1, bias=False, dtype=dtype, device=device)
        self.depthwise_conv = operations.Conv1d(dim, dim, kernel, groups=dim, bias=False, dtype=dtype, device=device)
        self.depthwise_layer_norm = operations.LayerNorm(dim, eps=config['layer_norm_eps'], dtype=dtype, device=device)
        self.pointwise_conv2 = operations.Conv1d(dim, dim, 1, bias=False, dtype=dtype, device=device)
        self.padding = kernel-1

    def forward(self, x, valid):
        x = self.layer_norm(x).masked_fill(~valid[..., None], 0)
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.padding, 0))).transpose(1, 2)
        return self.pointwise_conv2(F.silu(self.depthwise_layer_norm(x)).transpose(1, 2)).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim = config['hidden_size']
        self.heads = config['num_attention_heads']
        self.left, self.right = config['left_max_position_embeddings'], config['right_max_position_embeddings']
        self.linear_q = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.linear_k = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.linear_v = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.linear_out = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.distance_embedding = operations.Embedding(self.left+self.right+1, dim//self.heads, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, valid):
        batch, length, dim = x.shape
        q, k, v = (layer(x).reshape(batch, length, self.heads, -1).transpose(1, 2)
                   for layer in (self.linear_q, self.linear_k, self.linear_v))
        positions = torch.arange(length, device=x.device)
        distances = (positions[None]-positions[:, None]).clamp(-self.left, self.right)+self.left
        relative = self.distance_embedding(torch.arange(self.left+self.right+1, device=x.device))
        scores = torch.einsum('bhld,rd->bhlr', q, relative)
        bias = scores.gather(-1, distances[None, None].expand(batch, self.heads, -1, -1))*(dim//self.heads)**-0.5
        bias.masked_fill_(~valid[:, None, None, :], torch.finfo(x.dtype).min)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        x = attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
            self.heads, mask=bias, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.linear_out(x)


class EncoderLayer(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim, eps = config['hidden_size'], config['layer_norm_eps']
        for name in ('ffn1_layer_norm', 'self_attn_layer_norm', 'ffn2_layer_norm', 'final_layer_norm'):
            setattr(self, name, operations.LayerNorm(dim, eps=eps, dtype=dtype, device=device))
        self.ffn1 = FeedForward(config, dtype, device, operations)
        self.ffn2 = FeedForward(config, dtype, device, operations)
        self.self_attn = Attention(config, dtype, device, operations)
        self.conv_module = Convolution(config, dtype, device, operations)

    def forward(self, x, valid):
        x = x+0.5*self.ffn1(self.ffn1_layer_norm(x))
        x = x+self.self_attn(self.self_attn_layer_norm(x), valid)
        x = x+self.conv_module(x, valid)
        return self.final_layer_norm(x+0.5*self.ffn2(self.ffn2_layer_norm(x)))


class Encoder(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config, dtype, device, operations) for _ in range(config['num_hidden_layers'])])

    def forward(self, x, valid, output_layer, options):
        x = x.masked_fill(~valid[..., None], 0)
        layers = list(self.layers[:output_layer])
        queue = prefetch.make_prefetch_queue(layers, x.device, options)
        for layer in layers:
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, valid)
            prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return x


class Wav2Vec2Bert(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.feature_projection = FeatureProjection(config, dtype, device, operations)
        self.encoder = Encoder(config, dtype, device, operations)
        # The pinned inventory includes this training parameter, even in eval.
        self.masked_spec_embed = nn.Parameter(torch.empty(config['hidden_size'], dtype=dtype, device=device))

    def forward(self, features, valid, output_layer=17, options=None):
        return self.encoder(self.feature_projection(features), valid.bool(), output_layer, options or {})
