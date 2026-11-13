# Adapted from viitor-ai/viitor-voice-nar 40359993534de0980eefff2d5a25bddc30f6bf58.
# Apache-2.0; see LICENSE in this directory.
"""ViiTor masked Qwen3 backbone; all audio/text positions attend bidirectionally."""
from dataclasses import replace

import torch
from torch import nn

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen3_06BConfig


class ViiTorVoice(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        values = config['llm_config']
        native = replace(Qwen3_06BConfig(), **{key: values[key] for key in (
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'max_position_embeddings', 'rms_norm_eps')})
        native.head_dim = values['head_dim']
        native.rope_theta = values.get('rope_theta', values.get('rope_parameters', {}).get('rope_theta', 1000000.))
        self.llm = Llama2_(native, device=device, dtype=dtype, ops=operations)
        self.sizes = tuple(config['audio_codebook_sizes'])
        self.semantic_embedding = operations.Embedding(self.sizes[0], native.hidden_size, dtype=dtype, device=device)
        self.acoustic_embedding = operations.Embedding(sum(self.sizes[1:]), native.hidden_size, dtype=dtype, device=device)
        self.semantic_head = operations.Linear(native.hidden_size, self.sizes[0], bias=False, dtype=dtype, device=device)
        self.acoustic_head = operations.Linear(native.hidden_size, sum(self.sizes[1:]), bias=False, dtype=dtype, device=device)
        self.register_buffer('acoustic_codebook_offsets', torch.arange(len(self.sizes)-1, device=device)*self.sizes[1])

    def prepare_embed_inputs(self, ids, audio_mask):
        text = self.llm.embed_tokens(ids[:, 0])
        semantic = self.semantic_embedding(torch.where(audio_mask, ids[:, 0], 0))
        acoustic_ids = torch.where(audio_mask[:, None], ids[:, 1:], 0)
        acoustic = self.acoustic_embedding(acoustic_ids+self.acoustic_codebook_offsets[None, :, None]).sum(1)
        return torch.where(audio_mask[..., None], semantic+acoustic, text).contiguous()

    def forward(self, ids, audio_mask, allowed, options=None):
        x = self.prepare_embed_inputs(ids, audio_mask)
        # Preserve the branch-specific padding mask; never use the causal wrapper.
        mask = torch.zeros_like(allowed, dtype=x.dtype).masked_fill_(~allowed, torch.finfo(x.dtype).min)
        rotary = self.llm.compute_freqs_cis(torch.arange(x.shape[1], device=x.device)[None], x.device)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        queue = prefetch.make_prefetch_queue(list(self.llm.layers), x.device, options or {})
        for layer in self.llm.layers:
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x, _ = layer(x, attention_mask=mask, freqs_cis=rotary, optimized_attention=attention)
            prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        x = self.llm.norm(x)
        semantic = self.semantic_head(x)
        acoustic = self.acoustic_head(x).reshape(*x.shape[:2], len(self.sizes)-1, self.sizes[1]).permute(0, 2, 1, 3)
        return semantic, acoustic
