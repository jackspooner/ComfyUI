# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0; see moss_audio_tokenizer.LICENSE.
"""Pinned stereo Tokenizer v2; model residency belongs to AudioCodec.

Adapted from OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 at
f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169. Reuses native v1 transformer,
quantizer and call-private streaming state while retaining its distinct stack,
FP32 weights/BF16 compute, channel interleave and always-streaming decode.
"""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_for_device
from .moss_audio_tokenizer import CodecAttention, Patch, ProjectedTransformer, ResidualLFQ


class CodecV2Attention(CodecAttention):
    def forward(self, x, rotary, offset=0, cache=None):
        batch, length, dim = x.shape
        q, k, v = self.in_projs[0](x).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, rotary)
        # V2's actual streaming path appends the entire current chunk before
        # retiring old keys. Its retained RingKVCache class is not used here.
        # Reusing v1's ring would overwrite keys needed by earlier queries when
        # an upsampled decode chunk is longer than a stage's attention window.
        if cache is not None:
            if not cache:
                shape = (batch, self.heads, self.context, dim // self.heads)
                cache.extend([torch.zeros(shape, device=k.device, dtype=k.dtype), torch.zeros(shape, device=v.device, dtype=v.dtype)])
            past = cache[0].shape[2]
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
            key_positions = torch.arange(offset-past, offset+length, device=x.device)
            cache[0], cache[1] = k[:, :, -self.context:].clone(), v[:, :, -self.context:].clone()
        else:
            key_positions = torch.arange(length, device=x.device) + offset
        query_positions = torch.arange(length, device=x.device) + offset
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        output = []
        # Match the source's bounded non-streaming SDPA query partition. The
        # explicit local/causal mask is preserved for every selected backend.
        for start in range(0, length, 1500):
            comfy.model_management.throw_exception_if_processing_interrupted()
            end = min(start+1500, length)
            first_key = max(0, k.shape[2]-length+start-self.context+1)
            last_key = k.shape[2]-length+end
            positions = key_positions[first_key:last_key]
            delta = query_positions[start:end, None]-positions[None]
            valid = (positions[None] >= 0) & (delta >= 0) & (delta < self.context)
            mask = torch.zeros(delta.shape, device=x.device, dtype=x.dtype).masked_fill_(~valid, -torch.inf)
            output.append(attention(AttentionTensorContainer(q[:, :, start:end]),
                AttentionTensorContainer(k[:, :, first_key:last_key]), AttentionTensorContainer(v[:, :, first_key:last_key]),
                self.heads, mask=mask, skip_reshape=True, preferred_attention=self.comfy_attention))
        return self.out_projs[0](torch.cat(output, dim=1))


class MossAudioTokenizerV2(nn.Module):
    sample_rate = 48000
    hop_length = 3840
    channels = 2

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        settings = dict(device=device, dtype=dtype, operations=operations,
                        attention_class=CodecV2Attention, force_projection=True)
        # Interleaved stereo starts at 96 kHz. Each context is measured at its
        # own stage's frame rate, rather than reusing v1's mono windows.
        self.encoder = nn.ModuleList([
            Patch(240, True), ProjectedTransformer(240, 384, 768, 12, 3072, 12, 400, **settings),
            Patch(2, True), ProjectedTransformer(768, 384, 768, 12, 3072, 12, 400, **settings),
            Patch(2, True), ProjectedTransformer(768, 384, 768, 12, 3072, 12, 400, **settings),
            Patch(2, True), ProjectedTransformer(768, 384, 768, 12, 3072, 12, 400, **settings),
            Patch(2, True), ProjectedTransformer(768, 640, 768, 12, 3072, 12, 250, **settings),
            Patch(2, True), ProjectedTransformer(1280, 768, 1280, 20, 5120, 32, 125, **settings),
        ])
        self.quantizer = ResidualLFQ(device=device, dtype=dtype, operations=operations)
        self.decoder = nn.ModuleList([
            ProjectedTransformer(768, 1280, 1280, 20, 5120, 32, 125, **settings), Patch(2, False),
            ProjectedTransformer(640, 768, 768, 12, 3072, 12, 250, **settings), Patch(2, False),
            ProjectedTransformer(384, 768, 768, 12, 3072, 12, 400, **settings), Patch(2, False),
            ProjectedTransformer(384, 768, 768, 12, 3072, 12, 400, **settings), Patch(2, False),
            ProjectedTransformer(384, 768, 768, 12, 3072, 12, 400, **settings), Patch(2, False),
            ProjectedTransformer(384, 240, 768, 12, 3072, 12, 400, **settings), Patch(240, False),
        ])
        for module in (*self.encoder, *self.decoder):
            if isinstance(module, ProjectedTransformer):
                for layer in module.transformer.layers:
                    layer.preserve_norm_weight_dtype = True
                    layer.layer_scale_1.preserve_weight_dtype = True
                    layer.layer_scale_2.preserve_weight_dtype = True

    @staticmethod
    def interleave(waveform):
        return waveform.transpose(1, 2).contiguous().reshape(waveform.shape[0], 1, -1)

    @staticmethod
    def restore(waveform):
        return waveform.squeeze(1).contiguous().reshape(waveform.shape[0], -1, 2).transpose(1, 2).contiguous().float()

    def encode(self, waveform, lengths, count=32):
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError('MOSS Tokenizer v2 requires stereo [batch, 2, sample] audio')
        x = self.interleave(F.pad(waveform, (0, -waveform.shape[-1] % self.hop_length)))
        with torch.autocast(device_type=waveform.device.type, dtype=torch.bfloat16, cache_enabled=False):
            for module in self.encoder:
                comfy.model_management.throw_exception_if_processing_interrupted()
                x = module(x)
        code_lengths = [length // self.hop_length for length in lengths]
        return self.quantizer.encode(x.float(), code_lengths, count), code_lengths

    def decode(self, codes, chunk_frames=100):
        states = [({'offset': 0, 'layers': [[] for _ in module.transformer.layers]}
                   if isinstance(module, ProjectedTransformer) else None) for module in self.decoder]
        output = []
        for start in range(0, codes.shape[-1], chunk_frames):
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = self.quantizer.decode(codes[:, :, start:start + chunk_frames]).float()
            with torch.autocast(device_type=codes.device.type, dtype=torch.bfloat16, cache_enabled=False):
                for module, state in zip(self.decoder, states):
                    x = module(x, state) if state is not None else module(x)
            output.append(self.restore(x))
        return torch.cat(output, dim=-1)
