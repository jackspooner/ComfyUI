# Copyright (c) 2025 Amphion. MIT; see dualcodec.LICENSE.
# Adapted from DualCodec 0.4.2, 25hz_v1. DAC components derive from
# https://github.com/descriptinc/descript-audio-codec (MIT).
"""DualCodec discrete semantic/acoustic codec on native operations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.ops
from comfy.ldm.audio.moss_soundeffect_vae import Encoder, ResidualUnit
from comfy.ldm.minimax_music.dav import Snake1d


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, dtype, device, operations):
        super().__init__()
        self.dwconv = operations.Conv1d(dim, dim, 7, groups=dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.pwconv1 = operations.Linear(dim, intermediate_dim, dtype=dtype, device=device)
        self.pwconv2 = operations.Linear(intermediate_dim, dim, dtype=dtype, device=device)

    def forward(self, x):
        value = self.dwconv(F.pad(x, (6, 0))).transpose(1, 2)
        return x+self.pwconv2(F.gelu(self.pwconv1(self.norm(value)))).transpose(1, 2)


class VectorQuantize(nn.Module):
    def __init__(self, dim, size, code_dim, dtype, device, operations):
        super().__init__()
        self.in_proj = operations.Conv1d(dim, code_dim, 1, dtype=dtype, device=device)
        self.out_proj = operations.Conv1d(code_dim, dim, 1, dtype=dtype, device=device)
        self.codebook = operations.Embedding(size, code_dim, dtype=dtype, device=device)

    def decode(self, ids):
        return self.out_proj(self.codebook(ids).transpose(1, 2))

    def encode(self, x):
        latent = self.in_proj(x).transpose(1, 2)
        values = F.normalize(latent.flatten(0, 1))
        book = F.normalize(comfy.ops.cast_to_input(self.codebook.weight, latent))
        # Keep the published squared-distance expression, including its norms;
        # simplifying to a dot product changes tie decisions for near equal codes.
        distance = values.square().sum(1, keepdim=True)-2*values@book.t()+book.square().sum(1)[None]
        ids = (-distance).max(1).indices.reshape(latent.shape[:2])
        return self.decode(ids), ids


class ResidualVectorQuantize(nn.Module):
    def __init__(self, dim, books, size, code_dim, dtype, device, operations):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantize(dim, size, code_dim, dtype, device, operations) for _ in range(books)])

    def encode(self, x, books):
        codes = []
        quantized = torch.zeros_like(x)
        for quantizer in self.quantizers[:books]:
            mm.throw_exception_if_processing_interrupted()
            value, ids = quantizer.encode(x)
            x = x-value
            quantized = quantized+value
            codes.append(ids)
        return quantized, torch.stack(codes, 1)

    def decode(self, codes):
        result = 0.
        for index, quantizer in enumerate(self.quantizers[:codes.shape[1]]):
            mm.throw_exception_if_processing_interrupted()
            result = result+quantizer.decode(codes[:, index])
        return result


class DecoderBlock(nn.Module):
    def __init__(self, channels, stride, dtype, device, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels, dtype, device),
            # Unlike MOSS SoundEffect's DAC, odd strides have no output padding.
            operations.ConvTranspose1d(channels, channels//2, 2*stride, stride=stride,
                padding=math.ceil(stride/2), dtype=dtype, device=device),
            *(ResidualUnit(channels//2, dilation, dtype, device, operations) for dilation in (1, 3, 9)))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, latent, channels, rates, dtype, device, operations):
        super().__init__()
        layers = [operations.Conv1d(latent, channels, 7, padding=3, dtype=dtype, device=device)]
        for stride in rates:
            layers.append(DecoderBlock(channels, stride, dtype, device, operations))
            channels //= 2
        self.model = nn.Sequential(*layers, Snake1d(channels, dtype, device),
            operations.Conv1d(channels, 1, 7, padding=3, dtype=dtype, device=device), nn.Tanh())

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module):
    def __init__(self, encoder_dim, latent_dim, decoder_dim, rates, books, size, code_dim, dtype, device, operations):
        super().__init__()
        self.encoder = Encoder(encoder_dim, rates, latent_dim, dtype, device, operations)
        self.decoder = Decoder(latent_dim, decoder_dim, tuple(reversed(rates)), dtype, device, operations)
        self.quantizer = ResidualVectorQuantize(latent_dim, books, size, code_dim, dtype, device, operations)


class DualCodec(nn.Module):
    def __init__(self, encoder_dim=64, latent_dim=1024, decoder_dim=1536, rates=(4, 5, 6, 8),
                 books=11, size=1024, semantic_size=16384, code_dim=8, convnext_dim=768,
                 convnext_inner=2048, convnext_layers=4, dtype=None, device=None, operations=None):
        super().__init__()
        self.hop_length = math.prod(rates)
        self.dac = DAC(encoder_dim, latent_dim, decoder_dim, rates, books, size, code_dim, dtype, device, operations)
        self.convnext_encoder = nn.Sequential(operations.Conv1d(latent_dim, convnext_dim, 1, dtype=dtype, device=device),
            *(ConvNeXtBlock(convnext_dim, convnext_inner, dtype, device, operations) for _ in range(convnext_layers)))
        self.semantic_vq = ResidualVectorQuantize(convnext_dim, 1, semantic_size, code_dim, dtype, device, operations)
        self.convnext_decoder = nn.Sequential(
            *(ConvNeXtBlock(convnext_dim, convnext_inner, dtype, device, operations) for _ in range(convnext_layers)),
            operations.Conv1d(convnext_dim, latent_dim, 1, dtype=dtype, device=device))

    def encode(self, audio, semantic, books=12):
        if not 1 <= books <= len(self.dac.quantizer.quantizers)+1:
            raise ValueError('DualCodec requires an ordered prefix of its twelve codebooks')
        value, semantic_codes = self.semantic_vq.encode(self.convnext_encoder(semantic), 1)
        if books == 1:
            return semantic_codes
        semantic = self.convnext_decoder(value)
        audio = F.pad(audio, (0, -audio.shape[-1] % self.hop_length))
        latent = self.dac.encoder(audio)
        if abs(latent.shape[-1]-semantic.shape[-1]) > 2 or latent.shape[-1] < semantic.shape[-1]:
            raise ValueError('DualCodec acoustic and semantic frame counts do not agree')
        _, acoustic_codes = self.dac.quantizer.encode(latent[..., :semantic.shape[-1]]-semantic, books-1)
        return torch.cat((semantic_codes, acoustic_codes), 1)

    def decode(self, codes):
        semantic = self.convnext_decoder(self.semantic_vq.decode(codes[:, :1]))
        if codes.shape[1] > 1:
            semantic = semantic+self.dac.quantizer.decode(codes[:, 1:])
        return self.dac.decoder(semantic)
