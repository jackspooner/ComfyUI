"""MOSS SoundEffect v2 continuous DAC. See moss_soundeffect.LICENSE."""
import math

from torch import nn

from comfy.ldm.minimax_music.dav import Snake1d


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, dtype, device, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels, dtype, device),
            operations.Conv1d(channels, channels, 7, dilation=dilation, padding=3 * dilation, dtype=dtype, device=device),
            Snake1d(channels, dtype, device),
            operations.Conv1d(channels, channels, 1, dtype=dtype, device=device),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels, stride, dtype, device, operations):
        super().__init__()
        self.block = nn.Sequential(
            *(ResidualUnit(channels // 2, dilation, dtype, device, operations) for dilation in (1, 3, 9)),
            Snake1d(channels // 2, dtype, device),
            operations.Conv1d(channels // 2, channels, 2 * stride, stride=stride, padding=math.ceil(stride / 2), dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, rates, latent_dim, dtype, device, operations):
        super().__init__()
        layers = [operations.Conv1d(1, channels, 7, padding=3, dtype=dtype, device=device)]
        for stride in rates:
            channels *= 2
            layers.append(EncoderBlock(channels, stride, dtype, device, operations))
        layers.extend((Snake1d(channels, dtype, device), operations.Conv1d(channels, latent_dim, 3, padding=1, dtype=dtype, device=device)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels, stride, dtype, device, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels, dtype, device),
            operations.ConvTranspose1d(channels, channels // 2, 2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2, dtype=dtype, device=device),
            *(ResidualUnit(channels // 2, dilation, dtype, device, operations) for dilation in (1, 3, 9)),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, channels, rates, dtype, device, operations):
        super().__init__()
        layers = [operations.Conv1d(latent_dim, channels, 7, padding=3, dtype=dtype, device=device)]
        for stride in rates:
            layers.append(DecoderBlock(channels, stride, dtype, device, operations))
            channels //= 2
        layers.extend((Snake1d(channels, dtype, device), operations.Conv1d(channels, 1, 7, padding=3, dtype=dtype, device=device), nn.Tanh()))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MossSoundEffectVAE(nn.Module):
    def __init__(self, encoder_dim=128, encoder_rates=(2, 3, 4, 5, 8), latent_dim=128, decoder_dim=2048, decoder_rates=(8, 5, 4, 3, 2), dtype=None, device=None, operations=None):
        super().__init__()
        self.hop_length = math.prod(encoder_rates)
        self.latent_dim = latent_dim
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim, dtype, device, operations)
        self.quant_conv = operations.Conv1d(latent_dim, 2 * latent_dim, 1, dtype=dtype, device=device)
        self.post_quant_conv = operations.Conv1d(latent_dim, latent_dim, 1, dtype=dtype, device=device)
        self.decoder = Decoder(latent_dim, decoder_dim, decoder_rates, dtype, device, operations)

    def encode(self, audio):
        audio = nn.functional.pad(audio, (0, -audio.shape[-1] % self.hop_length))
        return self.quant_conv(self.encoder(audio))[:, :self.latent_dim].clone()

    def decode(self, latent):
        return self.decoder(self.post_quant_conv(latent))
