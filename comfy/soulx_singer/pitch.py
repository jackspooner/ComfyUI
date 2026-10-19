"""RMVPE inference architecture, adapted from Dream-High/RMVPE via SoulX-Singer.

CPU execution retains the native PyTorch bidirectional GRU.
"""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.audio
import comfy.model_management
import comfy.ops


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            device="meta",
        ).to_empty(device="cpu")

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            comfy.ops.manual_cast.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            comfy.ops.manual_cast.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            comfy.ops.manual_cast.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            comfy.ops.manual_cast.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = comfy.ops.manual_cast.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = comfy.ops.manual_cast.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            comfy.ops.manual_cast.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            comfy.ops.manual_cast.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = comfy.ops.manual_cast.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                comfy.ops.manual_cast.Linear(512, 360),
                nn.Identity(),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                comfy.ops.manual_cast.Linear(3 * 128, 360),
                nn.Identity(),
                nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class PitchExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2))
        self.register_buffer("mel_basis", torch.empty(128, 513))

    def forward(self, audio, sample_rate, threshold=0.03, frame_step=2):
        audio = comfy.audio.resample(audio, sample_rate, 16000)
        contours = []
        # Context overlap bounds activation memory without losing the original 10 ms grid.
        for start in range(0, audio.shape[-1], 16000 * 20):
            comfy.model_management.throw_exception_if_processing_interrupted()
            left = max(0, start - 16000)
            right = min(audio.shape[-1], start + 16000 * 21)
            chunk = audio[..., left:right]
            chunk = F.pad(chunk, (0, max(0, 513 - chunk.shape[-1])))
            window = torch.hann_window(1024, device=chunk.device, dtype=chunk.dtype)
            spec = torch.stft(chunk, 1024, 160, window=window, return_complex=True).abs()
            mel = (self.mel_basis @ spec).clamp(min=1e-5).log()
            frames = mel.shape[-1]
            salience = self.model(F.pad(mel, (0, (-frames) % 32)))[:, :frames]
            center = salience.argmax(-1)
            indexes = center[..., None] + torch.arange(9, device=salience.device)
            weights = F.pad(salience, (4, 4)).gather(-1, indexes)
            cents = F.pad(torch.arange(360, device=salience.device, dtype=torch.float32) * 20 + 1997.3794084376191, (4, 4))
            average = (weights * cents[indexes]).sum(-1) / weights.sum(-1).clamp(min=1e-12)
            f0 = torch.where(salience.amax(-1) > threshold, 10 * 2 ** (average / 1200), 0)
            offset = (start - left) // 160
            count = min(2000, (audio.shape[-1] - start + 159) // 160)
            contours.append(f0[:, offset:offset + count:frame_step])
        return torch.cat(contours, dim=-1)
