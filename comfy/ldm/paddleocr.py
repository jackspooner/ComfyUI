"""PP-OCRv6 inference, adapted from PaddlePaddle/Hugging Face Transformers.

Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
"""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def run_layers(layers, x):
    queue = comfy.model_prefetch.make_prefetch_queue(list(layers), x.device, {'prefetch_dynamic_vbars': True})
    for layer in layers:
        comfy.model_management.throw_exception_if_processing_interrupted()
        def core():
            nonlocal x
            x = layer(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
    return x


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, operations, stride=1, groups=1, activation=None, **kwargs):
        super().__init__()
        padding = tuple(k // 2 for k in kernel) if isinstance(kernel, (list, tuple)) else (kernel - 1) // 2
        self.convolution = operations.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, groups=groups, bias=False, **kwargs)
        self.normalization = operations.BatchNorm2d(out_channels, **kwargs)
        self.activation = {'relu': nn.ReLU, 'silu': nn.SiLU, None: nn.Identity}[activation]()

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, operations, reduction=4, **kwargs):
        super().__init__()
        self.convolutions = nn.ModuleList([
            operations.Conv2d(channels, channels // reduction, 1, **kwargs), nn.ReLU(),
            operations.Conv2d(channels // reduction, channels, 1, **kwargs), nn.Hardsigmoid(),
        ])

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        for layer in self.convolutions:
            scale = layer(scale)
        return x * scale


class LargeStem(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        _, mid, out = config['stem_channels']
        strides = config.get('stem_strides', [2, 1, 1, 2, 1])
        for name, cin, cout, kernel, stride in [
            ('stem1', 3, mid, 3, strides[0]), ('stem2a', mid, mid // 2, 2, strides[1]),
            ('stem2b', mid // 2, mid, 2, strides[2]), ('stem3', mid * 2, mid, 3, strides[3]),
            ('stem4', mid, out, 1, strides[4]),
        ]:
            setattr(self, name, ConvNorm(cin, cout, kernel, operations, stride=stride, activation='relu', **kwargs))

    def forward(self, x):
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        branch = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat([F.max_pool2d(x, 2, stride=1, ceil_mode=True), branch], dim=1)))


class LCBlock(nn.Module):
    def __init__(self, spec, operations, **kwargs):
        super().__init__()
        kernel, cin, cout, stride, se = spec
        self.has_residual = cin == cout and stride == 1
        if self.has_residual:
            self.token_conv = operations.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, groups=cin, **kwargs)
        else:
            self.token_conv = ConvNorm(cin, cin, kernel, operations, stride=stride, groups=cin, **kwargs)
        self.token_squeeze_excitation = SqueezeExcitation(cin, operations, **kwargs) if se else nn.Identity()
        self.channel_conv1 = ConvNorm(cin, cin * 2, 1, operations, **kwargs)
        self.channel_conv2 = ConvNorm(cin * 2, cout, 1, operations, **kwargs)

    def forward(self, x):
        x = self.token_squeeze_excitation(self.token_conv(x))
        out = self.channel_conv2(F.gelu(self.channel_conv1(x)))
        return x + out if self.has_residual else out


class LCStage(nn.Module):
    def __init__(self, specs, operations, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(LCBlock(spec, operations, **kwargs) for spec in specs)

    def forward(self, x):
        return run_layers(self.blocks, x)


class LCEncoder(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        self.convolution = LargeStem(config, operations, **kwargs)
        self.blocks = nn.ModuleList(LCStage(specs, operations, **kwargs) for specs in config['block_configs'])

    def forward(self, x):
        x = self.convolution(x)
        features = []
        for stage in self.blocks:
            x = stage(x)
            features.append(x)
        return features


class LCBackbone(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        self.encoder = LCEncoder(config, operations, **kwargs)

    def forward(self, x):
        return self.encoder(x)


class DetConvNorm(nn.Module):
    def __init__(self, cin, cout, kernel, operations, stride=1, padding=1, bias=False, transpose=False, **kwargs):
        super().__init__()
        if transpose:
            self.convolution = operations.ConvTranspose2d(cin, cout, kernel, stride=stride, **kwargs)
        else:
            self.convolution = operations.Conv2d(cin, cout, kernel, stride=stride, padding=padding, bias=bias, **kwargs)
        self.norm = operations.BatchNorm2d(cout, **kwargs)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        return self.act_fn(self.norm(self.convolution(x)))


class IntraclassBlock(nn.Module):
    def __init__(self, config, channels, reduction, operations, **kwargs):
        super().__init__()
        reduced = channels // reduction
        self.conv_reduce_channel = operations.Conv2d(channels, reduced, *config['reduce_channel'], **kwargs)
        self.branches = [
            ['symmetric_conv_long_' + ratio, 'vertical_long_to_small_conv_' + ratio, 'horizontal_small_to_long_conv_' + ratio]
            for ratio in ('longratio', 'midratio', 'shortratio')
        ]
        for names in self.branches:
            for name in names:
                setattr(self, name, operations.Conv2d(reduced, reduced, *config[name], **kwargs))
        kernel, stride, padding = config['return_channel']
        self.conv_final = DetConvNorm(reduced, channels, kernel, operations, stride=stride, padding=padding, bias=True, **kwargs)

    def forward(self, x):
        residual = x
        x = self.conv_reduce_channel(x)
        for names in self.branches:
            x = getattr(self, names[0])(x) + getattr(self, names[1])(x) + getattr(self, names[2])(x)
        return residual + self.conv_final(x)


class DetectionNeck(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        channels = config['neck_out_channels']
        inputs = [stage[-1][2] for stage in config['backbone_config']['block_configs']]
        self.input_channel_adjustment_convolution = nn.ModuleList(operations.Conv2d(c, channels, 1, bias=False, **kwargs) for c in inputs)
        self.input_feature_projection_convolution = nn.ModuleList(operations.Conv2d(channels, channels // 4, 9, padding=4, **kwargs) for _ in inputs)
        self.path_aggregation_head_convolution = nn.ModuleList(operations.Conv2d(channels // 4, channels // 4, 3, stride=2, padding=1, bias=False, **kwargs) for _ in inputs[1:])
        self.path_aggregation_lateral_convolution = nn.ModuleList(operations.Conv2d(channels // 4, channels // 4, 9, padding=4, **kwargs) for _ in inputs)
        self.intraclass_blocks = nn.ModuleList(IntraclassBlock(config['intraclass_block_config'], channels // 4, config['reduce_factor'], operations, **kwargs) for _ in inputs)

    def forward(self, features):
        adjusted = [conv(x) for conv, x in zip(self.input_channel_adjustment_convolution, features)]
        top_down = list(adjusted)
        for i in range(len(top_down) - 2, -1, -1):
            top_down[i] = adjusted[i] + F.interpolate(top_down[i + 1], scale_factor=2, mode='nearest')
        projected = [conv(x) for conv, x in zip(self.input_feature_projection_convolution, top_down)]
        bottom_up = [projected[0]]
        for i, conv in enumerate(self.path_aggregation_head_convolution, 1):
            bottom_up.append(projected[i] + conv(bottom_up[-1]))
        refined = []
        for i, (conv, block, x) in enumerate(zip(self.path_aggregation_lateral_convolution, self.intraclass_blocks, bottom_up)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = block(conv(x))
            refined.append(F.interpolate(x, scale_factor=2 ** i, mode='nearest') if i else x)
        return torch.cat(refined[::-1], dim=1)


class DetectionHead(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        c = config['neck_out_channels']
        k = config['kernel_list']
        self.conv_down = DetConvNorm(c, c // 4, k[0], operations, padding=k[0] // 2, **kwargs)
        self.conv_up = DetConvNorm(c // 4, c // 4, k[1], operations, stride=2, transpose=True, **kwargs)
        self.conv_final = operations.ConvTranspose2d(c // 4, 1, k[2], stride=2, **kwargs)

    def forward(self, x):
        return self.conv_final(self.conv_up(self.conv_down(x))).sigmoid()


class Detector(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = LCBackbone(config['backbone_config'], operations, **kwargs)
        self.model.neck = DetectionNeck(config, operations, **kwargs)
        self.head = DetectionHead(config, operations, **kwargs)

    def forward(self, x):
        return self.head(self.model.neck(self.model.backbone(x)))


class RecognitionAttention(nn.Module):
    def __init__(self, dim, heads, operations, **kwargs):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.qkv = operations.Linear(dim, dim * 3, **kwargs)
        self.projection = operations.Linear(dim, dim, **kwargs)

    def forward(self, x):
        q, k, v = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.heads, -1).unbind(2)
        q, k, v = [AttentionTensorContainer(t.transpose(1, 2)) for t in (q, k, v)]
        return self.projection(optimized_attention(q, k, v, self.heads, skip_reshape=True, preferred_attention=self.comfy_attention))


class RecognitionMLP(nn.Module):
    def __init__(self, dim, hidden, operations, **kwargs):
        super().__init__()
        self.fc1 = operations.Linear(dim, hidden, **kwargs)
        self.fc2 = operations.Linear(hidden, dim, **kwargs)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class RecognitionBlock(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        dim = config['hidden_size']
        eps = config.get('layer_norm_eps', 1e-6)
        self.layer_norm1 = operations.LayerNorm(dim, eps=eps, **kwargs)
        self.self_attn = RecognitionAttention(dim, config.get('num_attention_heads', 8), operations, **kwargs)
        self.layer_norm2 = operations.LayerNorm(dim, eps=eps, **kwargs)
        self.mlp = RecognitionMLP(dim, int(dim * config['mlp_ratio']), operations, **kwargs)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class RecognitionEncoder(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        cin = config['backbone_config']['block_configs'][-1][-1][2]
        dim = config['hidden_size']
        self.conv_block = nn.ModuleList([
            ConvNorm(cin, dim, (1, 1), operations, activation='silu', **kwargs),
            ConvNorm(cin, dim, (1, 1), operations, activation='silu', **kwargs),
            ConvNorm(dim, dim, config['conv_kernel_size'], operations, groups=dim, activation='silu', **kwargs),
        ])
        self.svtr_block = nn.ModuleList(RecognitionBlock(config, operations, **kwargs) for _ in range(config['depth']))
        self.norm = operations.LayerNorm(dim, eps=config.get('layer_norm_eps', 1e-6), **kwargs)

    def forward(self, x):
        residual = self.conv_block[0](x)
        x = self.conv_block[1](x)
        x = x + self.conv_block[2](x)
        b, c, h, w = x.shape
        x = self.norm(run_layers(self.svtr_block, x.flatten(2).transpose(1, 2)))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2) + residual
        return x.squeeze(2).transpose(1, 2)


class RecognitionHead(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        self.encoder = RecognitionEncoder(config, operations, **kwargs)
        self.head = operations.Linear(config['hidden_size'], config['head_out_channels'], **kwargs)

    def forward(self, x):
        return self.head(self.encoder(x)).softmax(dim=2)


class Recognizer(nn.Module):
    def __init__(self, config, operations, **kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = LCBackbone(config['backbone_config'], operations, **kwargs)
        self.head = RecognitionHead(config, operations, **kwargs)

    def forward(self, x):
        return self.head(F.avg_pool2d(self.model.backbone(x)[-1], (3, 2)))
