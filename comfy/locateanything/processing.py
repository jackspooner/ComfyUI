# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Adapted from nvidia/LocateAnything-3B; NVIDIA License, see LICENSE.
"""Local image-only preparation matching the installed LocateAnything processor."""
import math

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def prepare_image(image, config):
    patch, limit = config['patch_size'], config['in_token_limit']
    mw, mh = config['merge_kernel_size'][1]*patch, config['merge_kernel_size'][0]*patch
    image = image.convert('RGB')
    width, height = image.size
    count = (width//patch)*(height//patch)
    if count > limit:
        scale = math.sqrt(limit/count)
        image = image.resize((int(width*scale), int(height*scale)), Image.Resampling.BICUBIC)
    width, height = math.ceil(image.width/mw)*mw, math.ceil(image.height/mh)*mh
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    grid = (height//patch, width//patch)
    if max(grid) >= 512:
        raise ValueError('Exceed pos emb')
    pixels = TF.normalize(TF.to_tensor(image), config['image_mean'], config['image_std'])
    pixels = pixels.reshape(3, grid[0], patch, grid[1], patch).permute(1, 3, 0, 2, 4).reshape(-1, 3, patch, patch)
    return pixels, grid


def prepare_prompt(tokenizer, prompt, grid):
    count = grid[0]*grid[1]//4
    text = ('<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n'
            '<|im_start|>user\n<image 1><img>'+'<IMG_CONTEXT>'*count+'</img>'+prompt+
            '<|im_end|>\n<|im_start|>assistant\n')
    return torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)
