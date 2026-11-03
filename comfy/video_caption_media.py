"""Local tensor preparation for audio-visual captioning; no network or model loading."""
import math

import torch
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor

import comfy.audio
from comfy.text_encoders.qwen_vl import image_size
from comfy.text_generation_media import sample_video


def prepare_video(images, source_fps, fps, patch_size, max_pixels):
    frames, timestamps = sample_video(images, source_fps, fps)
    if frames.shape[-1] not in (3, 4):
        raise ValueError('Video frames must be RGB or RGBA')
    if len(frames) % 2:
        frames = torch.cat((frames, frames[-1:]))
        timestamps = [*timestamps, timestamps[-1]]
    height, width = image_size(*frames.shape[1:3], patch_size=patch_size, min_pixels=784, max_pixels=max_pixels)
    frames = F.interpolate(frames[..., :3].permute(0, 3, 1, 2).float(), (height, width), mode='bicubic', align_corners=False, antialias=True)
    mean, std = ([0.5] * 3, [0.5] * 3) if patch_size == 16 else ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    frames = (frames - frames.new_tensor(mean).view(1, 3, 1, 1)) / frames.new_tensor(std).view(1, 3, 1, 1)
    time, grid_h, grid_w = len(frames) // 2, height // patch_size, width // patch_size
    patches = frames.reshape(time, 2, 3, grid_h // 2, 2, patch_size, grid_w // 2, 2, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(time * grid_h * grid_w, 3 * 2 * patch_size ** 2)
    return patches, torch.tensor([[time, grid_h, grid_w]]), [(a + b) / 2 for a, b in zip(timestamps[::2], timestamps[1::2])]


def prepare_audio(audio):
    if audio is None:
        return None
    waveform = audio['waveform']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[-1] == 0:
        raise ValueError('Captioning requires one nonempty audio clip')
    waveform = comfy.audio.resample(waveform.float().mean(1), audio['sample_rate'], 16000)[0].cpu()
    waveform = F.pad(waveform, (0, max(0, 16000 - waveform.numel())))
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30)
    return torch.stack([extractor(chunk.numpy(), sampling_rate=16000, return_tensors='pt').input_features[0]
                        for chunk in waveform.split(480000)])


def media_prompt(tokenizer, family, prompt, grid, timestamps, audio_count, fps):
    time, height, width = grid[0].tolist()
    per_frame = height * width // 4
    # Each padded Whisper chunk emits 60 half-second audio tokens. The last
    # visual block retains the remainder, matching upstream's interleaving.
    audio_counts = [math.floor(i * 4 / fps) for i in range(time)]
    audio_counts = [min(audio_count, count) for count in audio_counts]
    counts = [b - a for a, b in zip(audio_counts, [*audio_counts[1:], audio_count])]
    blocks = []
    for i in range(time):
        content = '<|video_pad|>' * per_frame + '<|audio_pad|>' * counts[i]
        blocks.append(f'<{timestamps[i]:.1f} seconds><|vision_start|>{content}<|vision_end|>' if family == 'salmonn2_pro_8b' else content)
    content = ''.join(blocks)
    if family == 'padcaptioner_3b':
        content = '<|vision_start|>' + content + '<|vision_end|>'
    text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + content + '\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False).ids]), counts
