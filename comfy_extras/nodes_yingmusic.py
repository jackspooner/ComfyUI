import importlib
from pathlib import Path

import torch
from torch.nn import functional as F

import comfy.audio
import comfy.model_management
import comfy.sample
from comfy.audio_encoders.yingmusic import prepare_melody
import folder_paths
import latent_preview


def tokenizer_path():
    for root in folder_paths.get_folder_paths("text_encoders"):
        path = Path(root) / "yingmusic_singer_plus"
        if (path / "vocab.json").is_file():
            return path
    raise FileNotFoundError("Prepare YingMusic lyric assets with scripts/convert_yingmusic.py under models/text_encoders/yingmusic_singer_plus.")


def prepare_audio(audio, silence):
    waveform = comfy.audio.resample(audio["waveform"], audio["sample_rate"], 44100)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    else:
        waveform = waveform[:, :2]
    return F.pad(waveform, (0, round(silence * 44100)))


class YingMusicGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",), "audio_encoder": ("AUDIO_ENCODER",),
            "voice_reference": ("AUDIO",), "melody_reference": ("AUDIO",),
            "reference_lyrics": ("STRING", {"multiline": True, "default": ""}),
            "target_lyrics": ("STRING", {"multiline": True, "default": ""}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 32, "min": 1, "max": 1000}),
            "guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "time_shift": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 100.0}),
            "reference_silence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 30.0}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/yingmusic"
    DESCRIPTION = "Generate or replace singing from a voice reference, melody reference and Chinese/English lyrics. Use native YingMusic diffusion, melody encoder and VAE weights. Melody audio determines the duration."

    def generate(self, model, vae, audio_encoder, voice_reference, melody_reference, reference_lyrics,
                 target_lyrics, seed, steps, guidance, time_shift, reference_silence):
        tokenizer_module = importlib.import_module("comfy.text_encoders.yingmusic")
        tokenizer = tokenizer_module.YingMusicTokenizer(tokenizer_path())
        reference = prepare_audio(voice_reference, reference_silence)
        melody = prepare_audio(melody_reference, 1.0)
        # Upstream peak-normalizes only the VAE input to -6 dBFS. The melody encoder receives original levels.
        peak = reference.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            reference_latent = vae.encode((reference * (10 ** (-6 / 20)) / peak).movedim(1, -1))
        reference_frames = reference_latent.shape[-1]
        frames = reference_frames + melody.shape[-1] // 2048
        tokens = tokenizer.align(reference_lyrics, target_lyrics, reference_frames, frames)
        midi = prepare_melody(audio_encoder.encode_reference_pair(reference, melody), frames, reference_frames)
        reference_latent = F.pad(reference_latent, (0, frames - reference_frames))
        values = {"ying_tokens": torch.tensor([tokens], dtype=torch.long).expand(reference_latent.shape[0], -1), "ying_reference": reference_latent, "ying_melody": midi}
        positive = [[torch.empty(1, 0, 512), values]]
        negative = [[torch.empty(1, 0, 512), {**values, "ying_unconditioned": True}]]
        latent = torch.zeros(reference_latent.shape, dtype=reference_latent.dtype, device=comfy.model_management.intermediate_device())
        noise = comfy.sample.prepare_noise(latent, seed)
        times = torch.linspace(0, 1, steps + 1)
        sigmas = 1 - time_shift * times / (1 + (time_shift - 1) * times)
        callback = latent_preview.prepare_callback(model, steps)
        samples = comfy.sample.sample(model, noise, steps, guidance + 1, "euler", "simple", positive, negative, latent,
                                      sigmas=sigmas, callback=callback, seed=seed)
        samples = samples[..., reference_frames:-int(44100 / 2048)].clone()
        audio = vae.decode(samples).movedim(-1, 1)
        fade_samples = min(6615, audio.shape[-1])
        fade = (1 + torch.cos(torch.linspace(0, torch.pi, fade_samples, device=audio.device, dtype=audio.dtype))) / 2
        audio[..., -fade_samples:] *= fade
        return ({"waveform": audio, "sample_rate": 44100},)


NODE_CLASS_MAPPINGS = {"YingMusicGenerate": YingMusicGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"YingMusicGenerate": "YingMusic Generate / Edit Singing"}
