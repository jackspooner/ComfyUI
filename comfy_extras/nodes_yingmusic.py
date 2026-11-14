import importlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
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
        }, 'optional': {'lyric_assets': ('STRING', {'default': '', 'tooltip': 'Explicit converted lyric assets; empty selects the configured text-encoder roots.'})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/yingmusic"
    DESCRIPTION = "Generate or replace singing from a voice reference, melody reference and Chinese/English lyrics. Use native YingMusic diffusion, melody encoder and VAE weights. Melody audio determines the duration."

    def generate(self, model, vae, audio_encoder, voice_reference, melody_reference, reference_lyrics,
                 target_lyrics, seed, steps, guidance, time_shift, reference_silence, lyric_assets=''):
        tokenizer_module = importlib.import_module("comfy.text_encoders.yingmusic")
        tokenizer = tokenizer_module.YingMusicTokenizer(Path(lyric_assets) if lyric_assets else tokenizer_path())
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


class YingMusicPackageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'package_root': ('STRING', {'default': ''}), 'device': (['auto', 'cpu', 'cuda:0'],)}}

    RETURN_TYPES = ('MODEL', 'VAE', 'AUDIO_ENCODER', 'STRING')
    RETURN_NAMES = ('model', 'vae', 'audio_encoder', 'lyric_assets')
    FUNCTION = 'load'
    CATEGORY = 'model/loaders'
    DESCRIPTION = 'Bind a complete offline converted Singer Plus package using existing native component loaders and explicit device placement.'

    def load(self, package_root, device='auto'):
        import comfy.sd
        import comfy.utils
        import comfy.audio_encoders.audio_encoders
        root = Path(package_root)
        try:
            manifest = json.loads((root/'yingmusic_singer_plus.manifest.json').read_text())
        except (OSError, ValueError) as error:
            raise ValueError('YING_NATIVE_PACKAGE_MISSING: prepare the sibling Singer Plus package with scripts/convert_yingmusic.py') from error
        if (manifest.get('source_sha256') != 'b32dc225a6cfff37f2252b7d3b60cbf536f656de349eba6d5b773aa5f5b780e4' or
                manifest.get('implementation_revision') != 'baa409c2e7e5e775f09b4e92a220808f4827d2cc'):
            raise ValueError('YING_NATIVE_PACKAGE_INVALID: expected the exact pinned Singer Plus package')
        paths = {kind: root/kind/'yingmusic_singer_plus.safetensors' for kind in ('diffusion_models', 'vae', 'audio_encoders')}
        for path in paths.values():
            if not path.is_file() or str(path.relative_to(root)) not in manifest['files']:
                raise ValueError('YING_NATIVE_PACKAGE_MISSING: complete all converted Singer Plus components')
            with safe_open(str(path), framework='pt', device='cpu') as saved:
                metadata = saved.metadata() or {}
                if metadata.get('source_sha256') != manifest['source_sha256'] or len(saved.keys()) != manifest['files'][str(path.relative_to(root))]['tensors']:
                    raise ValueError('YING_NATIVE_PACKAGE_INVALID: component identity or complete inventory differs from its manifest')
        selected = None if device == 'auto' else torch.device(device)
        if selected is not None and selected.type == 'cuda' and selected != comfy.model_management.get_torch_device():
            raise ValueError('YingMusic package device must match the selected native GPU or CPU')
        options = {} if selected is None else {'load_device': selected, 'offload_device': torch.device('cpu')}
        model = comfy.sd.load_diffusion_model(str(paths['diffusion_models']), model_options=options)
        vae_path = str(paths['vae'])
        vae_sd, metadata = comfy.utils.load_torch_file(vae_path, safe_load=True, return_metadata=True)
        vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata, device=selected)
        vae.throw_exception_if_invalid()
        vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (vae_path, metadata, selected))
        encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(
            comfy.utils.load_torch_file(str(paths['audio_encoders']), safe_load=True), device=selected)
        return model, vae, encoder, str(root/'text_encoders/yingmusic_singer_plus')


class YingMusicSeparatorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'separator_name': (folder_paths.get_filename_list('audio_encoders'),)},
                'optional': {'device': (['auto', 'cpu'],)}}

    RETURN_TYPES = ('YINGMUSIC_SEPARATOR',)
    FUNCTION = 'load'
    CATEGORY = 'model/loaders'
    DESCRIPTION = 'Load the pinned converted YingMusic MelBandRoformer with native model patchers. Prepare offline; no downloads.'

    def load(self, separator_name, device='auto'):
        import comfy.yingmusic_separator
        path = folder_paths.get_full_path_or_raise('audio_encoders', separator_name)
        return (comfy.yingmusic_separator.runtime.load_component(path, 'yingmusic_roformer', torch.float32, cpu=device == 'cpu'),)


class YingMusicSeparateVocals:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'separator': ('YINGMUSIC_SEPARATOR',), 'audio': ('AUDIO',)}}

    RETURN_TYPES = ('AUDIO', 'AUDIO')
    RETURN_NAMES = ('vocals', 'accompaniment')
    FUNCTION = 'separate'
    CATEGORY = 'audio/yingmusic'
    DESCRIPTION = 'Separate with YingMusic’s exact eight-second overlapping preparation, batch fades and residual accompaniment.'

    def separate(self, separator, audio):
        import comfy.yingmusic_separator
        return comfy.yingmusic_separator.separate(separator, audio)


NODE_CLASS_MAPPINGS = {'YingMusicGenerate': YingMusicGenerate,
    'YingMusicPackageLoader': YingMusicPackageLoader,
    'YingMusicSeparatorLoader': YingMusicSeparatorLoader, 'YingMusicSeparateVocals': YingMusicSeparateVocals}
NODE_DISPLAY_NAME_MAPPINGS = {'YingMusicGenerate': 'YingMusic Generate / Edit Singing',
    'YingMusicSeparatorLoader': 'Load YingMusic Vocal Separator', 'YingMusicSeparateVocals': 'YingMusic Separate Vocals'}
