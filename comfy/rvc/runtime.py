"""Native RVC managed handles and file conversion. No application or network code."""

import json
from dataclasses import dataclass

import librosa
import numpy as np
from safetensors import safe_open
from scipy import signal
import torch
from torch.nn import functional as F

try:
    import faiss
except ImportError:
    faiss = None


import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.rvc.content import ContentEncoder
from comfy.ldm.rvc.model import Synthesizer
from comfy.soulx_singer.pitch import PitchExtractor


@dataclass
class Component:
    patcher: comfy.model_patcher.ModelPatcher
    metadata: dict


def load(path, kind):
    path = str(path)
    with safe_open(path, framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
    if metadata.get("format") != kind:
        raise ValueError(f"Expected {kind}; prepare local weights with scripts/convert_rvc.py (RMVPE: scripts/convert_soulx_singer.py).")
    cpu = kind == "soulx_rmvpe"
    device = torch.device("cpu") if cpu else mm.get_torch_device()
    offload = torch.device("cpu") if cpu else mm.unet_offload_device()
    if kind == "rvc_voice":
        model = Synthesizer(json.loads(metadata["config"]), metadata["version"], json.loads(metadata["use_f0"]), comfy.ops.manual_cast)
    elif kind == "rvc_contentvec":
        model = ContentEncoder(torch.float32, offload, comfy.ops.manual_cast)
    elif kind == "soulx_rmvpe":
        model = PitchExtractor()
    else:
        raise ValueError(f"Unsupported RVC component {kind}")
    model.load_state_dict(comfy.utils.load_torch_file(path, safe_load=True), strict=True)
    model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=offload)
    mm.archive_model_dtypes(model)
    return Component(patcher, metadata)


def activate(component, frames):
    # FP32 activations, relative attention and vocoder workspace scale with one bounded segment.
    memory = 256 * 1024 ** 2 + frames * frames * 2 * 4 + frames * 512 * 400 * 4
    mm.load_models_gpu([component.patcher], memory_required=memory)
    return component.patcher.model, component.patcher.load_device


def options(component):
    patcher = component.patcher
    return {**patcher.model_options.get("transformer_options", {}), "prefetch_dynamic_vbars": patcher.is_dynamic()}


def pitch_contour(component, waveform):
    model, device = activate(component, waveform.shape[-1] // 160)
    return model(waveform.to(device), 16000, frame_step=1).cpu()


def coarse_f0(f0):
    lo, hi = 1127 * np.log1p(50 / 700), 1127 * np.log1p(1100 / 700)
    mel = 1127 * torch.log1p(f0 / 700)
    mel = torch.where(mel > 0, (mel - lo) * 254 / (hi - lo) + 1, mel)
    return mel.clamp(1, 255).round().long()


def retrieve(features, index, vectors, ratio):
    scores, ids = index.search(features[0].float().cpu().numpy(), 8)
    if np.any(ids < 0):
        raise ValueError("Retrieval index did not return eight neighbors; use a populated matching index or disable retrieval.")
    # Exact duplicate vectors must remain finite instead of upstream's inf/inf.
    weights = np.square(1 / np.maximum(scores, 1e-12))
    weights /= weights.sum(axis=1, keepdims=True)
    result = np.sum(vectors[ids] * weights[..., None], axis=1)
    return features * (1 - ratio) + torch.from_numpy(result).to(features)[None] * ratio


def change_rms(source, target, rate, mix):
    a = librosa.feature.rms(y=source, frame_length=16000, hop_length=8000)
    b = librosa.feature.rms(y=target, frame_length=rate // 2 * 2, hop_length=rate // 2)
    a, b = [F.interpolate(torch.from_numpy(v)[None], size=len(target), mode="linear").flatten() for v in (a, b)]
    return target * (a.pow(1 - mix) * b.clamp(min=1e-6).pow(mix - 1)).numpy()


def convert(voice, content, audio, semitones, speaker, protect, rms_mix, seed, pitch_model=None, index_path=None, index_rate=0.):
    config = json.loads(voice.metadata["config"])
    rate, use_f0, version = config[-1], json.loads(voice.metadata["use_f0"]), voice.metadata["version"]
    if use_f0 and pitch_model is None:
        raise ValueError("This pitch-guided voice requires an RMVPE model.")
    waveform = audio["waveform"]
    if waveform.shape[0] != 1:
        raise ValueError("RVC converts one recording per execution; split AUDIO batches first.")
    waveform = comfy.audio.resample(waveform.float().mean(1), audio["sample_rate"], 16000)[0].cpu().numpy()
    if not waveform.size:
        raise ValueError("RVC needs nonempty source audio.")
    waveform = waveform / max(1., np.abs(waveform).max() / .95)
    b, a = signal.butter(5, 48, btype="high", fs=16000)
    waveform = signal.filtfilt(b, a, waveform, padlen=min(3 * max(len(a), len(b)), len(waveform) - 1)).astype(np.float32)
    index = vectors = None
    if index_path and index_rate != 0:
        if faiss is None:
            raise ImportError("RVC retrieval requires optional faiss-cpu; set index_rate=0 or install it in the ComfyUI runtime.")
        index = faiss.read_index(str(index_path))
        if index.d != (256 if version == "v1" else 768):
            raise ValueError("Retrieval index dimension differs from this voice version.")
        vectors = index.reconstruct_n(0, index.ntotal)
    # Four-second core segments with one-second context bound neural activation memory.
    padded = np.pad(waveform, (16000, 16320), mode="reflect")
    outputs = []
    progress = comfy.utils.ProgressBar((len(waveform) + 63999) // 64000)
    for start in range(0, len(waveform), 64000):
        mm.throw_exception_if_processing_interrupted()
        end = min(len(waveform), start + 64000)
        part = torch.from_numpy(padded[start:end + 32320].copy())[None]
        f0 = pitch_contour(pitch_model, part / max(float(torch.quantile(part.abs(), .999)), 1e-7)) if use_f0 else None
        encoder, device = activate(content, part.shape[-1] // 320)
        features = encoder(part.to(device), version, options(content))
        original = features
        if index is not None:
            features = retrieve(features, index, vectors, index_rate)
        features = F.interpolate(features.transpose(1, 2), scale_factor=2).transpose(1, 2)
        frames = min(part.shape[-1] // 160, features.shape[1])
        features = features[:, :frames]
        if use_f0:
            f0 = f0[:, :frames] * 2 ** (semitones / 12)
            if protect < .5:
                unblended = F.interpolate(original.transpose(1, 2), scale_factor=2).transpose(1, 2)[:, :frames]
                mask = torch.where(f0 >= 1, 1., protect).to(features)[..., None]
                features = features * mask + unblended * (1 - mask)
        model, device = activate(voice, frames)
        generator = torch.Generator(device=device).manual_seed((seed + start) % 2 ** 64)
        f0 = f0.to(device) if use_f0 else None
        pitch = coarse_f0(f0) if use_f0 else None
        converted = model(features.to(device), pitch, f0, torch.tensor([speaker], device=device), generator, options(voice)).float().cpu()[0, 0].numpy()
        outputs.append(converted[rate:rate + round((end - start) * rate / 16000)])
        progress.update(1)
    result = np.concatenate(outputs)
    if rms_mix != 1:
        result = change_rms(waveform, result, rate, rms_mix)
    result = result / max(1., float(np.abs(result).max()) / .99)
    return {"waveform": torch.from_numpy(result.astype(np.float32))[None, None], "sample_rate": rate}
