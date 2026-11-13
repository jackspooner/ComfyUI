# Adapted from ASLP-lab/YingMusic-Singer-Plus, source baa409c2e7e5e775f09b4e92a220808f4827d2cc.
# The bundled MusicSourceSeparationTraining code is MIT licensed (ZFTurbo).
"""Exact YingMusic demix preparation with the existing native MelBandRoformer."""
import math

import torch
from torch.nn import functional as F

import comfy.audio
import comfy.model_management as mm
import comfy.model_prefetch
import comfy.utils
from comfy.soulx_singer import runtime


def validate_metadata(metadata):
    expected = dict(format='yingmusic_roformer', kind='vocals',
        source_revision='baa409c2e7e5e775f09b4e92a220808f4827d2cc',
        model_revision='9b3f444f2bccd77fbc03e32eb7c334fc96040b8d',
        source_sha256='87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e',
        config_sha256='f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926',
        preprocessing='yingmusic_demix_v1')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('YING_NATIVE_PACKAGE_INVALID: prepare the exact pinned YingMusic separator package offline')


def demix(model, waveform, device, options, chunk=352800, overlap=2, batch_size=4):
    """One stereo item; retain the pinned batch-level edge-window decisions."""
    step, border, fade = chunk//overlap, chunk-chunk//overlap, chunk//10
    length = waveform.shape[-1]
    padded = F.pad(waveform, (border, border), mode='reflect') if length > 2*border and border else waveform
    result = torch.zeros(1, *padded.shape, dtype=torch.float32)
    counter = torch.zeros_like(result)
    window = torch.ones(chunk)
    window[:fade] = torch.linspace(0, 1, fade)
    window[-fade:] = torch.linspace(1, 0, fade)
    progress = comfy.utils.ProgressBar(math.ceil(padded.shape[-1]/step))
    pending, locations = [], []
    for start in range(0, padded.shape[-1], step):
        mm.throw_exception_if_processing_interrupted()
        part = padded[:, start:start+chunk]
        count = part.shape[-1]
        part = F.pad(part, (0, chunk-count), mode='reflect' if count > chunk//2 else 'constant')
        pending.append(part)
        locations.append((start, count))
        next_start = start+step
        if len(pending) == batch_size or next_start >= padded.shape[-1]:
            prediction = model(torch.stack(pending).to(device=device, dtype=model.dtype), options).float().cpu()
            edge_window = window.clone()
            if start == 0:
                edge_window[:fade] = 1
            elif next_start >= padded.shape[-1]:
                edge_window[-fade:] = 1
            for value, (position, count) in zip(prediction, locations):
                result[..., position:position+count] += value[..., :count]*edge_window[:count]
                counter[..., position:position+count] += edge_window[:count]
            pending.clear()
            locations.clear()
        progress.update(1)
    result = torch.nan_to_num(result/counter)
    if length > 2*border and border:
        result = result[..., border:-border]
    return result[0]


def separate(component, audio):
    validate_metadata(component.metadata)
    waveform = comfy.audio.resample(audio['waveform'].detach().to(device='cpu', dtype=torch.float32), audio['sample_rate'], 44100)
    if waveform.ndim != 3 or waveform.shape[1] not in (1, 2) or not waveform.shape[-1]:
        raise ValueError('YingMusic separation requires nonempty mono or stereo [batch, channel, sample] audio')
    waveform = waveform.repeat(1, 2, 1) if waveform.shape[1] == 1 else waveform
    if not torch.isfinite(waveform).all():
        raise ValueError('YingMusic separation requires finite audio samples')
    # FP32 baseline preserves native/reference component math. Weights and
    # activations remain under the normal patcher admission and offload owner.
    model, device, _ = runtime.activate(component, memory=2*1024**3)
    try:
        with torch.inference_mode():
            vocals = torch.stack([demix(model, item, device, runtime.transformer_options(component)) for item in waveform])
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
    return ({'waveform': vocals, 'sample_rate': 44100},
            {'waveform': waveform-vocals, 'sample_rate': 44100})
