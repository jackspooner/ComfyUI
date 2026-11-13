"""Local artifact loading and audio preparation shared by native ASR models."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.utils


def waveform(audio):
    value = audio['waveform']
    if value.ndim != 3 or value.shape[0] != 1 or min(value.shape[1:]) == 0:
        raise ValueError('Transcription requires one nonempty AUDIO clip')
    value = value.detach().float().cpu().mean(1)
    if not torch.isfinite(value).all():
        raise ValueError('Transcription audio contains non-finite samples')
    return comfy.audio.resample(value, int(audio['sample_rate']), 16000)[0]


def package(path, architecture):
    path = Path(path)
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        metadata = saved.metadata() or {}
    if metadata.get('architecture') != architecture or metadata.get('asr_package_version') != '1':
        raise ValueError(f'Expected a converted native {architecture} package; run scripts/convert_asr.py offline')
    receipt = json.loads(path.with_suffix('.manifest.json').read_text())
    if receipt.get('metadata') != metadata:
        raise ValueError('Native transcription package does not match its manifest')
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    expected = {entry['destination'] for entry in receipt['tensors'].values()} | {'tokenizer_json'}
    if set(state) != expected:
        raise ValueError('Native transcription tensor inventory differs from its manifest')
    return metadata, state


def patcher(model, state, device):
    fast_disk = comfy.storage.state_dict_fast_disk(state)
    model.load_state_dict(state, strict=True, assign=True)
    mm.archive_model_dtypes(model)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=device,
                                               offload_device=mm.unet_offload_device(), fast_disk=fast_disk)


def take_component(state, prefix):
    return {key.removeprefix(prefix): state.pop(key) for key in list(state) if key.startswith(prefix)}
