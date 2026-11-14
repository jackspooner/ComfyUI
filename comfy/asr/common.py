"""Local artifact loading and audio preparation shared by native ASR models."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.utils


def validate_audio(audio):
    """Validate one AUDIO clip and return CPU float32 mono without resampling."""
    if not isinstance(audio, Mapping) or 'waveform' not in audio or 'sample_rate' not in audio:
        raise ValueError('Transcription requires AUDIO waveform and sample_rate')
    value = audio['waveform']
    if not isinstance(value, torch.Tensor):
        raise TypeError('Transcription AUDIO waveform must be a tensor')
    if value.ndim != 3 or value.shape[0] != 1 or min(value.shape[1:]) == 0:
        raise ValueError('Transcription requires one nonempty AUDIO clip')
    value = value.detach().float().cpu().mean(1)
    if not torch.isfinite(value).all():
        raise ValueError('Transcription audio contains non-finite samples')
    if int(audio['sample_rate']) <= 0:
        raise ValueError('Transcription sample rate must be positive')
    return value


def waveform(audio):
    return comfy.audio.resample(validate_audio(audio), int(audio['sample_rate']), 16000)[0]


def package(path, architecture):
    path = Path(path)
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        metadata = saved.metadata() or {}
        if metadata.get('architecture') != architecture or metadata.get('asr_package_version') != '1':
            raise ValueError(f'Expected a converted native {architecture} package; run scripts/convert_asr.py offline')
        receipt = json.loads(path.with_suffix('.manifest.json').read_text())
        if receipt.get('metadata') != metadata:
            raise ValueError('Native transcription package does not match its manifest')
        entries = receipt['tensors'].values()
        destinations = [entry['destination'] for entry in entries]
        if len(set(destinations)) != len(destinations) or 'tokenizer_json' in destinations:
            raise ValueError('Native transcription manifest has duplicate or reserved tensor destinations')
        if set(saved.keys()) != set(destinations) | {'tokenizer_json'}:
            raise ValueError('Native transcription tensor inventory differs from its manifest')
        for entry in entries:
            tensor = saved.get_slice(entry['destination'])
            if entry['dtype'] not in ('BF16', 'F16', 'F32') or tensor.get_dtype() != entry['dtype'] or tensor.get_shape() != entry['shape']:
                raise ValueError(f'Native transcription tensor shape/dtype differs from its manifest: {entry["destination"]}')
        tokenizer = saved.get_slice('tokenizer_json')
        if tokenizer.get_dtype() != 'U8' or len(tokenizer.get_shape()) != 1 or not tokenizer.get_shape()[0]:
            raise ValueError('Native transcription tokenizer must be a nonempty U8 vector')
        tokenizer_json = saved.get_tensor('tokenizer_json').numpy().tobytes()
        if hashlib.sha256(tokenizer_json).hexdigest() != receipt.get('tokenizer_sha256'):
            raise ValueError('Native transcription tokenizer differs from its manifest')
        Tokenizer.from_str(tokenizer_json.decode('utf-8'))
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    return metadata, state


def patcher(model, state, device):
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    # AIMDO Linear parameters are absent until assignment; dimensions still
    # describe the architecture and must be checked before its lazy loader runs.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prefix = name + '.' if name else ''
            expected[prefix + 'weight'] = (module.out_features, module.in_features)
            if module.weight is None and module.comfy_need_lazy_init_bias:
                expected[prefix + 'bias'] = (module.out_features,)
    if set(state) != set(expected):
        raise ValueError(f'Native transcription architecture tensor inventory mismatch: missing={sorted(set(expected) - set(state))}, extra={sorted(set(state) - set(expected))}')
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape or state[name].dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f'Unsupported native transcription architecture tensor shape/dtype: {name}; expected shape {shape}, got {tuple(state[name].shape)} ({state[name].dtype})')
    fast_disk = comfy.storage.state_dict_fast_disk(state)
    model.load_state_dict(state, strict=True, assign=True)
    mm.archive_model_dtypes(model)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=device,
                                               offload_device=mm.unet_offload_device(), fast_disk=fast_disk)


def take_component(state, prefix):
    return {key.removeprefix(prefix): state.pop(key) for key in list(state) if key.startswith(prefix)}
