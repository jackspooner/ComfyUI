"""Pinned local components and native patchers; no runtime source imports or downloads."""
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.storage
import comfy.utils
from comfy.ldm.modules.attention import ComfyAttention, get_attention_function
from .model import ViiTorVoice
from comfy.audio_encoders.wav2vec2_bert import Wav2Vec2Bert
from comfy.ldm.audio.dualcodec import DualCodec

REVISION = 'c225ea3e999c5bb0092c41f01e1ab0f2576fcf3e'
IMPLEMENTATION = '40359993534de0980eefff2d5a25bddc30f6bf58'


@dataclass
class Component:
    patcher: comfy.model_patcher.ModelPatcher
    config: dict
    metadata: dict
    tokenizer: object = None
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    silence: torch.Tensor | None = None

    def options(self):
        return {**self.patcher.model_options.get('transformer_options', {}),
                'prefetch_dynamic_vbars': self.patcher.is_dynamic()}


def read_package(path, kind):
    path = Path(path).resolve()
    manifest_path = path.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise ValueError('VIITOR_NATIVE_PACKAGE_MISSING: preserve the converter manifest beside '+str(path))
    state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    manifest = json.loads(manifest_path.read_text())
    if (not metadata or metadata.get('format') != 'viitorvoice_'+kind or
            metadata.get('source_revision') != REVISION or metadata.get('implementation_revision') != IMPLEMENTATION or
            metadata.get('conversion_version') != '1' or manifest.get('metadata') != metadata):
        raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: prepare the pinned local package with scripts/convert_viitorvoice.py')
    return state, metadata


def load(path, kind, cpu=False, attention='pytorch', disable_dynamic=False):
    if kind not in ('language', 'semantic', 'codec'):
        raise ValueError('Unknown ViiTor component: '+str(kind))
    state, metadata = read_package(path, kind)
    config = json.loads(metadata.get('config', '{}'))
    fingerprints = dict(language='7b2cfffa203ca86ca46ccabd0969804cb6599706e81cd299f51669400655e2a8',
        semantic='9d5fc865ff2cafb5bb7cfbedad0c42db956a0f5e9819872e3ae53705194ad10d')
    if kind in fingerprints and hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != fingerprints[kind]:
        raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: component configuration differs from the pinned release')
    if kind == 'semantic' and hashlib.sha256(json.dumps(json.loads(metadata['processor']), sort_keys=True, separators=(',', ':')).encode()).hexdigest() != 'b7e1a0f992c5f0e72504ae9e929554ce87258547b06d574878d1896f4071dbd2':
        raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: feature processor differs from the pinned release')
    tokenizer, mean, std, silence = None, None, None, None
    if kind == 'language':
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(state.pop('tokenizer_json').numpy().tobytes().decode()),
            bos_token='<|cos_bos|>', eos_token='<|cos_eos|>', pad_token='<|cos_eos|>', unk_token='<unk>', mask_token='<mask>',
            clean_up_tokenization_spaces=False)
        model_class = ViiTorVoice
        if len(tokenizer) != 52697:
            raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: unexpected text tokenizer vocabulary')
    elif kind == 'semantic':
        mean, std = state.pop('semantic_mean'), state.pop('semantic_std')
        if mean.shape != (1024,) or std.shape != (1024,) or not torch.all(std > 0):
            raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: invalid semantic normalization statistics')
        model_class = Wav2Vec2Bert
    else:
        model_class = DualCodec
        silence = state.pop('silence_codes')
        if silence.shape != (12, 49) or silence.dtype != torch.int64:
            raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: missing pinned silence codes')
    arguments = {} if kind == 'codec' else dict(config=config)
    expected = model_class(**arguments, dtype=torch.float32, device='meta', operations=torch.nn).state_dict()
    if state.keys() != expected.keys() or any(state[key].shape != value.shape or state[key].dtype != value.dtype for key, value in expected.items()):
        raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: complete FP32 component inventory required')
    device = torch.device('cpu') if cpu else mm.get_torch_device()
    offload = torch.device('cpu') if cpu else mm.unet_offload_device()
    model = model_class(**arguments, dtype=torch.float32, device=offload, operations=comfy.ops.manual_cast).eval()
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=device, offload_device=offload, fast_disk=comfy.storage.state_dict_fast_disk(state))
    patcher.set_model_compute_dtype(torch.float32)
    model.load_state_dict(state, strict=True, assign=patcher.is_dynamic())
    function = get_attention_function(attention)
    for name, module in model.named_modules():
        if isinstance(module, ComfyAttention):
            patcher.add_object_patch(name+'.function', function)
    mm.archive_model_dtypes(model)
    patcher.cached_patcher_init = (load_patcher, (str(path), kind, cpu, attention))
    return Component(patcher, config, metadata, tokenizer, mean, std, silence)


def load_patcher(path, kind, cpu=False, attention='pytorch', disable_dynamic=False):
    return load(path, kind, cpu, attention, disable_dynamic).patcher
