"""Local packages and patcher-owned LocateAnything execution; no source imports."""
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.console_progress import ConsoleProgress, advance, phase
from comfy.ldm.modules.attention import ComfyAttention, get_attention_function
from .model import Language, Vision
from .processing import prepare_image, prepare_prompt
from .sampling import generate

REVISION = 'c32291ca5e996f5a7a485845b4f57a233936bba0'


@dataclass
class Component:
    patcher: comfy.model_patcher.ModelPatcher
    config: dict
    processor: dict
    tokenizer: Tokenizer | None
    dtype: torch.dtype


def load(path, kind, cpu=False, attention='pytorch'):
    if kind not in ('vision','language'):
        raise ValueError('Unknown LocateAnything component: '+str(kind))
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        metadata = saved.metadata() or {}
    if metadata.get('format') != 'locateanything_'+kind or metadata.get('source_revision') != REVISION:
        raise ValueError('LOCATE_NATIVE_PACKAGE_INVALID: prepare the pinned package with scripts/convert_locateanything.py offline')
    manifest_path = Path(path).with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise ValueError('LOCATE_NATIVE_PACKAGE_MISSING: retain the converter manifest beside each native component')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('metadata') != metadata:
        raise ValueError('LOCATE_NATIVE_PACKAGE_INVALID: component and converter manifest identities differ')
    config, processor = json.loads(metadata['config']), json.loads(metadata['processor'])
    for value, digest in ((config,'b1efe18c470e5afc335ac227c4f2ad3881dd25424636a3c770647fa8ea4f507f'),
                          (processor,'663e87a73e2d3291a3d3308bba72b7bdac587915e659bbc27bb8b6fe06b0f63a')):
        if hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest() != digest:
            raise ValueError('LOCATE_NATIVE_PACKAGE_INVALID: checkpoint configuration or image processor differs from the pinned release')
    device = torch.device('cpu') if cpu else mm.get_torch_device()
    offload = torch.device('cpu') if cpu else mm.unet_offload_device()
    dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    tokenizer = Tokenizer.from_str(state.pop('tokenizer_json').numpy().tobytes().decode()) if kind == 'language' else None
    model = (Vision if kind == 'vision' else Language)(config, dtype, offload, comfy.ops.manual_cast)
    model.load_state_dict(state, strict=True)
    model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=offload)
    function = get_attention_function(attention)
    for name, module in model.named_modules():
        if isinstance(module, ComfyAttention):
            patcher.add_object_patch(name+'.function', function)
    mm.archive_model_dtypes(model)
    return Component(patcher, config, processor, tokenizer, dtype)


def options(component):
    return {**component.patcher.model_options.get('transformer_options', {}),
            'prefetch_dynamic_vbars': component.patcher.is_dynamic()}


def locate(vision, language, image, prompt, mode, max_new_tokens, seed):
    if vision.config != language.config or vision.processor != language.processor:
        raise ValueError('LOCATE_NATIVE_PACKAGE_INVALID: vision and language must belong to the same pinned package')
    if mode not in ('hybrid', 'fast', 'slow'):
        raise ValueError('Unsupported LocateAnything generation_mode: '+str(mode))
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError('seed must be an integer between 0 and 18446744073709551615')
    if not 1 <= max_new_tokens <= 8192:
        raise ValueError('max_new_tokens must be between 1 and 8192')
    mm.throw_exception_if_processing_interrupted()
    pixels, grid = prepare_image(image, vision.processor)
    ids = prepare_prompt(language.tokenizer, prompt, grid)
    tokens, dim = pixels.shape[0], vision.config['vision_config']['hidden_size']
    # Reserve dense vision scores and block activations without assuming a
    # memory-efficient attention kernel. No second residency owner exists.
    memory = 2*16*tokens*tokens*4+12*tokens*dim*4
    try:
        with ConsoleProgress('LocateAnything', 'image encoding', unit='tokens'):
            mm.load_models_gpu([vision.patcher], memory_required=memory)
            visual = vision.patcher.model(pixels.to(device=vision.patcher.load_device, dtype=vision.dtype), grid, options(vision)).cpu()
            values = language.config['text_config']
            capacity = min(16384, ids.shape[1]+max_new_tokens)+6
            element_bytes = 4 if language.dtype == torch.float32 else 2
            cache_bytes = 2*values['num_hidden_layers']*values['num_key_value_heads']*capacity*(values['hidden_size']//values['num_attention_heads'])*element_bytes
            prefix = ids.shape[1]+6
            workspaces = 2*values['num_attention_heads']*prefix*prefix*4+12*prefix*values['hidden_size']*element_bytes
            mm.load_models_gpu([language.patcher], memory_required=cache_bytes+workspaces+512*1024**2)
            device = language.patcher.load_device
            phase('decoding')
            return generate(language.patcher.model, ids.to(device), visual.to(device=device, dtype=language.dtype),
                language.tokenizer, max_new_tokens, seed, mode, options(language), advance)
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
