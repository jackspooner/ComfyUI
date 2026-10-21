import json

import torch
from torch.nn import functional as F
from safetensors import safe_open

import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.unipase.wavlm import DeWavLM
from comfy.ldm.unipase.vocos import Adapter, Vocoder
from comfy.ldm.unipase.postnet import TFGridNet


def load(path):
    with safe_open(path, framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
    if metadata.get("format") != "unipase" or metadata.get("version") != "1":
        raise ValueError("Expected native UniPASE v1 weights; run scripts/convert_unipase.py with the four local released checkpoints.")
    configs = json.loads(metadata["configs"])
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    device, offload = mm.get_torch_device(), mm.unet_offload_device()
    bundle = {}
    for kind, model_class in (("encoder", DeWavLM), ("adapter", Adapter), ("vocoder", Vocoder), ("postnet", TFGridNet)):
        args = dict(dtype=torch.float32, device=offload, operations=comfy.ops.manual_cast)
        model = model_class(**configs[kind], **args) if kind == "postnet" else model_class(configs[kind], **args)
        prefix = kind + "."
        weights = {key[len(prefix):]: state.pop(key) for key in list(state) if key.startswith(prefix)}
        model.load_state_dict(weights, strict=True)
        model.eval()
        mm.archive_model_dtypes(model)
        bundle[kind] = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=offload)
    if state:
        raise ValueError(f"Unexpected UniPASE tensors: {sorted(state)}")
    return bundle


def activate(patcher, frames, recurrent=False):
    # FP32 weights and bounded activations; the native recurrent component must
    # be loaded in full because cuDNN owns its packed LSTM parameter interface.
    memory = 256 * 1024 ** 2 + frames * frames * 16 * 4 + frames * 1024 * 128 * 4
    mm.load_models_gpu([patcher], memory_required=memory, force_full_load=recurrent)
    options = {**patcher.model_options.get("transformer_options", {}), "prefetch_dynamic_vbars": patcher.is_dynamic()}
    return patcher.model, patcher.load_device, options


def segment(bundle, audio, rate, output_rate, enable_plc):
    source = comfy.audio.resample(audio, rate, 16000)
    encoder, device, options = activate(bundle["encoder"], source.shape[-1] // 320)
    acoustic, phonetic = encoder(source.to(device), enable_plc, options)
    adapter, device, options = activate(bundle["adapter"], acoustic.shape[1])
    features = adapter(acoustic.to(device), phonetic.to(device), options)
    del acoustic, phonetic
    vocoder, device, options = activate(bundle["vocoder"], features.shape[-1])
    waveform = vocoder(features.to(device), options)
    del features
    if output_rate > 16000:
        postnet, device, options = activate(bundle["postnet"], waveform.shape[-1] // 256, recurrent=True)
        waveform = postnet(waveform.to(device), 16000, output_rate, options)
    elif output_rate != 16000:
        waveform = comfy.audio.resample(waveform, 16000, output_rate)
    length = audio.shape[-1] * output_rate // rate
    return F.pad(waveform, (0, length - waveform.shape[-1])).cpu()


def segments(length, rate):
    if length <= 8 * rate:
        return [(0, length)]
    result = []
    start = 0
    while start < length:
        end = start + 8 * rate
        if end > length:
            if end - length < 2 * rate:
                result.append((start, length))
            else:
                result[-1] = (result[-1][0], length)
            break
        result.append((start, end))
        start += 4 * rate
    return result


def enhance(bundle, audio, output_sample_rate=0, enable_plc=True):
    waveform, rate = audio["waveform"], audio["sample_rate"]
    output_rate = output_sample_rate or rate
    if waveform.ndim != 3 or waveform.shape[-1] == 0 or rate <= 0 or output_rate <= 0:
        raise ValueError("UniPASE needs nonempty AUDIO [batch, channels, samples] and positive sample rates.")
    source = waveform.float().cpu()
    parts = segments(source.shape[-1], rate)
    result = torch.empty(*source.shape[:-1], source.shape[-1] * output_rate // rate, dtype=torch.float32)
    progress = comfy.utils.ProgressBar(source.shape[0] * source.shape[1] * len(parts))
    for batch in range(source.shape[0]):
        for channel in range(source.shape[1]):
            track = source[batch, channel]
            target = result[batch, channel]
            for i, (start, end) in enumerate(parts):
                mm.throw_exception_if_processing_interrupted()
                restored = segment(bundle, track[None, start:end], rate, output_rate, enable_plc)[0]
                begin = start * output_rate // rate + (2 * output_rate if i else 0)
                finish = (start + 6 * rate) * output_rate // rate if i < len(parts) - 1 else target.shape[-1]
                offset = begin - start * output_rate // rate
                target[begin:finish] = restored[offset:offset + finish - begin]
                progress.update(1)
            # Upstream file inference preserves the recording's peak after stitching.
            target.mul_(track.abs().amax() / (target.abs().amax() + 1e-8))
    return {"waveform": result, "sample_rate": output_rate}
