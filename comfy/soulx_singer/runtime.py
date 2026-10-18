"""ComfyUI-owned loading and execution for SoulX. No upstream application imports."""

import json
import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from safetensors import safe_open

import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.audio_encoders.whisper import AudioEncoder
from comfy.ldm.soulx_singer.model import Singer, coarse_f0
from comfy.ldm.soulx_singer.vocoder import Vocos
from comfy.soulx_singer.metadata import preprocess, values
from comfy.soulx_singer.pitch import PitchExtractor
from comfy.soulx_singer.separation import MelBandRoformer


@dataclass
class Component:
    patcher: comfy.model_patcher.ModelPatcher
    metadata: dict = field(repr=False)
    dtype: torch.dtype = torch.float32


class WhisperContent(AudioEncoder):
    def __init__(self, dtype, device, operations):
        super().__init__(80, 1500, 512, 8, 6, dtype, device, operations)

    def forward(self, x, transformer_options={}):
        x = F.gelu(self.conv2(F.gelu(self.conv1(x)))).transpose(1, 2)
        x = x + comfy.ops.cast_to_input(self.embed_positions.weight, x)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for block in self.layers:
            mm.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="whisper")
            result = block(x)
            if torch.is_grad_enabled():
                x = result
            else:
                x.copy_(result)
            del result
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="whisper")
        comfy.model_prefetch.malloc_graph_end()
        return self.layer_norm(x)


def load_component(path, expected_format, dtype=torch.float32, cpu=False):
    path = str(path)
    with safe_open(path, framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
    if metadata.get("format") != expected_format:
        raise ValueError(f"Expected a converted {expected_format} file: {path}. Run scripts/convert_soulx_singer.py offline.")
    device = torch.device("cpu") if cpu else mm.get_torch_device()
    offload = torch.device("cpu") if cpu else mm.unet_offload_device()
    if expected_format == "soulx_singer":
        config = json.loads(metadata["config"])
        enc, flow = config["model"]["encoder"], config["model"]["flow_matching"]
        model = Singer(svc=metadata["kind"] == "svc", dim=flow["hidden_size"], depth=flow["num_layers"],
                       heads=flow["num_heads"], encoder_dim=enc["text_dim"], encoder_depth=enc["num_layers"],
                       vocab_size=enc["vocab_size"], mel_dim=flow["mel_dim"], dtype=torch.float32, device=offload, operations=comfy.ops.manual_cast)
        model.dtype = dtype
    elif expected_format == "soulx_vocos":
        model = Vocos(dtype=torch.float32, device=offload, operations=comfy.ops.manual_cast)
    elif expected_format == "soulx_whisper":
        model = WhisperContent(torch.float32, offload, comfy.ops.manual_cast)
    elif expected_format == "soulx_rmvpe":
        model = PitchExtractor()
    elif expected_format == "soulx_roformer":
        model = MelBandRoformer(json.loads(metadata["bands"]), stems=2 if metadata["kind"] == "karaoke" else 1,
                               dtype=torch.float32, device=offload, operations=comfy.ops.manual_cast)
        model.dtype = dtype
    state = comfy.utils.load_torch_file(path, safe_load=True)
    # The separator's gather indices are derived from converter-authenticated band geometry.
    if expected_format == "soulx_roformer":
        state.update({"freq_indices": model.freq_indices, "num_bands_per_freq": model.num_bands_per_freq})
    model.load_state_dict(state, strict=True)
    model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=offload)
    mm.archive_model_dtypes(model)
    return Component(patcher, metadata, torch.float32 if cpu else dtype)


def activate(component, memory=0):
    patcher = component.patcher
    mm.load_models_gpu([patcher], memory_required=memory)
    return patcher.model, patcher.load_device, component.dtype


def transformer_options(component):
    patcher = component.patcher
    return {**patcher.model_options.get("transformer_options", {}), "prefetch_dynamic_vbars": patcher.is_dynamic()}


def mono(audio, rate=24000):
    waveform = audio["waveform"]
    if waveform.shape[0] != 1:
        raise ValueError("SoulX expects one reference song per execution; split the AUDIO batch first.")
    return comfy.audio.resample(waveform.float().mean(dim=1), audio["sample_rate"], rate)


def reference_mel(waveform, metadata):
    basis = torch.tensor(json.loads(metadata["mel.model.mel_basis"]), device=waveform.device, dtype=torch.float32)
    window = torch.tensor(json.loads(metadata["mel.model.hann_window"]), device=waveform.device, dtype=torch.float32)
    waveform = F.pad(waveform[:, None], (720, 720), mode="reflect")[:, 0]
    spec = torch.stft(waveform, 1920, 480, window=window, center=False, return_complex=True)
    magnitude = (spec.real.square() + spec.imag.square() + 1e-9).sqrt()
    return ((basis @ magnitude).clamp(min=1e-5).log().transpose(1, 2) + 4.92) / math.sqrt(8.14)


def whisper_features(waveform):
    waveform = comfy.audio.resample(waveform, 24000, 16000)
    frontend = comfy.audio.MelSpectrogram(16000, 400, 160, 80, f_max=8000).to(waveform.device)
    mel = frontend(waveform)[..., :-1].clamp(min=1e-10).log10()
    mel = (torch.maximum(mel, mel.amax() - 8) + 4) / 4
    valid = min(1500, (mel.shape[-1] + 1) // 2)
    return F.pad(mel[..., :3000], (0, max(0, 3000 - mel.shape[-1]))), valid


def content_features(component, waveform, frames):
    model, device, dtype = activate(component, memory=256 * 1024 ** 2)
    mel, valid = whisper_features(waveform.to(device))
    features = model(mel.to(dtype), transformer_options(component))[:, :valid]
    return F.pad(features, (0, 0, 0, max(0, frames - features.shape[1])))[:, :frames].clone()


def sample_flow(model, prompt, cond, target_frames, steps, cfg, seed, progress=None, transformer_options=None):
    generator = torch.Generator(device=cond.device).manual_seed(seed)
    x = torch.randn(1, target_frames, 128, generator=generator, device=cond.device, dtype=cond.dtype)
    # Both the projection bias and condition MLP remain present in upstream's null branch.
    conditioned = model.prepare_conditioning(cond)
    unconditioned = model.diff_estimator.prepare_conditioning(torch.zeros(1, target_frames, model.cond_emb.out_features,
                                                                         device=cond.device, dtype=cond.dtype))
    for step in range(steps):
        mm.throw_exception_if_processing_interrupted()
        time = torch.full((1,), (step + .5) / steps, device=x.device, dtype=x.dtype)
        positive = model(torch.cat((prompt, x), dim=1), time, conditioned, transformer_options or {})[:, prompt.shape[1]:]
        if cfg > 0:
            negative = model(x, time, unconditioned, transformer_options or {})
            guided = positive + cfg * (positive - negative)
            # A constant flow has no defined rescale ratio; preserve its finite guided value.
            std = guided.std()
            rescaled = guided * positive.std() / std.clamp(min=torch.finfo(std.dtype).tiny)
            positive = .75 * rescaled + .25 * guided
        x = x + positive / steps
        if progress:
            progress.update(1)
    return x


def decode(bundle, mel):
    vocoder, device, dtype = activate(bundle["vocoder"], memory=mel.shape[1] * 1024 * 128)
    return {"waveform": vocoder(mel.transpose(1, 2).to(device=device, dtype=dtype), transformer_options(bundle["vocoder"])).float().cpu(), "sample_rate": 24000}


def encode_svs(bundle, prompt_meta, target_meta, control, pitch_shift):
    model, device, dtype = activate(bundle["singer"], memory=512 * 1024 ** 2)
    phones = json.loads(bundle["singer"].metadata["phones"])
    pt, gt = [preprocess(meta, phones) for meta in (prompt_meta, target_meta)]
    offset = len(pt["phoneme"])
    def tensor(name, tail=None):
        return torch.tensor([pt[name] + (gt[name] if tail is None else tail)], device=device, dtype=torch.long)
    pitches = tensor("note_pitch").clone()
    pitches = torch.where(pitches > 0, pitches + pitch_shift, pitches).clamp(0, 255) if control == "score" else torch.zeros_like(pitches)
    f0pt = torch.tensor([pt["f0"] or [0.] * len(pt["mel2note"])], device=device)
    f0gt = torch.tensor([gt["f0"] or [0.] * len(gt["mel2note"])], device=device)
    if control == "melody" and (not pt["f0"] or not gt["f0"]):
        raise ValueError("Melody synthesis needs F0 in both metadata objects. Extract it from the reference AUDIO or use score mode.")
    f0 = torch.cat((coarse_f0(f0pt), coarse_f0(f0gt, pitch_shift)), dim=1) if control == "melody" else torch.zeros(1, len(pt["mel2note"]) + len(gt["mel2note"]), device=device, dtype=torch.long)
    features = model.encode_score(tensor("phoneme"), pitches, tensor("note_type"),
                                  tensor("mel2note", [i + offset for i in gt["mel2note"]]), f0)
    return model, features.to(dtype), len(pt["mel2note"]), len(gt["mel2note"])


def generate_svs(bundle, reference, prompt_meta, target_meta, control, pitch_shift, steps, cfg, seed):
    if bundle["singer"].metadata["kind"] != "svs":
        raise ValueError("Load the SVS checkpoint for lyric synthesis; SVC preserves the source words.")
    model, features, pt_frames, gt_frames = encode_svs(bundle, prompt_meta, target_meta, control, pitch_shift)
    waveform = mono(reference)[..., :pt_frames * 480].to(features.device)
    waveform = F.pad(waveform, (0, max(0, pt_frames * 480 - waveform.shape[-1])))
    prompt = reference_mel(waveform, bundle["singer"].metadata).to(features.dtype)
    mel = sample_flow(model, prompt, features, gt_frames, steps, cfg, seed, comfy.utils.ProgressBar(steps), transformer_options(bundle["singer"]))
    return decode(bundle, mel)


def convert_svc(bundle, reference, source, prompt_pitch, source_pitch, pitch_shift, steps, cfg, seed):
    if bundle["singer"].metadata["kind"] != "svc":
        raise ValueError("Load the SVC checkpoint for singing voice conversion.")
    source = mono(source)
    reference = mono(reference)
    frames = source.shape[-1] // 480
    source_f0 = fit_f0(source_pitch["f0"], frames)
    ref_frames = min(reference.shape[-1] // 480, len(prompt_pitch["f0"]), 1500)
    reference = reference[..., :ref_frames * 480]
    pt_content = content_features(bundle["whisper"], reference, ref_frames).cpu()
    result = torch.zeros(1, 1, source.shape[-1])
    weights = torch.zeros_like(result)
    progress = comfy.utils.ProgressBar(steps * max(1, math.ceil(frames / 1400)))
    for start in range(0, max(1, frames), 1400):
        end = min(frames, start + 1500)
        waveform = source[..., start * 480:end * 480]
        gt_content = content_features(bundle["whisper"], waveform, end - start)
        model, device, dtype = activate(bundle["singer"], memory=512 * 1024 ** 2)
        f0pt = torch.tensor([prompt_pitch["f0"][:ref_frames]], device=device)
        f0gt = torch.tensor([source_f0[start:end]], device=device)
        bins = torch.cat((coarse_f0(f0pt), coarse_f0(f0gt, pitch_shift)), dim=1)
        features = torch.cat((pt_content.to(device=device, dtype=dtype), gt_content.to(device=device, dtype=dtype)), dim=1)
        features = features + model.f0_encoder(bins, out_dtype=dtype)
        prompt = reference_mel(reference.to(device), bundle["singer"].metadata).to(dtype)
        mel = sample_flow(model, prompt, features, end - start, steps, cfg, (seed + start) % (2 ** 64), progress, transformer_options(bundle["singer"]))
        audio = decode(bundle, mel)["waveform"]
        count = min(audio.shape[-1], result.shape[-1] - start * 480)
        window = torch.ones(count)
        fade = min(48000, count // 2)
        if start:
            window[:fade] = torch.linspace(0, 1, fade)
        if end < frames:
            window[-fade:] = torch.linspace(1, 0, fade)
        result[..., start * 480:start * 480 + count] += audio[..., :count] * window
        weights[..., start * 480:start * 480 + count] += window
    return {"waveform": result / weights.clamp(min=1e-8), "sample_rate": 24000}


def fit_f0(contour, frames):
    contour = values(contour, float)
    return contour[:frames] + [0.] * max(0, frames - len(contour))


def separate(component, audio, chunk_seconds=5):
    model, device, dtype = activate(component, memory=1024 * 1024 ** 2)
    waveform = comfy.audio.resample(audio["waveform"].float(), audio["sample_rate"], 44100)
    waveform = waveform.repeat(1, 2, 1) if waveform.shape[1] == 1 else waveform[:, :2]
    chunk = max(2048, round(chunk_seconds * 44100))
    step = chunk // 2
    output = torch.zeros(waveform.shape[0], len(model.mask_estimators), 2, waveform.shape[-1])
    weights = torch.zeros(waveform.shape[-1])
    progress = comfy.utils.ProgressBar(math.ceil(waveform.shape[-1] / step))
    for start in range(0, waveform.shape[-1], step):
        mm.throw_exception_if_processing_interrupted()
        count = min(chunk, waveform.shape[-1] - start)
        part = F.pad(waveform[..., start:start + count], (0, chunk - count))
        prediction = model(part.to(device=device, dtype=dtype), transformer_options(component)).float().cpu()[..., :count]
        window = torch.hann_window(chunk, periodic=False).clamp(min=.001)[:count]
        output[..., start:start + count] += prediction * window
        weights[start:start + count] += window
        progress.update(1)
    output /= weights.clamp(min=1e-8)
    vocal = output[:, 0]
    # Preserve all original residual audio, including backing harmonies and reverberation.
    return {"waveform": vocal, "sample_rate": 44100}, {"waveform": waveform - vocal, "sample_rate": 44100}
