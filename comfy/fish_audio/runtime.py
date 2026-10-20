"""ComfyUI owns model residency; each call owns its decoding state."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from tokenizers import Tokenizer
from safetensors import safe_open
import comfy.audio
import comfy.ops
import comfy.storage
import comfy.utils
import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_prefetch as prefetch
from .config import FishOmniConfig
from .codec import Codec
from .model import Backbone
from .tower import Tower
from .prompt import build_drama3_prompt, validate_drama3_text, split_drama3_text, DRAMA3_PREFILL_CHUNK_SIZE, DRAMA3_DEFAULT_MAX_NEW_TOKENS, DRAMA3_CHUNK_MAX_NEW_TOKENS
from .history import Reference, Take, History, validate, validate_codes, cpu_codes


class GenerationModel(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.config = config
        self.dtype, self.device = dtype, device
        self.backbone = Backbone(config.text, dtype, device, operations)
        self.lm_head = operations.Linear(config.text.hidden_size, config.text.vocab_size, bias=False, dtype=dtype, device=device)
        self.tower = Tower(config, dtype, device, operations)

    def get_dtype(self):
        return self.dtype

    def get_dynamic_vram__units(self):
        return ([*self.backbone.layers, *self.tower.shared_layers,
                 *[(layer, head) for layer, head in zip(self.tower.residual_layers, self.tower.codebook_heads)]],
                [(self.backbone.embed_tokens, self.backbone.norm, self.lm_head, self.tower.codebook_embeddings)])

    def embed(self, frames):
        return self.tower.embed_backbone(frames, self.backbone.embed_tokens(frames[:, 0]))

    def generate(self, prompt, max_frames, settings, generator):
        capacity = len(prompt) + max_frames
        if not 0 < max_frames or capacity > self.config.text.max_position_embeddings:
            raise ValueError('Drama 3 output allowance exceeds remaining model context')
        frames = prompt.to(device=self.device)
        caches = [[] for _ in self.backbone.layers]
        tower_caches = ([[] for _ in self.tower.shared_layers], [[] for _ in self.tower.residual_layers])
        # Each prefill row labels the following frame; the last row is completed
        # by the first sampled frame, preserving the tower's temporal alignment.
        hidden = None
        for start in range(0, len(frames), DRAMA3_PREFILL_CHUNK_SIZE):
            mm.throw_exception_if_processing_interrupted()
            end = min(start + DRAMA3_PREFILL_CHUNK_SIZE, len(frames))
            hidden = self.backbone(self.embed(frames[start:end])[None], start, caches, capacity)[0]
            paired_end = min(end, len(frames) - 1)
            if paired_end > start:
                self.tower.prefill(hidden[:paired_end - start], start, frames[start:paired_end], frames[start + 1:paired_end + 1], tower_caches, capacity)
        hidden = hidden[-1:]
        previous, main_history, produced = frames[-1:], frames[:, 0], []
        finish_reason = 'length'
        for index in range(max_frames):
            mm.throw_exception_if_processing_interrupted()
            position = len(frames) - 1 + index
            sampled = self.tower.decode(self.lm_head(hidden), hidden, position, previous, main_history, tower_caches, capacity, settings, generator)
            if sampled[0, 0].item() == self.config.eos_token_id:
                finish_reason = 'eos'
                break
            produced.append(sampled[0, 1:].to('cpu', copy=True))
            previous = sampled
            main_history = torch.cat((main_history[-15:], sampled[:, 0]))
            if index + 1 < max_frames:
                hidden = self.backbone(self.embed(sampled)[None], position + 1, caches, capacity)[0]
        if not produced:
            raise ValueError('Drama 3 ended without producing audio frames')
        return torch.stack(produced, 1), finish_reason


@dataclass(frozen=True)
class Model:
    generation: comfy.model_patcher.CoreModelPatcher
    codec: comfy.model_patcher.CoreModelPatcher
    tokenizer: Tokenizer
    config: FishOmniConfig
    compatibility: str
    provenance: dict


def metadata(path, expected):
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        meta = saved.metadata() or {}
    if meta.get('format') != expected or meta.get('package_version') != '1':
        raise ValueError(f'Expected converted {expected} version 1; run scripts/convert_drama3.py offline')
    manifest_path = Path(path).with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if len(manifest.get('output_sha256', '')) != 64 or manifest.get('output_size') != Path(path).stat().st_size:
        raise ValueError('Converted package is incomplete or differs from its manifest')
    return meta, manifest


def load(generation_path, codec_path):
    device = mm.get_torch_device()
    if not mm.supports_nvfp4_compute(device):
        raise RuntimeError('The Drama 3 NVFP4 package requires a Blackwell CUDA GPU')
    meta, receipt = metadata(generation_path, 'drama3_nvfp4')
    codec_meta, codec_receipt = metadata(codec_path, 'fish_dac')
    if meta['codec_hash'] != codec_meta['source_hash']:
        raise ValueError('Select the codec packaged with this Drama 3 model')
    raw_config = json.loads(meta['config'])
    config = FishOmniConfig.from_dict(raw_config)
    dtype = torch.bfloat16
    operations = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
    # Quantized op loaders use their construction device for packed payloads.
    # Dense placeholders are uninitialized; expert banks allocate only on load.
    model = GenerationModel(config, dtype, torch.device('cpu'), operations)
    codec = Codec(dtype, torch.device('cpu'), comfy.ops.manual_cast)
    state = comfy.utils.load_torch_file(str(generation_path), safe_load=True)
    if set(state) != set(receipt['tensor_names']):
        raise ValueError('Generator tensor inventory differs from the conversion receipt')
    generation_fast_disk = comfy.storage.state_dict_fast_disk(state)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    state = comfy.utils.load_torch_file(str(codec_path), safe_load=True)
    if set(state) != set(codec_receipt['tensor_names']):
        raise ValueError('Codec tensor inventory differs from the conversion receipt')
    codec_fast_disk = comfy.storage.state_dict_fast_disk(state)
    codec.load_state_dict(state, strict=True, assign=True)
    del state
    mm.archive_model_dtypes(model)
    mm.archive_model_dtypes(codec)
    model.device = torch.device('cpu')
    codec.device = torch.device('cpu')
    generation = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=mm.unet_offload_device(), fast_disk=generation_fast_disk)
    codec_patcher = comfy.model_patcher.CoreModelPatcher(codec, load_device=device, offload_device=mm.vae_offload_device(), fast_disk=codec_fast_disk)
    tokenizer = Tokenizer.from_str(meta['tokenizer'])
    compatibility = hashlib.sha256((meta['tokenizer'] + meta['codec_hash'] + meta['config']).encode()).hexdigest()
    return Model(generation, codec_patcher, tokenizer, config, compatibility,
                 {'source_revision': meta['source_revision'], 'upstream_revision': meta['upstream_revision'],
                  'model_sha256': receipt['output_sha256'], 'codec_sha256': codec_receipt['output_sha256'],
                  'nvfp4_input_passes': 1, 'precision': 'NVFP4 routed experts, BF16 remainder'})


def encode_reference(model, audio, transcript, speaker):
    if type(speaker) is not int or not 0 <= speaker <= 15:
        raise ValueError('Drama 3 speaker ID must be 0–15')
    validate_drama3_text(transcript, 'reference transcript')
    waveform = audio['waveform']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.numel() == 0:
        raise ValueError('Supply one nonempty reference AUDIO clip')
    waveform = comfy.audio.resample(waveform.float().mean(1, keepdim=True), audio['sample_rate'], 44100)
    mm.throw_exception_if_processing_interrupted()
    # Encoder attention has four positions per VQ frame, including its masks.
    frames = (waveform.shape[-1] + 2047) // 2048
    mm.load_models_gpu([model.codec], memory_required=max(512 * 1024 ** 2, frames * frames * 96 + frames * 1024 * 128))
    try:
        with prefetch.malloc_graph_scope(model.codec.load_device):
            codes = model.codec.model.encode(waveform.to(device=model.codec.load_device, dtype=torch.bfloat16))[0]
        mm.throw_exception_if_processing_interrupted()
        result = Reference(transcript, speaker, cpu_codes(codes), model.compatibility)
        validate_codes(result.codes, reference=True)
        return result
    finally:
        prefetch.cleanup_prefetch_queues()


def synthesize(model, text, references, history, seed, temperature, top_p, top_k, max_frames, single_chunk):
    if history is None:
        history = History(model.compatibility)
    validate(history, model.compatibility)
    refs = history.references + tuple(references)
    validate(History(model.compatibility, refs), model.compatibility)
    transcripts = [f'<|speaker:{r.speaker}|> {r.transcript}' for r in refs]
    reference_codes = [r.codes for r in refs]
    previous = [(t, codes) for take in history.takes for t, codes in take.chunks]
    validate_drama3_text(text, 'text')
    texts = [text] if single_chunk else split_drama3_text(text)
    settings = (temperature, top_k, top_p)
    generated, finish_reasons = [], []
    # Validate every prompt's known context before loading the network. Later
    # chunks also include newly generated continuity, counted when available.
    for chunk in texts:
        build_drama3_prompt(chunk, model.tokenizer, model.config, transcripts, reference_codes, previous)
    generator = torch.Generator(device=model.generation.load_device).manual_seed(seed)
    try:
        for chunk in texts:
            continuity = generated if len(generated) < 2 else [generated[0], generated[-1]]
            prompt = build_drama3_prompt(chunk, model.tokenizer, model.config, transcripts, reference_codes, previous + continuity)
            allowance = max_frames if max_frames > 0 else (DRAMA3_DEFAULT_MAX_NEW_TOKENS if single_chunk else DRAMA3_CHUNK_MAX_NEW_TOKENS)
            allowance = min(allowance, model.config.text.max_position_embeddings - len(prompt))
            if allowance <= 0:
                raise ValueError('Drama 3 prompt leaves no generation capacity')
            # Both full-attention banks and thirteen temporal tower layers retain
            # KV over this execution; reserve caches plus bounded prefill workspace.
            c, a = model.config.text, model.config.audio_decoder
            per_frame = sum(t == 'full_attention' for t in c.layer_types) * c.num_key_value_heads * c.head_dim * 4
            per_frame += (a.shared_enrichment_layers + a.num_codebooks - 1) * a.n_local_heads * a.head_dim * 4
            recurrent = sum(t == 'linear_attention' for t in c.layer_types) * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4
            mm.load_models_gpu([model.generation], memory_required=(len(prompt) + allowance) * per_frame + recurrent + 1024 ** 3)
            with prefetch.malloc_graph_scope(model.generation.load_device):
                codes, reason = model.generation.model.generate(prompt, allowance, settings, generator)
            finish_reasons.append(reason)
            generated.append((chunk, cpu_codes(codes)))
        mm.throw_exception_if_processing_interrupted()
        waves = []
        for _, codes in generated:
            frames = codes.shape[1]
            mm.load_models_gpu([model.codec], memory_required=max(1024 ** 3, frames * frames * 16 + frames * 1024 * 128))
            with prefetch.malloc_graph_scope(model.codec.load_device):
                decoded = model.codec.model.decode(codes[None].to(model.codec.load_device))
            waves.append(decoded.float().cpu())
        mm.throw_exception_if_processing_interrupted()
        audio = {'waveform': torch.cat(waves, -1), 'sample_rate': 44100}
        take = Take(text, tuple(generated), dict(seed=seed, temperature=temperature, top_p=top_p, top_k=top_k,
                    max_frames=max_frames, single_chunk=single_chunk, finish_reasons=finish_reasons, **model.provenance))
        updated = History(model.compatibility, refs, history.takes + (take,))
        validate(updated)
        return audio, updated
    finally:
        prefetch.cleanup_prefetch_queues()
