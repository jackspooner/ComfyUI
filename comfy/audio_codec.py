"""Discrete audio codes and native codec lifecycle (separate from float VAEs)."""
from dataclasses import dataclass

import torch

import comfy.audio
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.storage
import comfy.utils
from comfy.ldm.audio.moss_audio_tokenizer import MossAudioTokenizer


@dataclass(frozen=True)
class AudioCodecIdentity:
    family: str
    revision: str
    sample_rate: int
    hop_length: int
    codebook_sizes: tuple[int, ...]


MOSS_V1 = AudioCodecIdentity('moss_audio_tokenizer', '3cd226ba2947efa357ef453bcad111b6eafba782', 24000, 1920, (1024,) * 32)


@dataclass(frozen=True)
class AudioCodeSegment:
    codes: torch.Tensor  # [batch, ordered codebook prefix, frames], integer IDs
    lengths: tuple[int, ...]
    source_sample_lengths: tuple[int, ...] | None = None
    decode_skip_frames: tuple[int, ...] | None = None


@dataclass(frozen=True)
class AudioCodes:
    codec: AudioCodecIdentity
    segments: tuple[AudioCodeSegment, ...]
    version: int = 1

    def validate(self, codec):
        if self.version != 1 or self.codec != codec:
            raise ValueError('AUDIO_CODES version or codec identity does not match the loaded audio codec')
        for segment in self.segments:
            codes = segment.codes
            if codes.ndim != 3 or codes.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
                raise ValueError('AUDIO_CODES requires integer [batch, codebook, frame] tensors')
            if not 1 <= codes.shape[1] <= len(codec.codebook_sizes):
                raise ValueError('AUDIO_CODES must contain an ordered prefix of the codec codebooks')
            if len(segment.lengths) != codes.shape[0] or any(type(n) is not int or not 0 <= n <= codes.shape[2] for n in segment.lengths):
                raise ValueError('AUDIO_CODES valid frame lengths must match the batch and stored frames')
            if segment.source_sample_lengths is not None and (len(segment.source_sample_lengths) != codes.shape[0] or any(type(n) is not int or n < 0 for n in segment.source_sample_lengths)):
                raise ValueError('AUDIO_CODES source sample lengths must be nonnegative integers matching the batch')
            if segment.decode_skip_frames is not None and (len(segment.decode_skip_frames) != codes.shape[0] or any(type(n) is not int or n < 0 for n in segment.decode_skip_frames)):
                raise ValueError('AUDIO_CODES decoded prefix lengths must be nonnegative integers matching the batch')
            for i, length in enumerate(segment.lengths):
                for book, size in enumerate(codec.codebook_sizes[:codes.shape[1]]):
                    valid = codes[i, book, :length]
                    if torch.any((valid < 0) | (valid >= size)):
                        raise ValueError(f'AUDIO_CODES contains invalid IDs in codebook {book}; remove protocol padding before decoding')


class AudioCodec:
    def __init__(self, state_dict, metadata, device=None, disable_dynamic=False):
        if not metadata or metadata.get('architecture') != MOSS_V1.family or metadata.get('model_revision') != MOSS_V1.revision or metadata.get('conversion_version') != '1':
            raise ValueError('Unsupported audio codec; prepare the pinned MOSS v1 artifact with scripts/convert_moss_codec.py')
        # Lazy native operations do not all check checkpoint shapes. Validate the
        # complete architecture before assigning any file-backed parameters.
        expected = MossAudioTokenizer(device='meta', dtype=torch.float32, operations=torch.nn).state_dict()
        if state_dict.keys() != expected.keys() or any(state_dict[k].shape != v.shape or state_dict[k].dtype != torch.float32 for k, v in expected.items()):
            raise ValueError('MOSS v1 audio codec requires the complete folded FP32 checkpoint')
        self.identity = MOSS_V1
        self.device = model_management.vae_device() if device is None else device
        offload_device = model_management.vae_offload_device()
        operations = comfy.ops.pick_operations(torch.float32, torch.float32, self.device, disable_fast_fp8=True)
        self.model = MossAudioTokenizer(device=offload_device, dtype=torch.float32, operations=operations).eval()
        model_management.archive_model_dtypes(self.model)
        patcher_type = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
        self.patcher = patcher_type(self.model, load_device=self.device, offload_device=offload_device, fast_disk=comfy.storage.state_dict_fast_disk(state_dict))
        self.patcher.set_model_compute_dtype(torch.float32)
        self.model.load_state_dict(state_dict, strict=True, assign=self.patcher.is_dynamic())

    def memory_required(self, frames, encode=False):
        # One stream at a time. Reserve dense attention, block workspaces and
        # both chunk outputs and their concatenation; weights are patcher-owned.
        tokens = frames * 8 if encode else min(frames, 100) * 8
        keys = tokens if encode or frames <= 100 else 1000
        attention = tokens * keys * (12 * 4 * 2 + 16)
        blocks = tokens * 768 * 4 * 48
        ring = 169984000 if not encode and frames > 100 else 0
        return attention + blocks + ring + frames * 1920 * 4 * 2 + 64 * 1024**2

    def load_model(self, frames, encode=False):
        with comfy.model_prefetch.pause_malloc_graph():
            model_management.load_models_gpu([self.patcher], memory_required=self.memory_required(frames, encode))

    def encode(self, audio, codebooks=32):
        if not 1 <= codebooks <= len(self.identity.codebook_sizes):
            raise ValueError('MOSS v1 supports 1 to 32 codebooks')
        waveform = audio['waveform']
        if waveform.ndim != 3 or waveform.shape[1] == 0 or not waveform.is_floating_point():
            raise ValueError('Audio codec input must be floating-point [batch, channel, sample] audio')
        waveform = waveform.to(device='cpu', dtype=torch.float32).mean(dim=1, keepdim=True)
        rate = audio['sample_rate']
        if rate <= 0 or int(rate) != rate:
            raise ValueError('Audio codec input sample rate must be a positive integer')
        if waveform.shape[-1]:
            waveform = comfy.audio.resample(waveform, rate, self.identity.sample_rate)
        samples = waveform.shape[-1]
        segments = []
        with model_management.cuda_device_context(self.device):
            if samples:
                self.load_model((samples + self.identity.hop_length - 1) // self.identity.hop_length, encode=True)
            with comfy.ops.use_quantized_matmul(self.model, self.device):
                for item in waveform.split(1):
                    model_management.throw_exception_if_processing_interrupted()
                    if samples:
                        codes, lengths = self.model.encode(item.to(self.device), [samples], codebooks)
                        codes = codes.to(device='cpu', copy=True)
                    else:
                        codes, lengths = torch.empty((1, codebooks, 0), dtype=torch.int64), [0]
                    segments.append(AudioCodeSegment(codes, tuple(lengths), (samples,)))
        return AudioCodes(self.identity, tuple(segments))

    def decode(self, codes):
        codes.validate(self.identity)
        output = []
        with model_management.cuda_device_context(self.device):
            frames = max((n for segment in codes.segments for n in segment.lengths), default=0)
            if frames:
                self.load_model(frames)
            with comfy.ops.use_quantized_matmul(self.model, self.device):
                for segment in codes.segments:
                    for i, length in enumerate(segment.lengths):
                        model_management.throw_exception_if_processing_interrupted()
                        if length:
                            valid = segment.codes[i:i + 1, :, :length].to(device=self.device, dtype=torch.int64)
                            waveform = self.model.decode(valid).to(device='cpu', copy=True)
                        else:
                            waveform = torch.empty((1, 1, 0), dtype=torch.float32)
                        skip = segment.decode_skip_frames[i] if segment.decode_skip_frames is not None else 0
                        if skip:
                            if skip >= length:
                                continue
                            # Retain causal codec context until decoding is complete.
                            waveform = waveform[..., int(waveform.shape[-1] * (skip / length)):].clone()
                        output.append({'waveform': waveform, 'sample_rate': self.identity.sample_rate})
        return output


def load_audio_codec(path, device=None, disable_dynamic=False):
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    codec = AudioCodec(state_dict, metadata, device=device, disable_dynamic=disable_dynamic)
    codec.patcher.cached_patcher_init = (load_audio_codec_patcher, (path, device))
    return codec


def load_audio_codec_patcher(path, device=None, disable_dynamic=False):
    return load_audio_codec(path, device=device, disable_dynamic=disable_dynamic).patcher
