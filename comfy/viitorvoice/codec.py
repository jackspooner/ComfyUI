"""Native DualCodec lifecycle, with separate W2V semantic residency."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import torchaudio
from transformers import SeamlessM4TFeatureExtractor

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.audio_codec import AudioCodes, AudioCodeSegment, DUALCODEC_V1
from . import package


class DualCodec:
    identity = DUALCODEC_V1

    def __init__(self, path, cpu=False, attention='pytorch', disable_dynamic=False):
        self.codec = package.load(path, 'codec', cpu, attention, disable_dynamic)
        # The exact companion lives in the same prepared bundle, including when
        # the selected VAE is exposed through a standard-category symlink.
        semantic = Path(path).resolve().parent.parent/'audio_encoders/viitor_w2v_bert.safetensors'
        if self.codec.metadata.get('semantic_component') != '../audio_encoders/viitor_w2v_bert.safetensors':
            raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: unexpected semantic component binding')
        self.semantic = package.load(semantic, 'semantic', cpu, attention, disable_dynamic)
        if self.codec.metadata['sources'] != self.semantic.metadata['sources']:
            raise ValueError('VIITOR_NATIVE_PACKAGE_INVALID: codec and semantic component sources differ')
        self.patcher = self.codec.patcher
        self.device = self.patcher.load_device
        self.extractor = SeamlessM4TFeatureExtractor(**json.loads(self.semantic.metadata['processor']))

    def encode(self, audio, codebooks=12):
        if not 1 <= codebooks <= 12:
            raise ValueError('DualCodec supports 1 to 12 ordered codebooks; select codebooks=12 for ViiTorVoice')
        waveform = audio['waveform']
        if waveform.ndim != 3 or not waveform.is_floating_point() or waveform.shape[1] == 0 or not torch.isfinite(waveform).all():
            raise ValueError('DualCodec input must be finite floating-point [batch, channel, sample] audio')
        rate = audio['sample_rate']
        if isinstance(rate, bool) or int(rate) != rate or rate <= 0:
            raise ValueError('DualCodec input sample rate must be a positive integer')
        waveform = waveform.to(device='cpu', dtype=torch.float32).mean(1, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, int(rate), 24000)
        # The original encoder first clips partial codec hops, without trimming
        # internal silence, changing loudness or dropping unselected samples.
        samples = waveform.shape[-1]//960*960
        if not 960 <= samples <= 720000:
            raise ValueError('DualCodec requires between 40 milliseconds and 30 seconds after hop alignment')
        waveform = waveform[..., :samples]
        segments = []
        try:
            for item in waveform.split(1):
                mm.throw_exception_if_processing_interrupted()
                small = torchaudio.functional.resample(item, 24000, 16000)
                features = self.extractor(small.numpy(), sampling_rate=16000, return_tensors='pt')
                length = features['input_features'].shape[1]
                # Scores, relative bias, QKV and convolution/FFN workspaces. The
                # relative distance table avoids materializing [T,T,head_dim].
                memory = 4*16*length*length*4+16*length*1024*4+256*1024**2
                with prefetch.pause_malloc_graph():
                    mm.load_models_gpu([self.semantic.patcher], memory_required=memory)
                semantic_device = self.semantic.patcher.load_device
                with mm.cuda_device_context(semantic_device), comfy.ops.use_quantized_matmul(self.semantic.patcher.model, semantic_device):
                    semantic = self.semantic.patcher.model(features['input_features'].to(semantic_device),
                        features['attention_mask'].to(semantic_device), output_layer=17, options=self.semantic.options())
                    semantic = ((semantic-self.semantic.mean.to(semantic_device))/self.semantic.std.to(semantic_device)).transpose(1, 2)
                    semantic = F.avg_pool1d(semantic, 2, 2).cpu()
                frames = semantic.shape[-1]
                with prefetch.pause_malloc_graph():
                    mm.load_models_gpu([self.patcher], memory_required=self.memory_required(frames, encode=True))
                with mm.cuda_device_context(self.device), comfy.ops.use_quantized_matmul(self.patcher.model, self.device):
                    codes = self.patcher.model.encode(item.to(self.device), semantic.to(self.device), codebooks).cpu()
                segments.append(AudioCodeSegment(codes, (frames,), (samples,)))
        finally:
            prefetch.cleanup_prefetch_queues()
        return AudioCodes(self.identity, tuple(segments))

    def memory_required(self, frames, encode=False):
        return frames*960*96*4*4+frames*16384*4+256*1024**2

    def decode(self, codes):
        codes.validate(self.identity)
        outputs = []
        try:
            for segment in codes.segments:
                for index, frames in enumerate(segment.lengths):
                    mm.throw_exception_if_processing_interrupted()
                    if frames:
                        gap = self.codec.silence[:segment.codes.shape[1], :25][None]
                        packed = torch.cat((gap, segment.codes[index:index+1, :, :frames], gap), -1)
                        with prefetch.pause_malloc_graph():
                            mm.load_models_gpu([self.patcher], memory_required=self.memory_required(frames+50))
                        with mm.cuda_device_context(self.device), comfy.ops.use_quantized_matmul(self.patcher.model, self.device):
                            wave = self.patcher.model.decode(packed.to(self.device)).cpu()
                        wave = wave[..., 25*960:25*960+frames*960-4].clone()
                    else:
                        wave = torch.empty(1, 1, 0)
                    skip = segment.decode_skip_frames[index] if segment.decode_skip_frames is not None else 0
                    wave = wave[..., skip*960:].clone()
                    outputs.append(dict(waveform=wave, sample_rate=24000))
        finally:
            prefetch.cleanup_prefetch_queues()
        return outputs
