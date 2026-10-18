"""Qwen2.5-Omni Thinker conditioning and AuK's learned layer fusion."""
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy import sd1_clip
from comfy.audio_encoders.whisper import EncoderLayer
from .llama import Qwen25_3B


def audio_features(audio):
    waveform = audio['waveform'].float().mean(dim=1)
    waveform = comfy.audio.resample(waveform, audio['sample_rate'], 16000)
    if waveform.shape[0] != 1:
        raise ValueError('AuK conditioning accepts one reference audio clip per instruction.')
    if waveform.shape[-1] < 640:
        waveform = F.pad(waveform, (0, 640 - waveform.shape[-1]))
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160,
                                        chunk_length=max(300, math.ceil(waveform.shape[-1] / 16000)), return_attention_mask=True)
    features = extractor(waveform[0].detach().cpu().numpy(), sampling_rate=16000, padding='max_length',
                         truncation=False, return_tensors='pt')
    length = math.ceil(waveform.shape[-1] / 160)
    return features.input_features[0, :, :length].clone()


class OmniAudioEncoder(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None, dim=1280, heads=20, layers=32, output_dim=2048):
        super().__init__()
        self.conv1 = operations.Conv1d(128, dim, 3, padding=1, dtype=dtype, device=device)
        self.conv2 = operations.Conv1d(dim, dim, 3, stride=2, padding=1, dtype=dtype, device=device)
        self.layers = nn.ModuleList([EncoderLayer(dim, heads, dim * 4, dtype=dtype, device=device, operations=operations) for _ in range(layers)])
        self.ln_post = operations.LayerNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, output_dim, dtype=dtype, device=device)

    def forward(self, features):
        chunks = features.T.split(200)
        lengths = [chunk.shape[0] for chunk in chunks]
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
        mask = torch.arange(padded.shape[-1], device=features.device)[None, :] < torch.tensor(lengths, device=features.device)[:, None]
        x = F.gelu(self.conv1(padded)) * mask[:, None, :]
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        half = x.shape[-1] // 2
        inv = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * (-math.log(10000) / (half - 1)))
        phase = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
        x = x + torch.cat((phase.sin(), phase.cos()), dim=-1).to(x.dtype)
        lengths = [(length + 1) // 2 for length in lengths]
        bias = x.new_zeros((len(chunks), 1, 1, x.shape[1]))
        for i, length in enumerate(lengths):
            bias[i, :, :, length:] = torch.finfo(x.dtype).min
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {'prefetch_dynamic_vbars': True})
        for layer in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, bias)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        x = torch.cat([x[i, :length] for i, length in enumerate(lengths)], dim=0)
        x = F.avg_pool1d(x.T, 2, stride=2).T
        return self.proj(self.ln_post(x))


class AuKThinker(Qwen25_3B):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)
        self.audio_tower = OmniAudioEncoder(dtype=dtype, device=device, operations=operations)
        self.layer_weights = nn.Parameter(torch.empty(self.num_layers, dtype=torch.float32, device=device))
        self.layer_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))

    def get_dynamic_vram__units(self):
        return [*self.audio_tower.layers, *self.model.layers], []

    def preprocess_embed(self, embed, device):
        if embed['type'] == 'audio':
            return self.audio_tower(audio_features(embed['data']).to(device=device, dtype=self.dtype)), None
        return None, None

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=False, dtype=None, embeds_info=()):
        _, hidden = self.model(input_ids, attention_mask, embeds=embeds, num_tokens=num_tokens,
                               intermediate_output='all', final_layer_norm_intermediate=False, dtype=self.dtype)
        # Native Llama returns embedding + pre-layer states + final normalized state,
        # matching HF hidden_states; AuK deliberately excludes the embedding state.
        hidden = F.layer_norm(hidden[:, 1:], (hidden.shape[-1],))
        weights = comfy.model_management.cast_to(self.layer_weights, dtype=torch.float32, device=hidden.device).softmax(dim=0)
        scale = comfy.model_management.cast_to(self.layer_scale, dtype=torch.float32, device=hidden.device)
        fused = (hidden * weights[None, :, None, None]).sum(dim=1) * scale
        return fused, None


class AuKClipModel(sd1_clip.SDClipModel):
    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, layer='last', textmodel_json_config={}, model_class=AuKThinker,
                         special_tokens={'pad': 151643}, layer_norm_hidden_state=False, enable_attention_masks=True,
                         return_attention_masks=True, model_options=model_options)


    def process_tokens(self, tokens, device):
        embeds, attention_mask, lengths, info = super().process_tokens(tokens, device)
        return embeds.to(dtype=self.transformer.dtype), attention_mask, lengths, info


class AuKTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data['tokenizer_json']
        self.config_json = tokenizer_data['auk_config_json']
        self.config = json.loads(self.config_json.numpy().tobytes())
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(self.tokenizer_json.numpy().tobytes().decode()),
                                               chat_template=self.config['chat_template'])

    def tokenize_with_weights(self, text, return_word_ids=False, audio=None, **kwargs):
        if audio is None and not text.endswith('|<no_prompt_audio>|'):
            text += '|<no_prompt_audio>|'
        content = [{'type': 'text', 'text': text}]
        if audio is not None:
            content.append({'type': 'audio', 'audio': 'local-reference'})
        formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        if audio is None:
            ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        else:
            before, after = formatted.split('<|AUDIO|>')
            ids = self.tokenizer.encode(before, add_special_tokens=False)
            ids += [{'type': 'audio', 'data': audio}]
            ids += self.tokenizer.encode(after, add_special_tokens=False)
        return {'auk': [[(token, 1.0, i) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}

    def state_dict(self):
        return {'tokenizer_json': self.tokenizer_json, 'auk_config_json': self.config_json}


def te(dtype_llama=None, llama_quantization_metadata=None):
    class AuKTEModel(sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = {**model_options, 'quantization_metadata': llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype_llama or dtype, name='auk', clip_model=AuKClipModel, model_options=model_options)
    return AuKTEModel
