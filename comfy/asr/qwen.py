"""Native Qwen3-ASR/ForcedAligner (Qwen, Apache-2.0).

Reference: QwenLM/Qwen3-ASR c17a131fe028b2e428b6e80a33d30bb4fa57b8df.
Networks reuse ComfyUI's Omni encoder and Qwen3 decoder, including prefetch.
"""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor
from transformers.models.qwen3_asr.processing_qwen3_asr import (
    LANGUAGE_CODE_TO_NAME, _parse_single_output, _fix_timestamps, _clean_tokens, Qwen3ASRProcessor,
)

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.console_progress import advance, phase, report_progress
from comfy.text_encoders.llama import Llama2_, Qwen3_8BConfig
from comfy.text_encoders.qwen3_omni import AudioEncoder, audio_output_length
from .common import waveform, package, patcher, take_component


class Encoder(AudioEncoder):
    def __init__(self, config, device, dtype, operations):
        super().__init__(config, device, dtype, operations)
        self.compute_dtype = dtype

    def get_dtype(self):
        return self.compute_dtype

    def get_dynamic_vram__units(self):
        return list(self.layers), [(self.conv2d1, self.conv2d2, self.conv2d3, self.conv_out,
                                   self.ln_post, self.proj1, self.proj2)]


class Decoder(Llama2_):
    def __init__(self, config, classes, dtype, device):
        base = Qwen3_8BConfig()
        cfg = SimpleNamespace(**{name: getattr(base, name) for name in dir(base) if not name.startswith('_')})
        for name in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                     'num_key_value_heads', 'head_dim', 'vocab_size', 'rms_norm_eps', 'rope_theta',
                     'max_position_embeddings'):
            setattr(cfg, name, config.get(name, config.get('rope_parameters', {}).get(name, getattr(cfg, name))) if name == 'rope_theta' else config[name])
        cfg.fixed_kv = True
        cfg.graph_dynamic_vbar_blocks = True
        cfg.prefetch_dynamic_vbars = True
        super().__init__(cfg, device=device, dtype=dtype, ops=comfy.ops.manual_cast)
        self.compute_dtype = dtype
        if classes != config['vocab_size']:
            self.lm_head = comfy.ops.manual_cast.Linear(cfg.hidden_size, classes, bias=False, device=device, dtype=dtype)

    def get_dynamic_vram__units(self):
        return list(self.layers), [(self.embed_tokens, self.norm, self.lm_head)]

    def get_dtype(self):
        return self.compute_dtype

    def compute_freqs_cis(self, position_ids, device):
        # Qwen rounds rotary sin/cos to the hidden dtype before applying RoPE.
        return tuple(value.to(self.norm.weight.dtype).float()
                     for value in super().compute_freqs_cis(position_ids, device))


@dataclass(frozen=True)
class Model:
    encoder: object
    decoder: object
    tokenizer: Tokenizer
    config: dict
    extractor: WhisperFeatureExtractor
    aligner: bool
    korean_scores: dict

    def get_models(self):
        return [self.encoder, self.decoder]


def networks(config, dtype=torch.bfloat16, device='cpu'):
    thinker = config['thinker_config']
    text = thinker['text_config']
    return (Encoder(thinker['audio_config'], device, dtype, comfy.ops.manual_cast),
            Decoder(text, thinker.get('classify_num', text['vocab_size']), dtype, device))


def load(path):
    meta, state = package(path, 'qwen3_asr')
    config = json.loads(meta['config'])
    encoder, decoder = networks(config)
    tokenizer = Tokenizer.from_str(bytes(state.pop('tokenizer_json').tolist()).decode())
    encoder_state, decoder_state = take_component(state, 'encoder.'), take_component(state, 'decoder.')
    if state:
        raise ValueError(f'Unexpected native Qwen tensors: {sorted(state)}')
    device = mm.get_torch_device()
    return Model(patcher(encoder, encoder_state, device), patcher(decoder, decoder_state, device), tokenizer,
                 config, WhisperFeatureExtractor(**json.loads(meta['preprocessor_config'])),
                 'classify_num' in config['thinker_config'], json.loads(meta.get('korean_scores', '{}')))


def split_audio(value, seconds=180):
    """Qwen's contiguous low-amplitude cuts, including its ±5 s search window."""
    samples, start = len(value), 0
    target, expand, window = int(seconds * 16000), 80000, 1600
    while samples - start > target:
        cut = start + target
        left, right = max(start, cut - expand), min(samples, cut + expand)
        segment = np.abs(value[left:right].numpy())
        sums = np.convolve(segment, np.ones(window, dtype=np.float32), mode='valid')
        if len(sums):
            low = int(np.argmin(sums))
            boundary = left + low + int(np.argmin(segment[low:low + window]))
        else:
            boundary = cut
        boundary = max(start + 1, min(boundary, samples))
        yield start, boundary
        start = boundary
    if start < samples:
        yield start, samples


def language_name(language):
    if not language or language.lower() == 'auto':
        return None
    result = LANGUAGE_CODE_TO_NAME.get(language.lower(), language.capitalize())
    if result not in LANGUAGE_CODE_TO_NAME.values():
        raise ValueError(f'Unsupported Qwen transcription language: {language}')
    return result


def prompt_ids(model, text, length):
    audio_id = model.config['thinker_config']['audio_token_id']
    ids = model.tokenizer.encode(text, add_special_tokens=False).ids
    if ids.count(audio_id) != 1:
        raise ValueError('Qwen prompt must contain exactly one audio placeholder')
    return torch.tensor([[token for value in ids for token in ([value] * length if value == audio_id else [value])]])


def embeddings(model, value, text):
    phase('encoding audio')
    value = torch.nn.functional.pad(value, (0, max(0, 8000 - len(value))))
    inputs = model.extractor(value.numpy(), sampling_rate=16000, padding='longest',
                             truncation=False, return_tensors='pt', return_attention_mask=True)
    length = int(inputs.attention_mask.sum())
    features = inputs.input_features[0, :, :length]
    device = model.encoder.load_device
    frames = audio_output_length(length)
    mm.load_models_gpu([model.encoder], memory_required=max(256 * 1024**2, length * 128 * 256 + frames * 1024 * 48))
    try:
        with prefetch.malloc_graph_scope(device):
            encoded = model.encoder.model(features.to(device=device, dtype=torch.bfloat16)).cpu()
    finally:
        prefetch.cleanup_prefetch_queues()
    ids = prompt_ids(model, text, encoded.shape[0])
    return ids, encoded


def decode(model, ids, encoded, max_new_tokens=1024, classify=False):
    phase('aligning words' if classify else 'decoding')
    decoder = model.decoder.model
    cfg, device = decoder.config, model.decoder.load_device
    capacity = ids.shape[1] + (0 if classify else max_new_tokens)
    if capacity > cfg.max_position_embeddings:
        raise ValueError('Qwen transcription exceeds model context; use shorter audio chunks')
    cache_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * capacity * cfg.head_dim * 2
    mm.load_models_gpu([model.decoder], memory_required=max(256 * 1024**2, cache_bytes + ids.shape[1]**2 * 4))
    cache = None
    try:
        ids = ids.to(device)
        hidden = decoder.embed_tokens(ids, out_dtype=torch.bfloat16)
        hidden[ids == model.config['thinker_config']['audio_token_id']] = encoded.to(hidden)
        if classify:
            hidden = decoder(None, embeds=hidden, past_key_values=[])[0]
            positions = ids == model.config['timestamp_token_id']
            return decoder.lm_head(hidden[positions]).float().argmax(-1).cpu()
        cache = decoder.init_kv_cache(1, capacity, device, torch.bfloat16)
        generated = []
        for index in range(max_new_tokens):
            mm.throw_exception_if_processing_interrupted()
            hidden, _, cache = decoder(None, embeds=hidden, past_key_values=cache)
            token = decoder.lm_head(hidden[:, -1:]).float().argmax(-1)
            value = int(token.item())
            if value in (151643, 151645):
                break
            generated.append(value)
            advance()
            hidden = decoder.embed_tokens(token, out_dtype=torch.bfloat16)
        return model.tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        cache = None
        prefetch.cleanup_prefetch_queues()


@report_progress('Qwen ASR', 'preparing audio', unit='tokens')
def transcribe(model, audio, language='', context='', max_new_tokens=1024, chunk_seconds=180):
    if model.aligner:
        raise ValueError('Select the Qwen ASR model for transcription')
    if not 1 <= max_new_tokens <= 8192 or not 1 <= chunk_seconds <= 1200:
        raise ValueError('Invalid Qwen token allowance or chunk length')
    value, forced = waveform(audio), language_name(language)
    chunks = []
    for start, end in split_audio(value, chunk_seconds):
        text = (f'<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n'
                '<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n')
        if forced:
            text += f'language {forced}<asr_text>'
        ids, encoded = embeddings(model, value[start:end], text)
        parsed = _parse_single_output(decode(model, ids, encoded, max_new_tokens))
        chunks.append({'start': start / 16000, 'end': end / 16000,
                       'language': forced or parsed['language'], 'text': parsed['transcription']})
    return {'text': ''.join(chunk['text'] for chunk in chunks), 'chunks': chunks,
            'segments': [], 'warnings': [], 'backend': 'comfyui-native-qwen'}


@report_progress('Qwen ForcedAligner', 'aligning words', unit='chunks')
def align(model, audio, transcript):
    if not model.aligner:
        raise ValueError('Select the Qwen ForcedAligner model for alignment')
    value = waveform(audio)
    result = {**transcript, 'segments': [], 'warnings': list(transcript.get('warnings', []))}
    # The installed Transformers processor carries the pinned Qwen word splitting
    # and timestamp repair algorithms; it performs no model loading or I/O here.
    processor = object.__new__(Qwen3ASRProcessor)
    for chunk in transcript['chunks']:
        if not chunk['text']:
            continue
        language = chunk['language'] or ''
        if language not in model.config['support_languages']:
            result['warnings'].append({'code': 'transcription_timestamps_incomplete',
                                       'message': f'Forced alignment does not support {language or "unknown language"}'})
            continue
        try:
            if language == 'Korean':
                if not model.korean_scores:
                    raise ImportError('Korean alignment requires conversion with --korean-dictionary from the pinned Qwen ASR assets')
                from soynlp.tokenizer import LTokenizer
                words = _clean_tokens(LTokenizer(scores=model.korean_scores).tokenize(chunk['text']))
            else:
                words = processor.split_words_for_alignment(chunk['text'], language)
        except ImportError as error:
            result['warnings'].append({'code': 'transcription_timestamps_incomplete', 'message': str(error)})
            continue
        if not words:
            continue
        text = '<|audio_start|><|audio_pad|><|audio_end|>' + '<timestamp><timestamp>'.join(words) + '<timestamp><timestamp>'
        start, end = round(chunk['start'] * 16000), round(chunk['end'] * 16000)
        ids, encoded = embeddings(model, value[start:end], text)
        times = decode(model, ids, encoded, classify=True).numpy() * model.config['timestamp_segment_time']
        if len(times) != 2 * len(words):
            raise ValueError('Forced aligner returned an invalid timestamp count')
        times = _fix_timestamps(times)
        advance()
        duration = (end - start) / 16000
        for index, word in enumerate(words):
            left, right = times[index * 2] / 1000, times[index * 2 + 1] / 1000
            if not 0 <= left <= right <= duration + 0.08:
                result['warnings'].append({'code': 'transcription_timestamps_incomplete',
                                           'message': 'Forced aligner returned an out-of-range word timestamp'})
                continue
            result['segments'].append({'text': word, 'start': round(chunk['start'] + left, 3),
                                       'end': round(chunk['start'] + min(right, duration), 3)})
    return result
