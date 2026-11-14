"""Whisper networks using native operations, attention and layer prefetch.

Architecture: OpenAI Whisper (MIT); generation controls reuse Transformers 5.22.2
(Apache-2.0). No Transformers network is constructed or moved to an accelerator.
"""
from dataclasses import dataclass
import copy
import json

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperTokenizerFast
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration, WhisperPreTrainedModel

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.audio_encoders.whisper import AudioEncoder
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_masked
from .common import package, patcher, take_component, waveform


class Encoder(AudioEncoder):
    def __init__(self, n_mels=128, n_ctx=1500, n_state=1280, n_head=20, n_layer=32,
                 dtype=None, device=None, operations=None):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer, dtype, device, operations)
        self.compute_dtype = dtype

    def get_dtype(self):
        return self.compute_dtype

    def get_dynamic_vram__units(self):
        return list(self.layers), [(self.conv1, self.conv2, self.embed_positions, self.layer_norm)]

    def forward(self, features):
        hidden = F.gelu(self.conv2(F.gelu(self.conv1(features)))).transpose(1, 2)
        if hidden.shape[1] != self.embed_positions.num_embeddings:
            raise ValueError('Whisper features must be padded to the encoder context')
        hidden = hidden + comfy.ops.cast_to_input(self.embed_positions.weight, hidden)
        queue = prefetch.make_prefetch_queue(list(self.layers), hidden.device, {'prefetch_dynamic_vbars': True})
        for layer in self.layers:
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden = layer(hidden)
            prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='block')
        return self.layer_norm(hidden)


class Attention(nn.Module):
    def __init__(self, dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, comfy.ops.manual_cast.Linear(dim, dim, bias=name != 'k_proj', dtype=dtype, device=device))

    def forward(self, hidden, source=None, mask=None, cache=None, index=0, heads=()):
        batch, length, dim = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        cross = source is not None
        owner = (cache.cross_attention_cache if cross else cache.self_attention_cache) if cache is not None else None
        if cross and cache is not None and cache.is_updated.get(index, False):
            key, value = owner.layers[index].keys, owner.layers[index].values
        else:
            source = hidden if source is None else source
            key = self.k_proj(source).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
            value = self.v_proj(source).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
            if owner is not None:
                key, value = owner.update(key, value, index)
                if cross:
                    cache.is_updated[index] = True
        aligned = None
        if heads:
            # Selected heads only, and only for the final teacher-forced pass.
            aligned = (query[:, heads].float() @ key[:, heads].float().transpose(-1, -2)) / (dim // self.heads)**0.5
        result = optimized_attention_masked(AttentionTensorContainer(query), AttentionTensorContainer(key),
                                            AttentionTensorContainer(value), self.heads, mask,
                                            skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_proj(result), aligned


class DecoderLayer(nn.Module):
    def __init__(self, config, dtype, device):
        super().__init__()
        dim = config.d_model
        self.self_attn = Attention(dim, config.decoder_attention_heads, dtype, device)
        self.encoder_attn = Attention(dim, config.decoder_attention_heads, dtype, device)
        for name in ('self_attn_layer_norm', 'encoder_attn_layer_norm', 'final_layer_norm'):
            setattr(self, name, comfy.ops.manual_cast.LayerNorm(dim, dtype=dtype, device=device))
        self.fc1 = comfy.ops.manual_cast.Linear(dim, config.decoder_ffn_dim, dtype=dtype, device=device)
        self.fc2 = comfy.ops.manual_cast.Linear(config.decoder_ffn_dim, dim, dtype=dtype, device=device)

    def forward(self, hidden, encoded, mask, cache, index, heads):
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), mask=mask, cache=cache, index=index)[0]
        output, aligned = self.encoder_attn(self.encoder_attn_layer_norm(hidden), encoded, cache=cache, index=index, heads=heads)
        hidden = hidden + output
        return hidden + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden)))), aligned


class Decoder(nn.Module):
    def __init__(self, config, dtype=torch.float16, device='cpu'):
        super().__init__()
        self.config = config
        self.compute_dtype = dtype
        self.embed_tokens = comfy.ops.manual_cast.Embedding(config.vocab_size, config.d_model, dtype=dtype, device=device)
        self.embed_positions = comfy.ops.manual_cast.Embedding(config.max_target_positions, config.d_model, dtype=dtype, device=device)
        self.layers = nn.ModuleList([DecoderLayer(config, dtype, device) for _ in range(config.decoder_layers)])
        self.layer_norm = comfy.ops.manual_cast.LayerNorm(config.d_model, dtype=dtype, device=device)
        self.proj_out = comfy.ops.manual_cast.Linear(config.d_model, config.vocab_size, bias=False, dtype=dtype, device=device)

    def get_dynamic_vram__units(self):
        return list(self.layers), [(self.embed_tokens, self.embed_positions, self.layer_norm, self.proj_out)]

    def get_dtype(self):
        return self.compute_dtype

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, past_key_values=None,
                inputs_embeds=None, position_ids=None, use_cache=True, alignment_heads=(), **kwargs):
        hidden = self.embed_tokens(input_ids, out_dtype=self.compute_dtype) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        length = hidden.shape[1]
        if position_ids is None:
            position_ids = torch.arange(past, past + length, device=hidden.device)[None]
        if int(position_ids.max()) >= self.config.max_target_positions:
            raise ValueError('Whisper decoding exceeds its token context')
        hidden = hidden + self.embed_positions(position_ids).to(hidden)
        mask = hidden.new_zeros((hidden.shape[0], 1, length, past + length))
        query = torch.arange(past, past + length, device=hidden.device)[:, None]
        key = torch.arange(past + length, device=hidden.device)[None]
        mask.masked_fill_(key > query, torch.finfo(hidden.dtype).min)
        if attention_mask is not None:
            mask.masked_fill_(~attention_mask[:, None, None, :past + length].bool(), torch.finfo(hidden.dtype).min)
        selected = {}
        for layer, head in alignment_heads:
            selected.setdefault(layer, []).append(head)
        alignments = []
        queue = prefetch.make_prefetch_queue(list(self.layers), hidden.device, {'prefetch_dynamic_vbars': True})
        for index, layer in enumerate(self.layers):
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden, aligned = layer(hidden, encoder_hidden_states, mask, past_key_values if use_cache else None,
                                        index, selected.get(index, ()))
                if aligned is not None:
                    alignments.append(aligned)
            prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='block')
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=self.layer_norm(hidden),
                    past_key_values=past_key_values, cross_attentions=tuple(alignments) if alignments else None)


def networks(config, dtype=torch.float16, device='cpu'):
    config = WhisperConfig(**config) if isinstance(config, dict) else config
    if config.encoder_ffn_dim != config.d_model * 4 or config.activation_function != 'gelu' or config.scale_embedding:
        raise ValueError('Unsupported Whisper encoder FFN, activation or embedding scale')
    return (Encoder(config.num_mel_bins, config.max_source_positions, config.d_model,
                    config.encoder_attention_heads, config.encoder_layers,
                    dtype=dtype, device=device, operations=comfy.ops.manual_cast), Decoder(config, dtype, device))


@dataclass(frozen=True)
class Model:
    encoder: object
    decoder: object
    config: WhisperConfig
    generation_config: dict
    tokenizer: object
    extractor: object
    dtype: torch.dtype

    def get_models(self):
        return [self.encoder, self.decoder]

    def _comfy_cache_tensors(self):
        return self.get_models()


def load(path, dtype=torch.float16):
    meta, state = package(path, 'whisper')
    config = WhisperConfig(**json.loads(meta['config']))
    from tokenizers import Tokenizer
    # Construct entirely from embedded assets; tokenizer special-token helpers
    # remain Whisper's implementation, with no from_pretrained/network lookup.
    saved_config = json.loads(meta['tokenizer_config'])
    token_config = {key: saved_config[key] for key in ('language', 'task', 'predict_timestamps', 'errors', 'model_max_length') if key in saved_config}
    # All added tokens and their properties already live in tokenizer_json.
    # Replaying HF's serialized AddedToken dictionaries would add them twice.
    embedded = Tokenizer.from_str(bytes(state.pop('tokenizer_json').tolist()).decode())
    generation_config = json.loads(meta['generation_config'])
    required_ids = [('<|startoftranscript|>', config.decoder_start_token_id), ('<|endoftext|>', config.eos_token_id)]
    for field, token in (('decoder_start_token_id', '<|startoftranscript|>'), ('eos_token_id', '<|endoftext|>'),
                         ('no_timestamps_token_id', '<|notimestamps|>'), ('prev_sot_token_id', '<|startofprev|>')):
        if field in generation_config:
            required_ids.append((token, generation_config[field]))
    required_ids.extend(generation_config.get('lang_to_id', {}).items())
    required_ids.extend((f'<|{task}|>', token) for task, token in generation_config.get('task_to_id', {}).items())
    if 'no_timestamps_token_id' in generation_config:
        required_ids.extend([('<|nospeech|>', generation_config['no_timestamps_token_id'] - 1),
                             ('<|0.00|>', generation_config['no_timestamps_token_id'] + 1)])
    for token, token_id in required_ids:
        if type(token_id) is not int or not 0 <= token_id < config.vocab_size or embedded.token_to_id(token) != token_id:
            raise ValueError(f'Unexpected native Whisper tokenizer ID: {token}')
    tokenizer = WhisperTokenizerFast(tokenizer_object=embedded, **token_config)
    encoder, decoder = networks(config, dtype)
    encoder_state, decoder_state = take_component(state, 'encoder.'), take_component(state, 'decoder.')
    if state:
        raise ValueError(f'Unexpected native Whisper tensors: {sorted(state)}')
    device = mm.get_torch_device()
    return Model(patcher(encoder, encoder_state, device), patcher(decoder, decoder_state, device), config,
                 generation_config, tokenizer,
                 WhisperFeatureExtractor(**json.loads(meta['preprocessor_config'])), dtype)


class AdmittedEncoder(nn.Module):
    main_input_name = 'input_features'

    def __init__(self, model):
        super().__init__()
        self.component, self.dtype, self.config = model.encoder, model.dtype, model.config

    @property
    def conv1(self):
        return self.component.model.conv1

    @property
    def conv2(self):
        return self.component.model.conv2

    def forward(self, input_features, **kwargs):
        mm.load_models_gpu([self.component], memory_required=max(256 * 1024**2, input_features.shape[0] * self.config.d_model * self.config.max_source_positions * 64))
        return BaseModelOutput(last_hidden_state=self.component.model(input_features.to(device=self.component.load_device, dtype=self.dtype)))


class AdmittedDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.component, self.dtype, self.config = model.decoder, model.dtype, model.config

    def memory_required(self, batch, length, source_length, dtype, cache=None, use_cache=True, alignment_heads=()):
        size = dtype.itemsize
        dim = self.config.d_model
        past = cache.get_seq_length() if cache is not None else 0
        cache_bytes = resident = largest = 0
        if use_cache:
            for cross, context in ((False, self.config.max_target_positions), (True, source_length)):
                owner = (cache.cross_attention_cache if cross else cache.self_attention_cache) if cache is not None else None
                for index in range(self.config.decoder_layers):
                    layer = owner.layers[index] if owner is not None and index < len(owner.layers) else None
                    tensors = (layer.keys, layer.values) if layer is not None and layer.is_initialized else (None, None)
                    for tensor in tensors:
                        cache_dtype = dtype
                        if tensor is not None:
                            resident += tensor.numel() * tensor.element_size()
                            cache_dtype = tensor.dtype if cross else torch.promote_types(dtype, tensor.dtype)
                        allocation = batch * context * dim * cache_dtype.itemsize
                        cache_bytes += allocation
                        largest = max(largest, allocation)
        # Dynamic concatenation and beam index_select retain one old K/V tensor
        # while allocating its replacement. Existing cache bytes are already live.
        required = max(0, cache_bytes - resident) + largest
        required += batch * dim * (length + source_length) * size * 8
        required += batch * length * (past + length) * size
        # Generation keeps native logits while computing float32 token scores.
        required += batch * length * self.config.vocab_size * (size + 4)
        if alignment_heads:
            required += batch * len(alignment_heads) * length * source_length * 4 * 2
            required += batch * dim * (length + source_length) * 4
        return required

    def forward(self, input_ids=None, **kwargs):
        embeds = kwargs.get('inputs_embeds')
        batch, length = input_ids.shape if embeds is None else embeds.shape[:2]
        encoded = kwargs.get('encoder_hidden_states')
        source_length = encoded.shape[1] if encoded is not None else self.config.max_source_positions
        required = self.memory_required(batch, length, source_length, self.dtype if embeds is None else embeds.dtype,
                                        kwargs.get('past_key_values'), kwargs.get('use_cache', True), kwargs.get('alignment_heads', ()))
        mm.load_models_gpu([self.component], memory_required=max(256 * 1024**2, required))
        return self.component.model(input_ids=input_ids, **kwargs)


class EncoderDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.decoder = AdmittedEncoder(model), AdmittedDecoder(model)

    def get_encoder(self, modality=None):
        return self.encoder

    def get_decoder(self, modality=None):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.component.model.embed_tokens

    def forward(self, input_features=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None,
                decoder_attention_mask=None, past_key_values=None, decoder_inputs_embeds=None,
                decoder_position_ids=None, use_cache=True, **kwargs):
        encoded = self.encoder(input_features) if encoder_outputs is None else encoder_outputs
        output = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                              encoder_hidden_states=encoded[0], past_key_values=past_key_values,
                              inputs_embeds=decoder_inputs_embeds, position_ids=decoder_position_ids, use_cache=use_cache)
        return Seq2SeqModelOutput(last_hidden_state=output.last_hidden_state, past_key_values=output.past_key_values,
                                  encoder_last_hidden_state=encoded[0])


class Generation(WhisperForConditionalGeneration):
    """An execution-local facade reusing upstream token/long-form controls only."""
    def __init__(self, model):
        WhisperPreTrainedModel.__init__(self, model.config)
        self.model = EncoderDecoder(model)
        self.proj_out = model.decoder.model.proj_out
        config = copy.deepcopy(model.generation_config)
        config['_from_model_config'] = False
        self.generation_config = GenerationConfig(**config)
        self.max_target_positions = model.config.max_target_positions
        self.execution_device = model.decoder.load_device

    @property
    def device(self):
        return self.execution_device

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        if expand_size > 1 and is_encoder_decoder:
            encoded = model_kwargs['encoder_outputs'].last_hidden_state
            decoder = self.model.decoder
            batch = encoded.shape[0] * expand_size
            required = decoder.memory_required(batch, input_ids.shape[1], encoded.shape[1], decoder.dtype,
                                                model_kwargs.get('past_key_values'), model_kwargs.get('use_cache', True))
            required += encoded.numel() * encoded.element_size() * expand_size
            mm.load_models_gpu([decoder.component], memory_required=max(256 * 1024**2, required))
        return super()._expand_inputs_for_generation(expand_size, is_encoder_decoder, input_ids, **model_kwargs)

    def _postprocess_outputs(self, seek_outputs, *args, **kwargs):
        # Neither long nor short segments need to retain completed KV caches.
        if not isinstance(seek_outputs, torch.Tensor):
            seek_outputs = type(seek_outputs)(**{key: value for key, value in seek_outputs.items() if key != 'past_key_values'})
        return super()._postprocess_outputs(seek_outputs, *args, **kwargs)


def align_words(model, audio, tokens, language=None, task='transcribe'):
    """Bounded OpenAI-style teacher-forced alignment using selected heads."""
    from transformers.models.whisper.generation_whisper import _dynamic_time_warping, _median_filter
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, _combine_tokens_into_words
    text_tokens = [int(token) for token in tokens if int(token) < model.config.eos_token_id]
    if not text_tokens:
        return []
    heads = model.generation_config.get('alignment_heads')
    if not heads:
        raise ValueError('Whisper package is missing alignment heads')
    tokenizer = copy.deepcopy(model.tokenizer)
    tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=False)
    prefix = tokenizer.prefix_tokens
    ids = torch.tensor([prefix + text_tokens + [model.config.eos_token_id]], device=model.decoder.load_device)
    if ids.shape[1] > model.config.max_target_positions:
        raise ValueError('Whisper alignment exceeds token context')
    features = model.extractor(audio.numpy(), sampling_rate=16000, padding='max_length', return_tensors='pt').input_features
    encoded = AdmittedEncoder(model)(features).last_hidden_state
    output = AdmittedDecoder(model)(ids, encoder_hidden_states=encoded, use_cache=False, alignment_heads=heads)
    valid_frames = max(1, min(encoded.shape[1], (len(audio) + 319) // 320))
    weights = torch.cat(output.cross_attentions, dim=1)[0, :, :, :valid_frames].float().softmax(-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std.clamp_min(1e-6)
    if valid_frames >= model.config.median_filter_width:
        weights = _median_filter(weights, model.config.median_filter_width)
    # Include the no-timestamps position predicting the first text token and
    # the final text position predicting EOT, matching OpenAI's n+1 boundaries.
    matrix = weights.mean(0)[len(prefix) - 1:-1].cpu().double().numpy()
    text_indices, time_indices = _dynamic_time_warping(-matrix)
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    boundaries = time_indices[jumps] * 0.02
    word_language = 'english' if task == 'translate' else LANGUAGES.get(language, language)
    words, word_tokens, indices = _combine_tokens_into_words(tokenizer, text_tokens, word_language)
    result = []
    for word, group in zip(words, indices):
        if group:
            result.append({'word': word, 'start': round(float(boundaries[group[0]]), 3),
                           'end': round(float(boundaries[min(group[-1] + 1, len(boundaries) - 1)]), 3)})
    return result


def transcribe_many(model, audios, language='en', beam_size=5, batch_size=1, condition_on_previous_text=False,
                    task='transcribe', chunk_length_s=0, vad_filter=False, initial_prompt=''):
    if task not in ('transcribe', 'translate') or not 1 <= beam_size <= 10 or not 1 <= batch_size <= 64:
        raise ValueError('Invalid Whisper task, beam size or batch size')
    if not 0 <= chunk_length_s <= 120:
        raise ValueError('Whisper chunk length must be between 0 and 120 seconds')
    values = [waveform(audio) for audio in audios]
    queues = []
    for value in values:
        if vad_filter:
            # This is CPU ONNX preprocessing only, never a CTranslate2 model.
            from faster_whisper.vad import get_speech_timestamps
            spans = [(span['start'], span['end']) for span in get_speech_timestamps(value.numpy())]
        else:
            spans = [(0, len(value))]
        jobs = []
        for start, end in spans:
            if chunk_length_s:
                step = round(chunk_length_s * 16000)
            elif batch_size > 1 and not condition_on_previous_text:
                step = model.config.max_source_positions * 320
            else:
                step = end - start
            for left in range(start, end, max(1, step)):
                jobs.append((left, min(end, left + step)))
        queues.append(jobs)
    results = [{'text': '', 'language': language or None, 'segments': [], 'warnings': [], 'backend': 'comfyui-native-whisper'} for _ in audios]
    generation = Generation(model)
    try:
        while any(queues):
            work = []
            for index, queue in enumerate(queues):
                while queue and len(work) < batch_size:
                    work.append((index, *queue.pop(0)))
                    if condition_on_previous_text:
                        break
            if not work:
                break
            # Previous-text prompts are private; group only exactly equal prompts.
            prompts = [initial_prompt + (results[index]['text'] if condition_on_previous_text else '') for index, _, _ in work]
            if len(set(prompts)) > 1:
                keep = prompts[0]
                for index, start, end in reversed([item for item, prompt in zip(work, prompts) if prompt != keep]):
                    queues[index].insert(0, (start, end))
                work = [item for item, prompt in zip(work, prompts) if prompt == keep]
                prompts = [keep] * len(work)
            clips = [values[index][start:end].numpy() for index, start, end in work]
            longest = max(len(clip) for clip in clips)
            inputs = model.extractor(clips, sampling_rate=16000, padding='max_length',
                                     max_length=max(longest, model.config.max_source_positions * 320), truncation=False,
                                     return_attention_mask=True, return_tensors='pt')
            kwargs = dict(language=language or None, task=task, num_beams=beam_size, do_sample=False,
                          return_timestamps=True, return_segments=True, return_dict_in_generate=True,
                          return_token_timestamps=False, condition_on_prev_tokens=condition_on_previous_text,
                          temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                          compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6,
                          attention_mask=inputs.attention_mask.to(model.decoder.load_device))
            if prompts[0]:
                kwargs['prompt_ids'] = model.tokenizer.get_prompt_ids(prompts[0], return_tensors='pt').to(model.decoder.load_device)
            # Long recordings keep their mel input across upstream window seeks.
            # Admit that allocation before copying it onto a pressured GPU.
            input_bytes = inputs.input_features.numel() * torch.empty((), dtype=model.dtype).element_size()
            encoder_workspace = len(work) * model.config.d_model * model.config.max_source_positions * 64
            mm.load_models_gpu([model.encoder], memory_required=max(256 * 1024**2, input_bytes + encoder_workspace))
            generated = generation.generate(inputs.input_features.to(device=model.decoder.load_device, dtype=model.dtype), **kwargs)
            for (index, offset, end), segments in zip(work, generated['segments']):
                for segment in segments:
                    tokens = segment['tokens'].detach().cpu().tolist()
                    text = model.tokenizer.decode(tokens, skip_special_tokens=True)
                    language_ids = {value: name.removeprefix('<|').removesuffix('|>')
                                    for name, value in model.generation_config.get('lang_to_id', {}).items()}
                    for token in segment['result']['sequences'].tolist():
                        if token in language_ids:
                            results[index]['language'] = language_ids[token]
                            break
                    left, right = float(segment['start']), float(segment['end'])
                    clip_start = offset + round(left * 16000)
                    clip_end = min(end, offset + round(right * 16000))
                    words = align_words(model, values[index][clip_start:clip_end], tokens,
                                        results[index]['language'], task) if text.strip() and clip_end > clip_start else []
                    base = offset / 16000 + left
                    for word in words:
                        word['start'] += base
                        word['end'] += base
                    results[index]['segments'].append({'text': text, 'start': base,
                            'end': min(end / 16000, offset / 16000 + right), 'words': words})
                    results[index]['text'] += text
            del generated
        return results
    finally:
        del generation
        prefetch.cleanup_prefetch_queues()
