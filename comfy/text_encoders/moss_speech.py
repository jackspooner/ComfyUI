"""MOSS speech tokenization, CLIP adapter and discrete-code output boundary."""
import torch
from tokenizers import Tokenizer

import comfy.audio
from comfy import sd1_clip
from comfy.audio_codec import AudioCodes, AudioCodeSegment, MOSS_V1
from .moss_tts_delay import MossTTSDelay, AUDIO_PAD
from .moss_text_normalization import normalize_tts_text


def reference_codes(reference, codebooks):
    reference.validate(MOSS_V1)
    codes = []
    for segment in reference.segments:
        if segment.codes.shape[1] < codebooks:
            raise ValueError(f'MOSS speech reference needs at least {codebooks} codebooks; encode with MOSS Speech Reference')
        for i, length in enumerate(segment.lengths):
            codes.append(segment.codes[i, :codebooks, :length].T.to(device='cpu', dtype=torch.int64, copy=True))
    return codes


def voice_prompt(text, instruction, language, reference=None, tokens=None, quality=None, sound_event=None, ambient_sound=None, dialogue=False):
    # Match the pinned processor's replacement order, including literal None.
    template = '<user_inst>\n- Reference(s):\n{reference}\n- Instruction:\n{instruction}\n- Tokens:\n{tokens}\n- Quality:\n{quality}\n- Sound Event:\n{sound_event}\n- Ambient Sound:\n{ambient_sound}\n- Language:\n{language}\n'
    if dialogue:
        template += '- Scene:\nNone\n'
    template += '- Text:\n{text}\n</user_inst>'
    if dialogue:
        template = template.replace('{tokens}', 'None')
    content = template
    for key, value in [('reference', reference), ('instruction', instruction), ('tokens', None if dialogue else tokens),
                       ('quality', quality), ('sound_event', sound_event), ('ambient_sound', ambient_sound), ('language', language), ('text', text)]:
        if key != 'tokens' or not dialogue:
            content = content.replace('{' + key + '}', str(value))
    return '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'


def encode_speech_reference(codec, audio):
    waveform = audio['waveform'].to(device='cpu', dtype=torch.float32).mean(dim=1, keepdim=True)
    if waveform.shape[-1]:
        waveform = comfy.audio.resample(waveform, audio['sample_rate'], MOSS_V1.sample_rate)
        gain = (-20 - 10 * torch.log10(waveform.square().mean(dim=-1, keepdim=True) + 1e-9)).clamp(-3, 3)
        waveform = waveform * 10 ** (gain / 20)
    return codec.encode({'waveform': waveform, 'sample_rate': MOSS_V1.sample_rate}, 32)


class MossSpeechTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data.get('tokenizer_json')
        if self.tokenizer_json is None:
            raise ValueError('MOSS speech requires an embedded tokenizer; prepare the checkpoint with scripts/convert_moss_speech.py')
        self.tokenizer = Tokenizer.from_str(self.tokenizer_json.numpy().tobytes().decode('utf-8'))
        self.codebooks = tokenizer_data.get('codebooks', 16)
        self.dialogue = tokenizer_data.get('dialogue', False)

    def tokenize_with_weights(self, text, return_word_ids=False, instruction='', language=None, references=None,
                              tokens=None, quality=None, sound_event=None, ambient_sound=None, normalize_text=False, continuation=None, **kwargs):
        if normalize_text:
            text = normalize_tts_text(text)
        codes = []
        labels = []
        speaker = 1
        for reference in references or ():
            if reference is None:
                if self.dialogue:
                    labels.append(f'[S{speaker}]: None')
                speaker += 1
                continue
            for item in reference_codes(reference, self.codebooks):
                codes.append(item)
                slots = '<|audio_user_slot|>' * (len(item) + self.codebooks - 1) if len(item) else ''
                labels.append(f'[S{speaker}]:\n<|audio_start|>{slots}<|audio_end|>')
                speaker += 1
        prompt = voice_prompt(text, instruction, language, '\n'.join(labels) if references is not None else None,
                              tokens, quality, sound_event, ambient_sound, self.dialogue)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        starts = [i + 1 for i, token in enumerate(ids) if token == 151652]
        ends = [i for i, token in enumerate(ids) if token == 151653]
        if len(starts) != len(codes) or len(ends) != len(codes):
            raise ValueError('Audio markers in the speech prompt do not match its references')
        if continuation is not None:
            prefix = reference_codes(continuation, self.codebooks)
            prefix = torch.cat(prefix) if prefix else torch.empty((0, self.codebooks), dtype=torch.int64)
            ids.append(151652)
            starts.append(len(ids))
            codes.append(prefix)
            ids.extend([151656] * len(prefix))
        result = {'moss_delay': [[(token, 1.0, i + 1) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}
        if codes:
            result['moss_references'] = list(zip(starts, codes))
        return result

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def state_dict(self):
        return {'tokenizer_json': self.tokenizer_json}


def undelay_segments(delayed):
    frames = max(0, delayed.shape[0] - delayed.shape[1] + 1)
    if not frames:
        return []
    codes = torch.stack([delayed[q:q + frames, q] for q in range(delayed.shape[1])], dim=1)
    # A budget may end before all delayed heads arrive. Only complete frames
    # can be looked up; never turn the protocol's learned pad into codec IDs.
    complete = (codes != AUDIO_PAD).all(dim=1).tolist()
    segments = []
    start = None
    for i, valid in enumerate([*complete, False]):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            segment = codes[start:i].T.unsqueeze(0).contiguous().clone()
            segments.append(AudioCodeSegment(segment, (i - start,)))
            start = None
    return segments


class MossSpeechClipModel(sd1_clip.SDClipModel):
    def __init__(self, device='cpu', dtype=None, model_options={}, speech_config=None):
        super().__init__(device=device, dtype=dtype, textmodel_json_config=speech_config or {}, model_class=MossTTSDelay,
                         special_tokens={'pad': 151643}, layer_norm_hidden_state=False,
                         enable_attention_masks=True, model_options=model_options)
        self.can_assign_sd = False

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=True, assign=self.can_assign_sd)

    def generation_length(self, tokens, max_length):
        if max_length != 0:
            return max_length
        return max(0, self.transformer.model.config.max_position_embeddings - max(map(len, tokens['moss_delay'])))

    def audio_generation_memory(self, tokens, max_length):
        batch = len(tokens['moss_delay'])
        prefix = max(len(row) for row in tokens['moss_delay'])
        capacity = prefix + self.generation_length(tokens, max_length)
        config = self.transformer.model.config
        size = self.transformer.dtype.itemsize
        kv = batch * capacity * config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * size
        work = batch * (prefix * config.hidden_size * 48 * size + config.num_attention_heads * prefix**2 * 8 + config.vocab_size * 8)
        return kv + work + batch * capacity * (self.transformer.codebooks + 1) * 8

    def generate_audio_codes(self, tokens, max_length=1000, seed=0, **sampling):
        rows = [[token[0] for token in row] for row in tokens['moss_delay']]
        length = max(map(len, rows))
        ids = torch.full((len(rows), length, self.transformer.codebooks + 1), AUDIO_PAD, device=self.execution_device, dtype=torch.int64)
        ids[..., 0] = 151643
        mask = torch.zeros((len(rows), length), device=self.execution_device, dtype=torch.int64)
        for i, row in enumerate(rows):
            ids[i, -len(row):, 0] = torch.tensor(row, device=ids.device)
            mask[i, -len(row):] = 1
        for start, codes in tokens.get('moss_references', ()):
            codes = codes.to(device=ids.device)
            for q in range(self.transformer.codebooks):
                count = max(0, min(len(codes), length - start - q))
                ids[0, start + q:start + q + count, q + 1] = codes[:count, q]
        generated = self.transformer.generate_delay(ids, mask, self.generation_length(tokens, max_length), seed, **sampling)
        segments = []
        for prefix_length, assistant in generated:
            decoded = undelay_segments(assistant[:, 1:])
            if prefix_length and decoded:
                first = decoded[0]
                decoded[0] = AudioCodeSegment(first.codes, first.lengths, decode_skip_frames=(prefix_length,))
            segments.extend(decoded)
        return AudioCodes(MOSS_V1, tuple(segments))


def te(dtype_llama=None, llama_quantization_metadata=None, speech_config=None):
    class MossSpeechTEModel(sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = {**model_options, 'quantization_metadata': llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype_llama or dtype, name='moss_delay', clip_model=MossSpeechClipModel,
                             model_options=model_options, speech_config=speech_config)

        def audio_generation_memory(self, tokens, max_length):
            return self.moss_delay.audio_generation_memory(tokens, max_length)

        def generate_audio_codes(self, tokens, max_length=1000, seed=0, **sampling):
            return self.moss_delay.generate_audio_codes(tokens, max_length, seed, **sampling)
    return MossSpeechTEModel
