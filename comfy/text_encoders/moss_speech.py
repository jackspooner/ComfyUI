"""MOSS speech tokenization, CLIP adapter and discrete-code output boundary."""
import torch
from tokenizers import Tokenizer

from comfy import sd1_clip
from comfy.audio_codec import AudioCodes, AudioCodeSegment, MOSS_V1
from .moss_tts_delay import MossTTSDelay, AUDIO_PAD


def voice_prompt(text, instruction, language):
    # Match the pinned processor's replacement order, including literal None.
    template = '<user_inst>\n- Reference(s):\nNone\n- Instruction:\n{instruction}\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\n{language}\n- Text:\n{text}\n</user_inst>'
    content = template.replace('{instruction}', str(instruction)).replace('{language}', str(language)).replace('{text}', str(text))
    return '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'


class MossSpeechTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data.get('tokenizer_json')
        if self.tokenizer_json is None:
            raise ValueError('MOSS speech requires an embedded tokenizer; prepare the checkpoint with scripts/convert_moss_speech.py')
        self.tokenizer = Tokenizer.from_str(self.tokenizer_json.numpy().tobytes().decode('utf-8'))

    def tokenize_with_weights(self, text, return_word_ids=False, instruction='', language=None, **kwargs):
        ids = self.tokenizer.encode(voice_prompt(text, instruction, language), add_special_tokens=False).ids
        return {'moss_delay': [[(token, 1.0, i + 1) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}

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
    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, textmodel_json_config={}, model_class=MossTTSDelay,
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
        generated = self.transformer.generate_delay(ids, mask, self.generation_length(tokens, max_length), seed, **sampling)
        segments = []
        for _, assistant in generated:
            segments.extend(undelay_segments(assistant[:, 1:]))
        return AudioCodes(MOSS_V1, tuple(segments))


def te(dtype_llama=None, llama_quantization_metadata=None):
    class MossSpeechTEModel(sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = {**model_options, 'quantization_metadata': llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype_llama or dtype, name='moss_delay', clip_model=MossSpeechClipModel, model_options=model_options)

        def audio_generation_memory(self, tokens, max_length):
            return self.moss_delay.audio_generation_memory(tokens, max_length)

        def generate_audio_codes(self, tokens, max_length=1000, seed=0, **sampling):
            return self.moss_delay.generate_audio_codes(tokens, max_length, seed, **sampling)
    return MossSpeechTEModel
