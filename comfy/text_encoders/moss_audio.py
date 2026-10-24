"""Native MOSS-Audio (OpenMOSS, Apache-2.0): Qwen3 with audio DeepStack."""
import math
import re

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
from comfy import sd1_clip
from comfy.text_generation import format_qwen_chat
from comfy.audio_encoders.whisper import EncoderLayer
from .llama import Qwen3_8B


AUDIO_TOKEN = 151654
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_SPAN = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")


def audio_token_ids(length):
    ids = []
    for start in range(0, length, 25):
        count = min(25, length - start)
        ids.extend([AUDIO_TOKEN] * count)
        if count == 25:
            ids.extend(15 + int(d) for d in str((start + count) // 25 * 2))
    return ids


def audio_features(audio):
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1 or min(waveform.shape[1:]) == 0:
        raise ValueError("MOSS-Audio requires one nonempty AUDIO clip with shape [1, channels, samples].")
    if not isinstance(sample_rate, int) or sample_rate <= 0 or not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError("MOSS-Audio requires a positive sample rate and finite floating-point audio samples.")
    waveform = waveform.float().mean(dim=1)
    waveform = comfy.audio.resample(waveform, sample_rate, 16000)
    # STFT reflect padding needs more than half an FFT window, even for a tiny clip.
    if waveform.shape[-1] <= 200:
        waveform = F.pad(waveform, (0, 201 - waveform.shape[-1]))
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160)
    features = extractor(waveform[0].detach().cpu().numpy(), sampling_rate=16000,
                         padding=False, truncation=False, return_tensors="np", device="cpu").input_features
    return torch.from_numpy(features[0])


class MossAudioEncoder(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None, d_model=1280, n_heads=20, d_ff=5120,
                 n_layers=32, conv_channels=480, capture_layers=(8, 16, 24)):
        super().__init__()
        self.capture_layers = capture_layers
        self.conv1 = operations.Conv2d(1, conv_channels, 3, stride=2, padding=1, dtype=dtype, device=device)
        self.conv2 = operations.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1, dtype=dtype, device=device)
        self.conv3 = operations.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1, dtype=dtype, device=device)
        self.stem_proj = operations.Linear(conv_channels * 16, d_model, dtype=dtype, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dtype=dtype, device=device, operations=operations) for _ in range(n_layers)])
        self.layer_norm = operations.LayerNorm(d_model, eps=1e-5, dtype=dtype, device=device)

    def encode_chunks(self, features, lengths):
        x = F.gelu(self.conv1(features.unsqueeze(1)))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.stem_proj(x.permute(0, 3, 1, 2).flatten(2))
        lengths = [(length + 7) // 8 for length in lengths]
        x = x[:, :max(lengths)]
        half = x.shape[-1] // 2
        inv = torch.exp(-math.log(10000.0) / (half - 1) * torch.arange(half, device=x.device, dtype=torch.float32))
        phase = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
        x = x + torch.cat((phase.sin(), phase.cos()), dim=-1).to(x.dtype)
        mask = x.new_zeros((len(lengths), 1, 1, x.shape[1]))
        for i, length in enumerate(lengths):
            mask[i, :, :, length:] = torch.finfo(x.dtype).min
        captured = []
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for i, layer in enumerate(self.layers):
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, mask)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope="block")
            if i in self.capture_layers:
                captured.append(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        x = self.layer_norm(x)
        return [torch.cat([value[i, :length] for i, length in enumerate(lengths)]) for value in [x, *captured]]

    def forward(self, features):
        chunks = features.T.split(400, dim=0)
        lengths = [chunk.shape[0] for chunk in chunks]
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
        outputs = [[] for _ in range(1 + len(self.capture_layers))]
        for start in range(0, len(chunks), 64):
            encoded = self.encode_chunks(padded[start:start + 64], lengths[start:start + 64])
            for destination, value in zip(outputs, encoded):
                destination.append(value)
        return [torch.cat(parts) for parts in outputs]


class AudioAdapter(nn.Module):
    def __init__(self, input_size=1280, hidden_size=8192, output_size=4096, dtype=None, device=None, operations=None):
        super().__init__()
        self.gate_proj = operations.Linear(input_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(input_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(hidden_size, output_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossAudio(Qwen3_8B):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)
        self.audio_encoder = MossAudioEncoder(dtype=dtype, device=device, operations=operations)
        self.audio_adapter = AudioAdapter(dtype=dtype, device=device, operations=operations)
        self.deepstack_audio_merger_list = nn.ModuleList([AudioAdapter(dtype=dtype, device=device, operations=operations) for _ in range(3)])

    def get_dynamic_vram__units(self):
        units, last_units = self.model.get_dynamic_vram__units()
        return [*self.audio_encoder.layers, *units], last_units

    def preprocess_embed(self, embed, device):
        if embed["type"] != "audio":
            return None, None
        mel = audio_features(embed["data"]).to(device=device, dtype=self.dtype)
        encoded, *captured = self.audio_encoder(mel)
        audio = self.audio_adapter(encoded)
        deepstack = [adapter(value) for adapter, value in zip(self.deepstack_audio_merger_list, captured)]
        ids = audio_token_ids(audio.shape[0])
        tokens = torch.tensor(ids, device=device, dtype=torch.long)
        mask = tokens == AUDIO_TOKEN
        embeddings = self.model.embed_tokens(tokens, out_dtype=audio.dtype)
        embeddings[mask] = audio
        return embeddings, {"mask": mask, "deepstack": deepstack}

    def audio_inputs(self, embeds, embeds_info):
        for entry in embeds_info:
            if entry["type"] == "audio":
                mask = torch.zeros(embeds.shape[:2], device=embeds.device, dtype=torch.bool)
                start = entry["index"]
                mask[0, start:start + entry["size"]] = entry["extra"]["mask"]
                return mask, entry["extra"]["deepstack"]
        return None, None

    def forward(self, input_ids, *args, embeds_info=(), **kwargs):
        mask, deepstack = self.audio_inputs(kwargs.get("embeds"), embeds_info)
        return self.model(input_ids, *args, visual_pos_masks=mask, deepstack_embeds=deepstack, **kwargs)


class MossAudioClipModel(sd1_clip.SDClipModel):
    structured_output_family = "moss_audio"

    def __init__(self, device="cpu", layer="last", layer_idx=-1, dtype=None, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype,
                         special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=MossAudio,
                         enable_attention_masks=True, return_attention_masks=True, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0, mtp=True, constraint=None):
        if isinstance(tokens, dict):
            tokens = tokens["moss_audio"]
        embeds, _, _, info = self.process_tokens([[t[0] for t in row] for row in tokens], self.execution_device)
        if embeds.shape[1] + max_length > self.transformer.model.config.max_position_embeddings:
            raise ValueError(f"MOSS-Audio needs {embeds.shape[1] + max_length} context tokens; its context holds 40960. Shorten the audio, prompt or output length.")
        mask, deepstack = self.transformer.audio_inputs(embeds, info)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p,
                                         repetition_penalty, seed, presence_penalty=presence_penalty,
                                         visual_pos_masks=mask, deepstack_embeds=deepstack, constraint=constraint)


class EmbeddedTokenizer(PreTrainedTokenizerFast):
    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        if tokenizer_data is None:
            raise ValueError("MOSS-Audio checkpoint is missing tokenizer_json; prepare it with scripts/convert_moss_audio.py.")
        raw = tokenizer_data.numpy().tobytes() if isinstance(tokenizer_data, torch.Tensor) else tokenizer_data
        return cls(tokenizer_object=Tokenizer.from_str(raw.decode("utf-8")))


class MossAudioSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data.get("tokenizer_json")
        super().__init__(self.tokenizer_json, embedding_directory=embedding_directory, embedding_size=4096,
                         embedding_key="moss_audio", tokenizer_class=EmbeddedTokenizer, has_start_token=False,
                         has_end_token=False, pad_with_end=False, pad_to_max_length=False, max_length=99999999,
                         min_length=1, pad_token=151643, tokenizer_data=tokenizer_data, disable_weights=True)

    def state_dict(self):
        return {"tokenizer_json": self.tokenizer_json}


class MossAudioTokenizer(sd1_clip.SD1Tokenizer):
    def tokenize_chat(self, messages, thinking=False):
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
        return self.tokenize_with_weights(format_qwen_chat(messages, thinking), skip_template=True, thinking=thinking)

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="moss_audio", tokenizer=MossAudioSDTokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, audio=None, thinking=True, skip_template=False,
                              system_prompt="", llama_template=None, **kwargs):
        if not skip_template:
            if llama_template is not None:
                text = llama_template.format(text)
            else:
                prefix = "<|audio_bos|><|AUDIO|><|audio_eos|>\n" if audio is not None else ""
                text = "<|im_start|>system\n" + (system_prompt or "You are a helpful assistant.") + "<|im_end|>\n<|im_start|>user\n" + prefix + text + "<|im_end|>\n<|im_start|>assistant\n"
            if not thinking:
                text += "<think>\n\n</think>\n\n"
        spans = list(AUDIO_SPAN.finditer(text))
        if len(spans) != int(audio is not None):
            raise ValueError("MOSS-Audio requires exactly one <|audio_bos|><|AUDIO|><|audio_eos|> placeholder when audio is supplied, and none for text only.")
        tokenizer = self.moss_audio.tokenizer
        if spans:
            span = spans[0]
            ids = tokenizer.encode(text[:span.start()], add_special_tokens=False)
            ids += [AUDIO_START, {"type": "audio", "data": audio}, AUDIO_END]
            ids += tokenizer.encode(text[span.end():], add_special_tokens=False)
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
        return {"moss_audio": [[(token, 1.0, i + 1) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}


def te(dtype_llama=None, llama_quantization_metadata=None):
    class MossAudioTEModel(sd1_clip.SD1ClipModel):
        def get_chat_context_length(self):
            return self.moss_audio.transformer.model.config.max_position_embeddings

        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                raise ValueError("MOSS-Audio currently supports the original BF16 checkpoint only.")
            super().__init__(device=device, dtype=dtype_llama or dtype, name="moss_audio", clip_model=MossAudioClipModel, model_options=model_options)
    return MossAudioTEModel
