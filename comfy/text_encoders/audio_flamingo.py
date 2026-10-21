"""Native NVIDIA Audio Flamingo 3; math follows Transformers' Apache-2.0 implementation."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy import sd1_clip
from comfy.audio_encoders.whisper import AudioEncoder
from .llama import BaseLlama, BaseQwen3, BaseGenerate, Llama2_, Qwen25_7BVLI_Config
from .moss_audio import EmbeddedTokenizer


@dataclass
class AudioFlamingoConfig(Qwen25_7BVLI_Config):
    vocab_size: int = 151672
    max_position_embeddings: int = 32768
    lm_head: bool = True
    fixed_kv: bool = True
    graph_dynamic_vbar_blocks = True
    prefetch_dynamic_vbars = True
    rope_dims = None
    stop_tokens = [151645]


def audio_features(audio):
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or min(waveform.shape[1:]) == 0:
        raise ValueError("Audio Flamingo requires one nonempty AUDIO clip.")
    waveform = comfy.audio.resample(waveform.float().mean(dim=1), audio["sample_rate"], 16000)[0]
    # The encoder's pooled output needs at least two convolution frames.
    if waveform.shape[-1] < 321:
        waveform = F.pad(waveform, (0, 321 - waveform.shape[-1]))
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160)
    result = []
    for chunk in waveform.split(480000):
        features = extractor(chunk.detach().cpu().numpy(), sampling_rate=16000, padding="max_length",
                             truncation=False, return_tensors="np", device="cpu").input_features
        result.append((torch.from_numpy(features), (chunk.shape[-1] + 159) // 160))
    return result


class AudioFlamingoEncoder(AudioEncoder):
    def forward(self, features, length):
        hidden = F.gelu(self.conv2(F.gelu(self.conv1(features)))).transpose(1, 2)
        hidden = hidden + comfy.ops.cast_to_input(self.embed_positions.weight[:hidden.shape[1]], hidden)
        valid = (length + 1) // 2
        mask = hidden.new_zeros((1, 1, 1, hidden.shape[1]))
        mask[..., valid:] = torch.finfo(hidden.dtype).min
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), hidden.device, {"prefetch_dynamic_vbars": True})
        for layer in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden = layer(hidden, mask)
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope="block")
        hidden = self.layer_norm(F.avg_pool1d(hidden.transpose(1, 2), 2).transpose(1, 2))
        return hidden[:, :valid // 2].clone()


class AudioProjector(nn.Module):
    def __init__(self, dtype, device, operations):
        super().__init__()
        self.linear_1 = operations.Linear(1280, 3584, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(3584, 3584, dtype=dtype, device=device)

    def forward(self, hidden):
        return self.linear_2(F.gelu(self.linear_1(hidden)))


class AudioFlamingo(BaseLlama, BaseQwen3, BaseGenerate, nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = AudioFlamingoConfig(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype
        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.audio_tower = AudioFlamingoEncoder(dtype=dtype, device=device, operations=operations)
        self.multi_modal_projector = AudioProjector(dtype, device, operations)

    def get_dynamic_vram__units(self):
        units, last = self.model.get_dynamic_vram__units()
        return [*self.audio_tower.layers, *units], last

    def preprocess_embed(self, embed, device):
        if embed["type"] != "audio":
            return None, None
        encoded = [self.multi_modal_projector(self.audio_tower(features.to(device=device, dtype=self.dtype), length))[0]
                   for features, length in audio_features(embed["data"])]
        return torch.cat(encoded), None


class AudioFlamingoClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=-1, dtype=None, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype,
                         special_tokens={"pad": 151671}, layer_norm_hidden_state=False, model_class=AudioFlamingo,
                         enable_attention_masks=True, return_attention_masks=True, model_options=model_options)


class AudioFlamingoSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json = tokenizer_data.get("tokenizer_json")
        if self.tokenizer_json is None:
            raise ValueError("Audio Flamingo checkpoint is missing tokenizer_json; use scripts/convert_audio_flamingo.py.")
        super().__init__(self.tokenizer_json, embedding_directory=embedding_directory, embedding_size=3584,
                         embedding_key="audio_flamingo", tokenizer_class=EmbeddedTokenizer, has_start_token=False,
                         has_end_token=False, pad_with_end=False, pad_to_max_length=False, max_length=99999999,
                         min_length=1, pad_token=151671, tokenizer_data=tokenizer_data, disable_weights=True)

    def state_dict(self):
        return {"tokenizer_json": self.tokenizer_json}


class AudioFlamingoTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="audio_flamingo", tokenizer=AudioFlamingoSDTokenizer)

    def tokenize_chat(self, messages, thinking=False):
        if any(not isinstance(message["content"], str) for message in messages):
            raise ValueError("Audio Flamingo does not support images in conversation history.")
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
        text = "".join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n' for message in messages)
        return self.tokenize_with_weights(text + "<|im_start|>assistant\n", skip_template=True)

    def tokenize_with_weights(self, text, return_word_ids=False, audio=None, skip_template=False,
                              system_prompt="", llama_template=None, **kwargs):
        if not skip_template:
            if llama_template is not None:
                text = llama_template.format(text)
            else:
                text = ("<|im_start|>system\n" + (system_prompt or "You are a helpful assistant.") + "<|im_end|>\n"
                        + "<|im_start|>user\n" + ("<sound>" if audio is not None else "") + text
                        + "<|im_end|>\n<|im_start|>assistant\n")
        if text.count("<sound>") != int(audio is not None):
            raise ValueError("Audio Flamingo requires one <sound> placeholder with audio and none for text only.")
        tokenizer = self.audio_flamingo.tokenizer
        if audio is not None:
            before, after = text.split("<sound>")
            ids = tokenizer.encode(before, add_special_tokens=False) + [{"type": "audio", "data": audio}]
            ids += tokenizer.encode(after, add_special_tokens=False)
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
        return {"audio_flamingo": [[(token, 1.0, i + 1) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}


def te(dtype_llama=None, llama_quantization_metadata=None):
    class AudioFlamingoTEModel(sd1_clip.SD1ClipModel):
        def get_chat_context_length(self):
            return self.audio_flamingo.transformer.model.config.max_position_embeddings

        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = {**model_options, "quantization_metadata": llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype_llama or dtype, name="audio_flamingo", clip_model=AudioFlamingoClipModel, model_options=model_options)
    return AudioFlamingoTEModel
