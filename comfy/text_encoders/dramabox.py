import itertools
import types

import torch

import comfy.model_management
from comfy.text_encoders.lt import Gemma3_12BModel, LTXAVGemmaTokenizer


def _fix_rope_position_ids(transformer):
    if getattr(transformer, "_dramabox_rope_fixed", False):
        return
    original_forward = transformer.forward

    def forward(x, attention_mask=None, **kwargs):
        if kwargs.get("position_ids") is None and attention_mask is not None:
            kwargs["position_ids"] = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        return original_forward(x, attention_mask, **kwargs)

    transformer.forward = forward
    transformer._dramabox_rope_fixed = True


class _DummyVideoEmbed(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], self.out_features, device=x.device, dtype=x.dtype)


def _audio_only_feature_forward(feature_extractor, hidden_states, attention_mask, padding_side="left"):
    from ltx_core.text_encoders.gemma.feature_extractor import _rescale_norm, norm_and_concat_per_token_rms

    encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
    normed = norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
    audio_dim = feature_extractor.audio_aggregate_embed.out_features
    audio = feature_extractor.audio_aggregate_embed(
        _rescale_norm(normed, audio_dim, feature_extractor.embedding_dim)
    )
    video = audio.new_zeros((audio.shape[0], audio.shape[1], 1))
    return video, audio


def _audio_only_create_embeddings(embeddings_processor, video_features, audio_features, additive_attention_mask):
    mask = additive_attention_mask
    while mask.dim() > 2:
        mask = mask[:, 0]
    binary_mask = (mask >= -1.0).to(torch.int64)
    audio_encoded = None
    if embeddings_processor.audio_connector is not None:
        audio_encoded, _ = embeddings_processor.audio_connector(audio_features, additive_attention_mask)
    return video_features, audio_encoded, binary_mask


def _load_embeddings_processor(audio_components_path, device):
    try:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
            EMBEDDINGS_PROCESSOR_KEY_OPS,
            EmbeddingsProcessorConfigurator,
        )
    except ImportError as exc:
        raise RuntimeError("DramaBox CLIP requires the ltx-core Python package.") from exc

    builder = SingleGPUModelBuilder(
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_path=audio_components_path,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    )
    config = builder.model_config()
    with torch.device(device):
        return EmbeddingsProcessorConfigurator.from_config(config).to(torch.bfloat16)


class DramaBoxTEModel(torch.nn.Module):
    _EP_KEY_MAP = {
        "text_embedding_projection.aggregate_embed.": "feature_extractor.aggregate_embed.",
        "text_embedding_projection.video_aggregate_embed.": "feature_extractor.video_aggregate_embed.",
        "text_embedding_projection.audio_aggregate_embed.": "feature_extractor.audio_aggregate_embed.",
        "model.diffusion_model.video_embeddings_connector.": "video_connector.",
        "model.diffusion_model.audio_embeddings_connector.": "audio_connector.",
    }

    def __init__(self, dtype_llama=None, device="cpu", dtype=None, model_options={}, audio_components_path=None):
        super().__init__()
        self.dtypes = {dtype, dtype_llama}
        self.execution_device = None
        self.gemma3_12b = Gemma3_12BModel(
            device=device,
            dtype=dtype_llama,
            model_options=model_options,
            layer="all",
            layer_idx=None,
        )
        _fix_rope_position_ids(self.gemma3_12b.transformer)
        self.embeddings_processor = _load_embeddings_processor(audio_components_path, device)
        self._enable_audio_only_embeddings()

    def _enable_audio_only_embeddings(self):
        embeddings_processor = self.embeddings_processor
        if embeddings_processor.video_connector is not None:
            del embeddings_processor.video_connector
            embeddings_processor.video_connector = None

        feature_extractor = embeddings_processor.feature_extractor
        if feature_extractor.video_aggregate_embed is not None:
            out_features = feature_extractor.video_aggregate_embed.out_features
            del feature_extractor.video_aggregate_embed
            feature_extractor.video_aggregate_embed = _DummyVideoEmbed(out_features)

        if feature_extractor.audio_aggregate_embed is not None:
            feature_extractor.forward = types.MethodType(_audio_only_feature_forward, feature_extractor)
        embeddings_processor.create_embeddings = types.MethodType(
            _audio_only_create_embeddings, embeddings_processor
        )

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)
        self.gemma3_12b.set_clip_options(options)

    def reset_clip_options(self):
        self.gemma3_12b.reset_clip_options()
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        out, _pooled, extra = self.gemma3_12b.encode_token_weights(token_weight_pairs["gemma3_12b"])
        out = out.float().movedim(1, -1)
        attention_mask = extra["attention_mask"]
        embeddings_processor_device = next(self.embeddings_processor.parameters()).device
        embeddings_processor_dtype = next(self.embeddings_processor.parameters()).dtype
        encoded = self.embeddings_processor.process_hidden_states(
            out.to(device=embeddings_processor_device, dtype=embeddings_processor_dtype),
            attention_mask.to(embeddings_processor_device),
            "left",
        )
        return encoded.audio_encoding, None, {}

    def load_sd(self, state_dict):
        if "model.layers.47.self_attn.q_norm.weight" in state_dict:
            missing, unexpected = self.gemma3_12b.load_sd(state_dict)
            missing = [
                key for key in missing
                if not key.startswith(("vision_model.", "multi_modal_projector."))
            ]
            return missing, unexpected

        embeddings_state_dict = {}
        for key, value in state_dict.items():
            for source_prefix, target_prefix in self._EP_KEY_MAP.items():
                if key.startswith(source_prefix):
                    embeddings_state_dict[target_prefix + key[len(source_prefix):]] = value
                    break
        if not embeddings_state_dict:
            return [], []

        missing, unexpected = self.embeddings_processor.load_state_dict(
            embeddings_state_dict,
            strict=False,
            assign=getattr(self, "can_assign_sd", False),
        )
        missing = [
            key for key in missing
            if not key.startswith(("video_connector.", "feature_extractor.video_aggregate_embed."))
        ]
        return list(missing), list(unexpected)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        constant = 3.0 if comfy.model_management.should_use_bf16(device) else 6.0
        gemma_pairs = token_weight_pairs.get("gemma3_12b", [])
        if not gemma_pairs:
            return 0
        padding = min(sum(1 for _ in itertools.takewhile(lambda token: token[0] == 0, batch)) for batch in gemma_pairs)
        tokens = sum(len(batch) for batch in gemma_pairs) - padding
        return max(tokens, 642) * constant * 1024 * 1024


def te(dtype_llama=None, llama_quantization_metadata=None, audio_components_path=None):
    if audio_components_path is None:
        raise ValueError("DramaBox CLIP requires the Dramabox audio-components checkpoint.")

    class DramaBoxTE(DramaBoxTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            options = model_options
            if llama_quantization_metadata is not None:
                options = options.copy()
                options["llama_quantization_metadata"] = llama_quantization_metadata
            super().__init__(
                dtype_llama=dtype_llama,
                device=device,
                dtype=dtype_llama if dtype_llama is not None else dtype,
                model_options=options,
                audio_components_path=audio_components_path,
            )

    return DramaBoxTE


def split_state_dicts(state_dicts):
    gemma_indices = [
        index for index, state_dict in enumerate(state_dicts)
        if "model.layers.47.self_attn.q_norm.weight" in state_dict
    ]
    audio_indices = [
        index for index, state_dict in enumerate(state_dicts)
        if "text_embedding_projection.audio_aggregate_embed.weight" in state_dict
    ]
    if len(gemma_indices) != 1 or len(audio_indices) != 1 or gemma_indices[0] == audio_indices[0]:
        raise ValueError(
            "DramaBox CLIP requires one Gemma 3 12B text encoder and one Dramabox audio-components checkpoint."
        )
    return gemma_indices[0], audio_indices[0]
