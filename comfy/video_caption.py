"""Patcher-managed native audio-visual captioners with request-local KV/grounding state."""
import json
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.storage
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen25_3BConfig, Qwen3VL_8BConfig, apply_rope
from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel, QWEN3VL_VISION, QWEN3VL_VISION_COMMON
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
from comfy.video_caption_audio import CaptionAudioEncoder
from comfy.video_caption_grounding import PadGrounding, SalmonnPositions
from comfy.video_caption_media import prepare_video, prepare_audio, media_prompt


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) if isinstance(item, dict) else item for key, item in value.items()})


class CaptionDecoder(Llama2_):
    def step(self, hidden, positions, cache, mask=None, deepstack=None, visual_mask=None, capture=False):
        past = 0 if cache is None else cache[0][2]
        length = hidden.shape[1]
        if mask is None and length > 1:
            keys = torch.arange(past + length, device=hidden.device)
            queries = torch.arange(past, past + length, device=hidden.device)
            mask = hidden.new_zeros((1, 1, length, past + length)).masked_fill(keys[None, :] > queries[:, None], torch.finfo(hidden.dtype).min)
        # Upstream builders return [axes, batch, tokens]; native Qwen rotary
        # preparation consumes [axes, tokens] for this single-video decoder.
        if positions.ndim == 3:
            positions = positions[:, 0]
        frequencies = self.compute_freqs_cis(positions.to(hidden.device), hidden.device)
        attention = optimized_attention_for_device(hidden.device, mask=mask is not None, small_input=True)
        snapshots = {}
        next_cache = []
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), hidden.device, {'prefetch_dynamic_vbars': True})
        for index, layer in enumerate(self.layers):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if capture and index in (6, len(self.layers) - 1):
                snapshots[index] = hidden.clone()
            def core():
                nonlocal hidden
                if capture and cache is None and index == 6:
                    normalized = layer.input_layernorm(hidden)
                    attn = layer.self_attn
                    batch, count, _ = normalized.shape
                    query = attn.q_proj(normalized).view(batch, count, attn.num_heads, attn.head_dim).transpose(1, 2)
                    key = attn.k_proj(normalized).view(batch, count, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                    query, key = apply_rope(query, key, freqs_cis=frequencies)
                    key = key.repeat_interleave(attn.num_heads // attn.num_kv_heads, dim=1)
                    scores = query[:, :, -1:] @ key.transpose(-1, -2) / attn.head_dim ** 0.5
                    if mask is not None:
                        scores = scores + mask[:, :, -1:]
                    snapshots['attention'] = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype).mean(1)[0, 0, :, None]
                hidden, current = layer(hidden, mask, frequencies, attention, [] if cache is None else cache[index])
                next_cache.append(current)
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='block')
            if deepstack is not None and index < len(deepstack):
                hidden[visual_mask] = hidden[visual_mask] + deepstack[index].to(hidden)
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='block')
        return self.norm(hidden), next_cache, snapshots


class CaptionNetwork(nn.Module):
    def __init__(self, config, family, dtype, device, operations):
        super().__init__()
        self.family = family
        self.config = namespace(config)
        self.dtype = dtype
        if family == 'salmonn2_pro_8b':
            text_config = Qwen3VL_8BConfig()
            vision_config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION['qwen3vl_8b'], 'out_hidden_size': 4096}
            self.visual = Qwen3VLVisionModel(vision_config, dtype=dtype, device=device, ops=operations)
        elif family == 'padcaptioner_3b':
            text_config = Qwen25_3BConfig()
            text_config.rope_dims = [16, 24, 24]
            self.visual = Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=2048, device=device, dtype=dtype, ops=operations)
            for name in ('frm_head', 'evi_head', 'bnd_frm_head', 'bnd_evi_head'):
                output = 4096 if name == 'bnd_evi_head' else 2048
                self.add_module(name, nn.Sequential(operations.LayerNorm(2048, dtype=dtype, device=device),
                    operations.Linear(2048, 1024, dtype=dtype, device=device), nn.GELU(),
                    operations.Linear(1024, output, dtype=dtype, device=device)))
        else:
            raise ValueError('Unsupported captioner architecture')
        self.model = CaptionDecoder(text_config, device=device, dtype=dtype, ops=operations)
        self.audio = CaptionAudioEncoder(config['audio_config'], dtype, device, operations)

    def get_dynamic_vram__units(self):
        return [*self.audio.layers, *self.model.layers, *self.visual.blocks], []

    def logits(self, hidden):
        if self.family == 'padcaptioner_3b':
            return F.linear(hidden, comfy.ops.cast_to_input(self.model.embed_tokens.weight, hidden))
        return self.model.lm_head(hidden)

    def embed_media(self, ids, patches, grid, audio_features):
        hidden = self.model.embed_tokens(ids, out_dtype=self.dtype)
        visual = self.visual(patches.to(hidden.device, dtype=torch.float32), grid.to(hidden.device))
        deepstack = None
        if self.family == 'salmonn2_pro_8b':
            visual, deepstack = visual
        mask = ids == self.config.video_token_id
        hidden[mask] = visual.to(hidden)
        if audio_features is not None:
            features = []
            for chunk in audio_features:
                comfy.model_management.throw_exception_if_processing_interrupted()
                features.append(self.audio(chunk[None].to(hidden)).flatten(0, 1))
            hidden[ids == self.config.audio_token_id] = torch.cat(features).to(hidden)
        return hidden, deepstack, mask


class PadSession(PadGrounding):
    def __init__(self, network, ids, timestep_ranges, timestep_duration):
        self.network = network
        self.config = network.config
        self.frm_head, self.evi_head = network.frm_head, network.evi_head
        self.bnd_frm_head, self.bnd_evi_head = network.bnd_frm_head, network.bnd_evi_head
        self.ids = ids
        self.duration = timestep_duration
        self.video_ranges, self.audio_ranges = timestep_ranges
        self.start = torch.where(ids[0] == self.config.vision_start_token_id)[0][:1]
        self.end = torch.where(ids[0] == self.config.vision_end_token_id)[0][-1:]
        self.serial_evi_tgts = [[]]
        self._branch_prefix_visible = None

    def initialize(self, snapshots):
        start, end = int(self.start[0]) + 1, int(self.end[0])
        self.cache_img_state = [snapshots[6][0, start:end]]
        self.cache_img_state_score = [snapshots['attention'][start:end]]
        self.cache_timestep_token_ranges = [self._get_timestep_token_ranges_from_per_timestep(0, self.start, self.end,
            self.ids, len(self.video_ranges), [self.video_ranges], [self.audio_ranges])]
        hidden = snapshots[len(self.network.model.layers) - 1]
        args = (0, self.start, self.end, self.ids)
        self.frame_features = self._compute_vision_info_per_timestep(*args, self.frm_head(hidden)[0, start:end],
            len(self.video_ranges), [self.video_ranges], [self.audio_ranges])
        self.boundary_features = self._compute_vision_info_per_timestep(*args, self.bnd_frm_head(hidden)[0, start:end],
            len(self.video_ranges), [self.video_ranges], [self.audio_ranges])

    def readout(self, hidden):
        query = self.evi_head(hidden)
        scores = torch.sigmoid(query @ self.frame_features.t() / query.shape[-1] ** 0.5)
        if self.config.grounding_coverage_mask and self.serial_evi_tgts[0]:
            masked = scores.clone()
            for intervals in self.serial_evi_tgts[0]:
                for start, end in intervals:
                    masked[:, max(0, int(start)):int(end) + 1] = 0
            if masked.max() > 0:
                scores = masked
        intervals = self.find_and_merge(scores)
        if self.config.grounding_snap_window:
            a, b = self._boundary_sims(hidden[0], self.boundary_features)
            intervals = self._snap_readout(intervals, a, b)
        return intervals or [[0, 0]]


class VideoCaptioner:
    supports_audio = True

    def __init__(self, state_dict, metadata):
        if metadata.get('conversion_version') != '1' or metadata.get('architecture') not in ('salmonn2_pro_8b', 'padcaptioner_3b'):
            raise ValueError('Use scripts/convert_video_captioner.py to prepare a complete native captioner')
        self.family = metadata['architecture']
        self.load_device = comfy.model_management.text_encoder_device()
        offload = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.bfloat16 if comfy.model_management.supports_dtype(self.load_device, torch.bfloat16) else torch.float32
        tokenizer = state_dict.pop('tokenizer_json')
        self.tokenizer = Tokenizer.from_str(tokenizer.numpy().tobytes().decode())
        config = json.loads(metadata['captioner_config'])
        self.network = CaptionNetwork(config, self.family, self.dtype, offload, comfy.ops.manual_cast).eval()
        self.network.load_state_dict(state_dict, strict=True, assign=True)
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.network, load_device=self.load_device, offload_device=offload,
            fast_disk=comfy.storage.state_dict_fast_disk(state_dict))

    def caption(self, images, source_fps, audio, prompt, fps, max_pixels, max_tokens, max_events):
        patch = 16 if self.family == 'salmonn2_pro_8b' else 14
        patches, grid, timestamps = prepare_video(images, source_fps, fps, patch, max_pixels)
        features = prepare_audio(audio)
        audio_count = 0 if features is None else len(features) * 60
        ids, _ = media_prompt(self.tokenizer, self.family, prompt, grid, timestamps, audio_count, min(fps, source_fps))
        limit = self.network.model.config.max_position_embeddings
        output_reserve = max_tokens * max_events if self.family == 'padcaptioner_3b' else max_tokens
        if ids.shape[1] + output_reserve > limit:
            raise ValueError(f'Caption input/output exceeds {limit} tokens; reduce frames, resolution or output budget')
        # KV and prefill workspace are separate from model weight residency.
        config = self.network.model.config
        memory = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * (ids.shape[1] + output_reserve) * 2
        memory += max(ids.shape[1] * config.hidden_size * 64, int(grid.prod()) ** 2 * 4)
        comfy.model_management.load_models_gpu([self.patcher], memory_required=memory)
        ids = ids.to(self.load_device)
        hidden, deepstack, visual_mask = self.network.embed_media(ids, patches, grid, features)
        if self.family == 'salmonn2_pro_8b':
            position_owner = SalmonnPositions()
            position_owner.config = self.network.config
            positions, delta = position_owner.get_rope_index(ids, video_grid_thw=grid)
            output = self._salmonn(ids, hidden, positions, int(delta[0]), deepstack, visual_mask, max_tokens)
            return dict(text=self.tokenizer.decode(output, skip_special_tokens=True).strip(), events=[], generated_tokens=len(output))
        return self._pad(ids, hidden, grid, min(fps, source_fps), max_tokens, max_events, audio_count)

    def _salmonn(self, ids, hidden, positions, delta, deepstack, visual_mask, max_tokens):
        network = self.network
        cache = None
        output = []
        for step in range(max_tokens):
            hidden, cache, _ = network.model.step(hidden, positions, cache, deepstack=deepstack, visual_mask=visual_mask)
            token = int(network.logits(hidden[:, -1:]).argmax(-1).item())
            if token == network.config.eos_token_id:
                break
            output.append(token)
            hidden = network.model.embed_tokens(torch.tensor([[token]], device=ids.device), out_dtype=network.dtype)
            positions = torch.full((3, 1, 1), ids.shape[1] + step + delta, device=ids.device, dtype=torch.long)
            deepstack = None
        return output

    def _pad(self, ids, hidden, grid, fps, max_tokens, max_events, audio_count):
        network = self.network
        time, height, width = grid[0].tolist()
        per_frame = height * width // 4
        video_ranges = [[(i * per_frame, (i + 1) * per_frame)] for i in range(time)]
        audio_ranges = []
        for index in range(time):
            start = min(audio_count, math.floor(index * 4 / fps))
            end = min(audio_count, math.ceil((index + 1) * 4 / fps))
            audio_ranges.append([(start, end)] if end > start else [])
        session = PadSession(network, ids, (video_ranges, audio_ranges), 2 / fps)
        # The upstream position builder mixes scalar/grid arithmetic with audio
        # indices. Build its small integer layout on CPU before native transfer.
        positions, delta = session.get_rope_index(ids.cpu(), video_grid_thw=grid.cpu(), audio_lengths=[audio_count] if audio_count else None,
            second_per_grid_ts=[2 / fps])
        prefix_length = ids.shape[1]
        mask = hidden.new_full((1, 1, prefix_length, prefix_length), torch.finfo(hidden.dtype).min).triu_(1)
        start, end = int(session.start[0]) + 1, int(session.end[0])
        mask[:, :, start:end, start:end] = 0
        output, cache, snapshots = network.model.step(hidden, positions, None, mask=mask, capture=True)
        session.initialize(snapshots)
        planned = []
        for index in range(max_events + 1):
            logits = network.logits(output[:, -1])
            token = network.config.evi_end_token_id if index == max_events else (
                network.config.evi_token_id if logits[0, network.config.evi_token_id] > logits[0, network.config.evi_end_token_id] else network.config.evi_end_token_id)
            if token == network.config.evi_end_token_id:
                break
            interval = session.readout(snapshots[len(network.model.layers) - 1][:, -1])
            planned.append(interval)
            session.serial_evi_tgts[0].append(interval)
            embedding = network.model.embed_tokens(torch.tensor([[token]], device=ids.device), out_dtype=network.dtype)
            embedding[0, 0] += session._aggregate_interval_feature(0, interval)
            positions = torch.full((3, 1, 1), cache[0][2] + int(delta[0]), device=ids.device, dtype=torch.long)
            output, cache, snapshots = network.model.step(embedding, positions, cache, capture=True)
        if not planned:
            return dict(text='', events=[], generated_tokens=0)
        branches = len(planned)
        prefix = cache[0][2]
        session._branch_prefix_visible = session._build_branch_prefix_visibility(ids, branches, prefix, ids.device)
        embedding = network.model.embed_tokens(torch.full((1, branches), network.config.evi_token_id, device=ids.device), out_dtype=network.dtype)
        for index, interval in enumerate(planned):
            embedding[0, index] += session._aggregate_interval_feature(0, interval)
        tokens = [[] for _ in planned]
        finished = [False] * branches
        for step in range(max_tokens):
            positions = torch.full((3, 1, branches), prefix + step + int(delta[0]), device=ids.device, dtype=torch.long)
            mask = session._branch_visibility_mask(branches, cache[0][2] + branches, prefix, hidden.dtype, ids.device)
            output, cache, _ = network.model.step(embedding, positions, cache, mask=mask)
            logits = network.logits(output[0])
            banned = [network.config.evi_token_id, network.config.evi_end_token_id, network.config.shift_token_id,
                network.config.audio_token_id, network.config.video_token_id, network.config.image_token_id,
                network.config.vision_start_token_id, network.config.vision_end_token_id]
            logits[:, banned] = torch.finfo(logits.dtype).min
            selected = logits.argmax(-1)
            for index, token in enumerate(selected.tolist()):
                if not finished[index]:
                    if token == network.config.eos_token_id:
                        finished[index] = True
                    else:
                        tokens[index].append(token)
                else:
                    selected[index] = network.config.eos_token_id
            if all(finished):
                break
            embedding = network.model.embed_tokens(selected[None], out_dtype=network.dtype)
        events = [dict(intervals=[[a * session.duration, (b + 1) * session.duration] for a, b in intervals],
            text=self.tokenizer.decode(body, skip_special_tokens=True).strip(), truncated=not finished[index])
            for index, (intervals, body) in enumerate(zip(planned, tokens))]
        return dict(text='\n'.join(f"{event['intervals']}: {event['text']}" for event in events), events=events,
                    generated_tokens=sum(map(len, tokens)), planned_events=branches)
