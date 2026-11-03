# SPDX-License-Identifier: Apache-2.0
"""Audio-visual positions and PadCaptioner grounding, adapted from upstream.

Copyright 2025-2026 Tsinghua University, ByteDance and PadCaptioner authors.
Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0); see docs/video-captioners.md
for exact revisions and attribution. Only inference helpers are retained.
"""
import torch
from torch.nn import functional as F

class PadGrounding:

    def get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, audio_lengths=None, second_per_grid_ts=None, attention_mask=None):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (audio_lengths is not None and video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            video_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = (0, 0)
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_videos = video_nums
                for _ in range(video_nums):
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    t, h, w = (video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2])
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size)
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * 2
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    video_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    audio_len = audio_lengths[video_index]
                    time_index_audio = torch.arange(audio_len, device=input_ids.device)
                    w_index_audio = time_index_audio
                    h_index_audio = torch.zeros_like(time_index_audio)
                    audio_pos = torch.stack([time_index_audio, h_index_audio, w_index_audio]) + st_idx + text_len
                    audio_visual_pos = torch.zeros_like(torch.cat((video_pos, audio_pos), dim=1))
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                    audio_visual_pos[:, input_ids[ed:st] == audio_token_id] = audio_pos
                    audio_visual_pos[:, input_ids[ed:st] == video_token_id] = video_pos
                    llm_pos_ids_list.append(audio_visual_pos)
                    video_index += 1
                    remain_videos -= 1
                if st < len(input_tokens):
                    if len(llm_pos_ids_list) > 0:
                        last_time_max = llm_pos_ids_list[-1][0].max()
                        st_idx = last_time_max + 1
                    else:
                        st_idx = 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        elif input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            image_index, video_index = (0, 0)
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = (0, 0)
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = (image_nums, video_nums)
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2])
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2])
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size)
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * 2
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        elif input_ids is not None and audio_lengths is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            audio_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                audio_nums = 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = (vision_tokens == audio_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_audios = audio_nums
                for _ in range(audio_nums):
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    ed = ed_audio
                    llm_grid_t = audio_lengths[audio_index]
                    llm_grid_h = 1
                    llm_grid_w = 1
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    time_index = torch.arange(llm_grid_t, device=input_ids.device)
                    h_index = time_index
                    w_index = torch.zeros_like(time_index)
                    audio_pos = torch.stack([time_index, h_index, w_index]) + st_idx + text_len
                    llm_pos_ids_list.append(audio_pos)
                    st = ed + llm_grid_t
                    audio_index += 1
                    remain_audios -= 1
                if st < len(input_tokens):
                    if len(llm_pos_ids_list) > 0:
                        last_time_max = llm_pos_ids_list[-1][0].max()
                        st_idx = last_time_max + 1
                    else:
                        st_idx = 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
            return (position_ids, mrope_position_deltas)

    def _boundary_sims(self, hidden_b, bnd_info):
        q = self.bnd_evi_head(hidden_b.unsqueeze(0))
        qs, qe = q.split(q.shape[-1] // 2, dim=-1)
        scale = qs.shape[-1] ** 0.5
        bi = bnd_info.to(qs.dtype)
        return ((qs @ bi.t() / scale).squeeze(0).float(), (qe @ bi.t() / scale).squeeze(0).float())

    def _snap_readout(self, pred, sim_s, sim_e):
        win = int(getattr(self.config, 'grounding_snap_window', 0) or 0)
        if not pred or win <= 0 or sim_s is None:
            return pred
        st, en = (int(pred[0][0]), int(pred[0][1]))
        T = sim_s.shape[0]
        a, c = (max(0, st - win), min(T - 1, st + win))
        st2 = a + int(torch.argmax(sim_s[a:c + 1]).item())
        a, c = (max(0, en - win), min(T - 1, en + win))
        en2 = a + int(torch.argmax(sim_e[a:c + 1]).item())
        if not sim_s[st2] > sim_s[st]:
            st2 = st
        if not sim_e[en2] > sim_e[en]:
            en2 = en
        if st2 > en2:
            return pred
        return [[st2, en2]]

    def find_and_merge(self, matrix):
        if matrix.dim() == 2:
            matrix = matrix.squeeze(0)
        similarity_Threshold = getattr(self.config, 'grounding_threshold', 0.8)
        threshold = torch.max(matrix) * similarity_Threshold
        salient = (matrix >= threshold).nonzero(as_tuple=True)[0].tolist()
        if not salient:
            return []
        runs = [[salient[0], salient[0]]]
        for t in salient[1:]:
            if t == runs[-1][1] + 1:
                runs[-1][1] = t
            else:
                runs.append([t, t])
        merge_gap = getattr(self.config, 'grounding_merge_gap', 2)
        merged = [runs[0]]
        for st, en in runs[1:]:
            if st - merged[-1][1] - 1 <= merge_gap:
                merged[-1][1] = en
            else:
                merged.append([st, en])
        if len(merged) == 1:
            best = merged[0]
        else:
            best = max(merged, key=lambda seg: matrix[seg[0]:seg[1] + 1].sum().item())
        return [[best[0], best[1]]]

    def _compute_vision_info_per_timestep(self, b, vision_start_positions, vision_end_positions, input_ids, frm_vision_tokens, num_timesteps, video_token_indices_per_timestep, audio_token_indices_per_timestep):
        start_pos = vision_start_positions[b].item() + 1
        end_pos = vision_end_positions[b].item()
        vision_input_ids = input_ids[b, start_pos:end_pos]
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        video_mask = vision_input_ids == video_token_id
        audio_mask = vision_input_ids == audio_token_id
        device = frm_vision_tokens.device
        dtype = frm_vision_tokens.dtype
        hidden_size = frm_vision_tokens.shape[1]
        zero_vec = torch.zeros(hidden_size, device=device, dtype=dtype)
        vid_per_ts = video_token_indices_per_timestep[b] if b < len(video_token_indices_per_timestep) else []
        aud_per_ts = audio_token_indices_per_timestep[b] if b < len(audio_token_indices_per_timestep) else []
        video_positions = torch.where(video_mask)[0]
        audio_positions = torch.where(audio_mask)[0]
        n_video = video_positions.numel()
        n_audio = audio_positions.numel()

        def ranges_to_vision_indices(ranges, positions, max_len):
            if not ranges:
                return None
            indices_list = []
            for r in ranges:
                if isinstance(r, (tuple, list)) and len(r) == 2 and (not isinstance(r[0], (tuple, list))):
                    ranges_one = [tuple(r)]
                else:
                    ranges_one = list(r) if r else []
                for s, e in ranges_one:
                    s = max(0, min(int(s), max_len))
                    e = max(s, min(int(e), max_len))
                    if s < e:
                        indices_list.append(positions[s:e])
            if not indices_list:
                return None
            return torch.cat(indices_list, dim=0)
        num_ts = max(num_timesteps, len(vid_per_ts), len(aud_per_ts))
        video_timestep_means = []
        audio_timestep_means = []
        for t in range(num_ts):
            v_ranges = vid_per_ts[t] if t < len(vid_per_ts) else []
            a_ranges = aud_per_ts[t] if t < len(aud_per_ts) else []
            vid_idx = ranges_to_vision_indices(v_ranges, video_positions, n_video)
            aud_idx = ranges_to_vision_indices(a_ranges, audio_positions, n_audio)
            video_mean = frm_vision_tokens[vid_idx].mean(dim=0) if vid_idx is not None and vid_idx.numel() > 0 else zero_vec
            audio_mean = frm_vision_tokens[aud_idx].mean(dim=0) if aud_idx is not None and aud_idx.numel() > 0 else zero_vec
            video_timestep_means.append(video_mean)
            audio_timestep_means.append(audio_mean)
        video_timestep_means_tensor = torch.stack(video_timestep_means, dim=0)
        audio_timestep_means_tensor = torch.stack(audio_timestep_means, dim=0)
        aw = getattr(self.config, 'audio_mean_weight', 1.0)
        info = video_timestep_means_tensor + aw * audio_timestep_means_tensor
        return info

    def _get_timestep_token_ranges_from_per_timestep(self, b, vision_start_positions, vision_end_positions, input_ids, num_timesteps, video_token_indices_per_timestep, audio_token_indices_per_timestep):
        start_pos = vision_start_positions[b].item() + 1
        end_pos = vision_end_positions[b].item()
        vision_input_ids = input_ids[b, start_pos:end_pos]
        video_mask = vision_input_ids == self.config.video_token_id
        audio_mask = vision_input_ids == self.config.audio_token_id
        video_positions = torch.where(video_mask)[0]
        audio_positions = torch.where(audio_mask)[0]
        n_video, n_audio = (video_positions.numel(), audio_positions.numel())

        def _ranges_to_vision_indices(ranges, positions, max_len):
            if not ranges:
                return None
            idx_list = []
            if isinstance(ranges, (tuple, list)) and len(ranges) == 2 and (not isinstance(ranges[0], (tuple, list))):
                ranges = [tuple(ranges)]
            for r in ranges:
                if isinstance(r, (tuple, list)) and len(r) == 2 and (not isinstance(r[0], (tuple, list))):
                    rs = [tuple(r)]
                else:
                    rs = list(r) if r else []
                for s, e in rs:
                    s = max(0, min(int(s), max_len))
                    e = max(s, min(int(e), max_len))
                    if s < e:
                        idx_list.extend(positions[s:e].tolist())
            if not idx_list:
                return None
            return torch.tensor(idx_list, dtype=torch.long, device=positions.device)
        vid_per_ts = video_token_indices_per_timestep[b] if b < len(video_token_indices_per_timestep) else []
        aud_per_ts = audio_token_indices_per_timestep[b] if b < len(audio_token_indices_per_timestep) else []
        out = []
        for t in range(num_timesteps):
            vr = vid_per_ts[t] if t < len(vid_per_ts) else []
            ar = aud_per_ts[t] if t < len(aud_per_ts) else []
            v_idx = _ranges_to_vision_indices(vr, video_positions, n_video)
            a_idx = _ranges_to_vision_indices(ar, audio_positions, n_audio)
            out.append((v_idx, a_idx))
        return out

    def _aggregate_interval_feature(self, b, intervals):
        cached = self.cache_timestep_token_ranges[b]
        cache_hidden = self.cache_img_state[b]
        cache_hidden_score = self.cache_img_state_score[b] if self.cache_img_state_score[b] is not None else None
        hidden_size = cache_hidden.shape[1]
        device, dtype = (cache_hidden.device, cache_hidden.dtype)
        video_list, audio_list, video_scores_list, audio_scores_list = ([], [], [], [])
        ts_list = [ts for seg_start, seg_end in intervals for ts in range(seg_start, seg_end + 1)]
        for ts in ts_list:
            if ts < 0 or ts >= len(cached):
                continue
            v_idx, a_idx = cached[ts]
            if v_idx is not None and v_idx.numel() > 0:
                video_list.append(cache_hidden[v_idx])
                if cache_hidden_score is not None:
                    video_scores_list.append(cache_hidden_score[v_idx])
            if a_idx is not None and a_idx.numel() > 0:
                audio_list.append(cache_hidden[a_idx])
                if cache_hidden_score is not None:
                    audio_scores_list.append(cache_hidden_score[a_idx])

        def _weighted_mean(feat_list, score_list):
            if not feat_list:
                return torch.zeros(hidden_size, device=device, dtype=dtype)
            feat = torch.cat(feat_list, dim=0)
            if cache_hidden_score is not None and len(score_list) > 0:
                score = torch.cat(score_list, dim=0)
                score = F.softmax(score.squeeze(-1), dim=0).unsqueeze(-1)
                return torch.sum(feat * score, dim=0).to(device=device, dtype=dtype)
            return torch.mean(feat, dim=0).to(device=device, dtype=dtype)
        aw = float(getattr(self.config, 'audio_mean_weight', 1.0))
        return _weighted_mean(video_list, video_scores_list) + aw * _weighted_mean(audio_list, audio_scores_list)

    def _build_branch_prefix_visibility(self, input_ids, K, prefix_len, device):
        margin = int(getattr(self.config, 'branch_interval_margin', 1))
        ids = input_ids[0]
        vs_pos = (ids == self.config.vision_start_token_id).nonzero(as_tuple=True)[0]
        ve_pos = (ids == self.config.vision_end_token_id).nonzero(as_tuple=True)[0]
        if vs_pos.numel() == 0 or ve_pos.numel() == 0:
            return None
        vs, ve = (vs_pos[0].item(), ve_pos[-1].item())
        cached = self.cache_timestep_token_ranges[0]
        if ve - (vs + 1) != self.cache_img_state[0].shape[0] or ve > prefix_len:
            return None
        visible = torch.ones(K, prefix_len, dtype=torch.bool, device=device)
        visible[:, vs + 1:ve] = False
        if getattr(self.config, 'branch_interval_audio_full', False):
            visible[:, vs + 1:ve] |= (ids[vs + 1:ve] == self.config.audio_token_id).unsqueeze(0)
        for j in range(K):
            intervals = self.serial_evi_tgts[0][j] if j < len(self.serial_evi_tgts[0]) else [[0, 0]]
            for seg_start, seg_end in intervals:
                lo, hi = (max(0, int(seg_start) - margin), min(len(cached) - 1, int(seg_end) + margin))
                for ts in range(lo, hi + 1):
                    v_idx, a_idx = cached[ts]
                    for idx in (v_idx, a_idx):
                        if idx is not None and idx.numel() > 0:
                            visible[j, vs + 1 + idx] = True
        return visible

    def _branch_visibility_mask(self, num_query, kv_len, prefix_len, dtype, device):
        K = num_query
        neg = torch.finfo(dtype).min
        mask = torch.full((1, 1, K, kv_len), neg, dtype=dtype, device=device)
        mask[:, :, :, :prefix_len] = 0.0
        pv = getattr(self, '_branch_prefix_visible', None)
        if pv is not None and tuple(pv.shape) == (K, prefix_len):
            mask[0, 0, :, :prefix_len] = torch.where(pv, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(neg, dtype=dtype, device=device))
        if kv_len > prefix_len:
            owners = torch.arange(kv_len - prefix_len, device=device) % K
            visible = owners.unsqueeze(0) == torch.arange(K, device=device).unsqueeze(1)
            mask[0, 0, :, prefix_len:] = torch.where(visible, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(neg, dtype=dtype, device=device))
        return mask

class SalmonnPositions:

    def get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            image_index, video_index = (0, 0)
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = (0, 0)
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = (image_nums, video_nums)
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2])
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2])
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size)
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
            return (position_ids, mrope_position_deltas)
