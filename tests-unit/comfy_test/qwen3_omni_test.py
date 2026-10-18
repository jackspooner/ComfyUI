import pytest
import torch
from torch.nn import functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig, Qwen3OmniMoeTextConfig, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeThinkerConfig)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeThinkerForConditionalGeneration)

from comfy.cli_args import args
args.cpu = True
import comfy.ops
from comfy.text_encoders import qwen3_omni as omni


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


@pytest.mark.parametrize('with_deepstack', [False, True])
def test_text_routing_prefill_and_cached_decoding(with_deepstack):
    torch.manual_seed(17)
    config = Qwen3OmniMoeTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64, moe_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, rope_parameters={'rope_type': 'default', 'rope_theta': 1000000.,
                                                             'mrope_section': [1, 1, 2], 'interleaved': True})
    config._attn_implementation = 'sdpa'
    config._experts_implementation = 'eager'
    reference = Qwen3OmniMoeThinkerTextModel(config).eval()
    native_config = omni.OmniConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64, moe_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, rope_dims=[1, 1, 2])
    native = omni.TextModel(native_config, 'cpu', torch.float32, comfy.ops.mixed_precision_ops({}, torch.float32))
    state = {key + '.weight' if key.endswith(('experts.gate_up_proj', 'experts.down_proj')) else key: value
             for key, value in reference.state_dict().items()}
    state['lm_head.weight'] = torch.randn(64, 32)
    missing, unexpected = native.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    hidden = torch.randn(1, 6, 32)
    positions = torch.arange(6).expand(3, -1).clone()
    positions[1, 2:4] = 2
    mask = torch.tensor([[False, False, True, True, False, False]])
    deepstack = [torch.randn(2, 32) for _ in range(3)] if with_deepstack else None
    upstream = reference(inputs_embeds=hidden, position_ids=positions.unsqueeze(1), use_cache=True,
                         visual_pos_masks=mask, deepstack_visual_embeds=deepstack)
    actual, _, cache = native(None, embeds=hidden.clone(), position_ids=positions,
                              visual_pos_masks=mask, deepstack_embeds=deepstack,
                              past_key_values=native.init_kv_cache(1, 16, torch.device('cpu'), torch.float32))
    torch.testing.assert_close(actual, upstream.last_hidden_state, atol=1e-5, rtol=1e-4)
    for index in range(3):
        hidden = torch.randn(1, 1, 32)
        upstream = reference(inputs_embeds=hidden, position_ids=torch.full((3, 1, 1), 6 + index),
                             past_key_values=upstream.past_key_values, use_cache=True)
        actual, _, cache = native(None, embeds=hidden.clone(), position_ids=torch.full((3, 1), 6 + index), past_key_values=cache)
        torch.testing.assert_close(actual, upstream.last_hidden_state, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('frames', [1, 8, 99, 100, 101, 199, 800, 803])
def test_audio_chunks_features_and_attention_windows(frames):
    torch.manual_seed(19)
    config = Qwen3OmniMoeAudioEncoderConfig(d_model=16, encoder_attention_heads=2, encoder_ffn_dim=32,
                                          encoder_layers=2, downsample_hidden_size=2, output_dim=32)
    config._attn_implementation = 'sdpa'
    reference = Qwen3OmniMoeAudioEncoder(config).eval()
    native = omni.AudioEncoder(config.to_dict(), 'cpu', torch.float32, comfy.ops.manual_cast)
    native.load_state_dict(reference.state_dict())
    features = torch.randn(128, frames)
    expected = reference(features, feature_lens=torch.tensor([frames])).last_hidden_state
    actual = native(features)
    assert actual.shape[0] == omni.audio_output_length(frames)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_vision_non_square_batch_deepstack_and_padded_projection():
    torch.manual_seed(23)
    config = Qwen3OmniMoeVisionEncoderConfig(hidden_size=32, intermediate_size=48, depth=3, num_heads=4,
        out_hidden_size=32, num_position_embeddings=16, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[0, 1, 2])
    config._attn_implementation = 'sdpa'
    reference = Qwen3OmniMoeVisionEncoder(config).eval()
    native = omni.VisionEncoder(config.to_dict(), 'cpu', torch.float32, comfy.ops.manual_cast)
    state = {}
    for key, value in reference.state_dict().items():
        target = key.replace('merger_list.', 'deepstack_merger_list.')
        if target.startswith(('merger.', 'deepstack_merger_list.')):
            target = target.replace('.ln_q.', '.norm.').replace('.mlp.0.', '.linear_fc1.').replace('.mlp.2.', '.linear_fc2.')
        if '.mlp.linear_fc2.weight' in target:
            value = F.pad(value, (0, 16))
        state[target] = value
    native.load_state_dict(state)
    grids = [[1, 4, 6], [1, 6, 4]]
    pixels = torch.randn(48, 3 * 2 * 4 * 4)
    expected = reference(pixels, grid_thw=torch.tensor(grids))
    actual, deepstack = native(pixels, grids)
    torch.testing.assert_close(actual, expected.pooler_output, atol=1e-5, rtol=1e-4)
    for actual, expected in zip(deepstack, expected.deepstack_features):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_image_and_audio_modality_positions_match_upstream():
    config = Qwen3OmniMoeThinkerConfig(audio_token_id=omni.AUDIO_TOKEN, audio_start_token_id=151669,
                                     vision_start_token_id=151652)
    with torch.device('meta'):
        reference = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    ids, info, grids = [1, 2], [], [[1, 4, 4], [1, 4, 6]]
    for grid in grids:
        ids.append(151652)
        size = grid[1] * grid[2] // 4
        info.append({'type': 'image', 'index': len(ids), 'size': size, 'extra': {'grid': grid}})
        ids.extend([omni.IMAGE_TOKEN] * size)
        ids.extend([151653, 3])
    ids.extend([151669, *([omni.AUDIO_TOKEN] * 5), 151670, 4])
    inputs = torch.tensor([ids])
    positions, _ = reference.get_rope_index(inputs, image_grid_thw=torch.tensor(grids),
        attention_mask=torch.ones_like(inputs), audio_seqlens=torch.tensor([33]))
    actual = omni.media_positions(info, len(ids), torch.device('cpu'))
    torch.testing.assert_close(actual, positions[:, 0], atol=0, rtol=0)


def test_detector_requires_every_signature_key_and_asset():
    from comfy.sd import detect_te_model, TEModel
    signature = {'audio_tower.conv_out.weight': 1280, 'visual.merger.linear_fc2.weight': 2048,
                 'visual.deepstack_merger_list.0.norm.weight': 4608,
                 'model.layers.47.self_attn.q_norm.weight': 128, 'model.norm.weight': 2048,
                 'model.layers.0.mlp.experts.gate_up_proj.weight': 128}
    state = {key: torch.empty(size, 1, device='meta') for key, size in signature.items()}
    state.update({key: torch.empty(1, dtype=torch.uint8) for key in omni.ASSET_KEYS})
    assert detect_te_model(state) == TEModel.QWEN3_OMNI
    for key in state:
        assert detect_te_model({name: value for name, value in state.items() if name != key}) != TEModel.QWEN3_OMNI
    for key in signature:
        changed = {**state, key: torch.empty(1, device='meta')}
        assert detect_te_model(changed) != TEModel.QWEN3_OMNI
