import json

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
from comfy.quant_ops import BlockWiseFP8E4M3Layout, QuantizedTensor, TensorCoreFP8E4M3Layout, TensorCoreNVFP4Layout
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


@pytest.fixture
def tiny_clip():
    text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, vocab_size=64, moe_intermediate_size=16,
                num_experts=4, num_experts_per_tok=2, max_position_embeddings=16)
    audio = Qwen3OmniMoeAudioEncoderConfig(d_model=16, encoder_attention_heads=2, encoder_ffn_dim=32,
                                          encoder_layers=1, downsample_hidden_size=2, output_dim=32).to_dict()
    vision = Qwen3OmniMoeVisionEncoderConfig(hidden_size=32, intermediate_size=48, depth=1, num_heads=4,
        out_hidden_size=32, num_position_embeddings=16, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, deepstack_visual_indexes=[0]).to_dict()
    config = {'thinker_config': {'text_config': text, 'audio_config': audio, 'vision_config': vision},
              'processor_config': {'image_processor': {'patch_size': 4, 'merge_size': 2,
                                                     'size': {'shortest_edge': 64, 'longest_edge': 1024}},
                                   'feature_extractor': {'sampling_rate': 16000, 'hop_length': 160, 'n_fft': 400}}}
    return omni.te(config, dtype_llama=torch.float32)(device='cpu')


def test_automatic_budget_and_overflow_precede_generation_workspace(tiny_clip, monkeypatch):
    clip = tiny_clip.qwen3_omni
    clip.execution_device = torch.device('cpu')
    monkeypatch.setattr(clip, 'process_tokens', lambda *a: (torch.zeros(1, 4, 32), None, None, []))
    budgets = []
    def generate(*args, **kwargs):
        budgets.append(args[2])
        return [151645]
    monkeypatch.setattr(clip.transformer, 'generate', generate)
    arguments = dict(tokens=[[(1, 1.)] * 4], do_sample=False, max_length=0, temperature=1., top_k=0,
                     top_p=1., min_p=0., repetition_penalty=1., seed=17, mtp=False)
    assert clip.generate(**arguments) == [151645]
    assert budgets == [12]
    with pytest.raises(ValueError, match='after media expansion'):
        clip.generate(**{**arguments, 'max_length': 13})
    monkeypatch.setattr(clip, 'process_tokens', lambda *a: (torch.zeros(1, 16, 32), None, None, []))
    with pytest.raises(ValueError, match='no room'):
        clip.generate(**arguments)
    assert budgets == [12]


def test_admission_counts_expanded_images_and_incomplete_artifacts(tiny_clip):
    assert tiny_clip.qwen3_omni.transformer.model.layers[0].self_attn.q_proj._nvfp4_input_passes == 3
    row = [(1, 1.)] * 4
    assert tiny_clip.generation_memory_estimation_function({'qwen3_omni': [row]}, 0, torch.device('cpu')) > 0
    image_row = [(1, 1.), ({'type': 'image', 'data': torch.zeros(1, 16, 16, 3)}, 1.), (2, 1.), (3, 1.)]
    with pytest.raises(ValueError, match='after media expansion'):
        tiny_clip.generation_memory_estimation_function({'qwen3_omni': [image_row]}, 13, torch.device('cpu'))
    with pytest.raises(ValueError, match='Incomplete Qwen3-Omni artifact'):
        tiny_clip.qwen3_omni.load_sd({})


def test_explicit_omni_type_rejects_an_unrecognized_package():
    from comfy.sd import CLIPType, load_text_encoder_state_dicts
    with pytest.raises(ValueError, match='converted native package'):
        load_text_encoder_state_dicts([{}], clip_type=CLIPType.QWEN3_OMNI)


@pytest.fixture
def native_state(tiny_clip):
    model = tiny_clip.qwen3_omni.transformer
    state = {name: torch.randn_like(value) for name, value in model.named_parameters()}
    for name, module in model.named_modules():
        if isinstance(module, comfy.ops.MixedPrecisionOp):
            state[name + '.weight'] = torch.randn(module._orig_shape)
    return state


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_native_package_accepts_dense_projections(tiny_clip, native_state, dtype):
    state = {name: value.to(dtype) for name, value in native_state.items()}
    missing, unexpected = tiny_clip.qwen3_omni.load_sd(state)
    assert not missing and not unexpected
    model = tiny_clip.qwen3_omni.transformer.model
    inputs = torch.randn(2, 32)
    expected = F.linear(inputs, state['model.layers.0.self_attn.q_proj.weight'].float())
    torch.testing.assert_close(model.layers[0].self_attn.q_proj(inputs), expected)
    expected = F.linear(inputs, state['model.layers.0.mlp.experts.gate_up_proj.weight'][1].float())
    torch.testing.assert_close(model.layers[0].mlp.experts.gate_up_proj.expert_linear(inputs, 1), expected)


@pytest.fixture
def mixed_state(native_state):
    for name in ('model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.experts.gate_up_proj'):
        weight = native_state[name + '.weight']
        values = weight.unbind() if weight.ndim == 3 else [weight]
        parts = [TensorCoreNVFP4Layout.get_plain_tensors(QuantizedTensor.from_float(value.bfloat16(), 'TensorCoreNVFP4Layout'))
                 for value in values]
        for index, suffix in enumerate(('.weight', '.weight_scale_2', '.weight_scale')):
            native_state[name + suffix] = torch.stack([part[index] for part in parts]) if weight.ndim == 3 else parts[0][index]
        descriptor = {'format': 'nvfp4'}
        if weight.ndim == 3:
            descriptor['num_experts'] = weight.shape[0]
        native_state[name + '.comfy_quant'] = torch.tensor(list(json.dumps(descriptor).encode()), dtype=torch.uint8)
    return native_state


def test_native_package_mixes_dense_and_nvfp4(tiny_clip, mixed_state):
    missing, unexpected = tiny_clip.qwen3_omni.load_sd(mixed_state)
    assert not missing and not unexpected
    model = tiny_clip.qwen3_omni.transformer.model
    inputs = torch.randn(2, 32)
    for module in (model.layers[0].self_attn.q_proj, model.layers[0].mlp.experts.gate_up_proj):
        assert module.quant_format == 'nvfp4'
        bank = module.weight.ndim == 3
        weight = module.expert_weight(1).dequantize() if bank else module.weight.dequantize()
        actual = module.expert_linear(inputs, 1) if bank else module(inputs)
        expected = F.linear(inputs, weight)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('fault', ['descriptor', 'format', 'weight_shape', 'weight_dtype', 'blocks_shape',
                                  'blocks_dtype', 'scale_shape', 'scale_dtype', 'experts', 'dense_shape', 'orphan_scales'])
def test_native_package_rejects_invalid_projection_before_loading(tiny_clip, mixed_state, fault, monkeypatch):
    name = 'model.layers.0.mlp.experts.gate_up_proj'
    if fault == 'descriptor':
        del mixed_state[name + '.comfy_quant']
    elif fault in ('format', 'experts'):
        descriptor = {'format': 'nvfp4', 'num_experts': 3} if fault == 'experts' else {'format': 'float8_e4m3fn'}
        mixed_state[name + '.comfy_quant'] = torch.tensor(list(json.dumps(descriptor).encode()), dtype=torch.uint8)
    elif fault == 'dense_shape':
        mixed_state['model.lm_head.weight'] = torch.zeros(1, 32)
    elif fault == 'orphan_scales':
        mixed_state['model.lm_head.weight_scale'] = torch.tensor(1.)
    else:
        suffix = {'weight': '.weight', 'blocks': '.weight_scale', 'scale': '.weight_scale_2'}[fault.split('_')[0]]
        value = mixed_state[name + suffix]
        mixed_state[name + suffix] = value.unsqueeze(0) if fault.endswith('shape') else value.to(torch.int32)
    monkeypatch.setattr(omni.sd1_clip.SDClipModel, 'load_sd', lambda *a: pytest.fail('invalid package reached state loading'))
    with pytest.raises(ValueError, match='Qwen3-Omni'):
        tiny_clip.qwen3_omni.load_sd(mixed_state)


@pytest.fixture(params=['float8_e4m3fn', 'float8_e4m3fn_block128'])
def fp8_state(native_state, request):
    layout = BlockWiseFP8E4M3Layout if request.param == 'float8_e4m3fn_block128' else TensorCoreFP8E4M3Layout
    for name in ('model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.experts.gate_up_proj'):
        weight = native_state[name + '.weight']
        values = weight.unbind() if weight.ndim == 3 else [weight]
        parts = [layout.quantize(value, scale='recalculate') for value in values]
        native_state[name + '.weight'] = torch.stack([part[0] for part in parts]) if weight.ndim == 3 else parts[0][0]
        native_state[name + '.weight_scale'] = torch.stack([part[1].scale for part in parts]) if weight.ndim == 3 else parts[0][1].scale
        descriptor = {'format': request.param, 'full_precision_matrix_mult': True}
        if weight.ndim == 3:
            descriptor['num_experts'] = weight.shape[0]
        native_state[name + '.comfy_quant'] = torch.tensor(list(json.dumps(descriptor).encode()), dtype=torch.uint8)
    return native_state


def test_native_fp8_dense_and_expert_projection(tiny_clip, fp8_state):
    missing, unexpected = tiny_clip.qwen3_omni.load_sd(fp8_state)
    assert not missing and not unexpected
    model = tiny_clip.qwen3_omni.transformer.model
    inputs = torch.randn(2, 32)
    for name, module, expert in [('model.layers.0.self_attn.q_proj', model.layers[0].self_attn.q_proj, None),
                                ('model.layers.0.mlp.experts.gate_up_proj', model.layers[0].mlp.experts.gate_up_proj, 1)]:
        weight, scale = fp8_state[name + '.weight'], fp8_state[name + '.weight_scale']
        if expert is not None:
            weight, scale = weight[expert], scale[expert]
        expected = F.linear(inputs, weight.float() * scale)
        actual = module(inputs) if expert is None else module.expert_linear(inputs, expert)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert module.quant_format == json.loads(fp8_state[name + '.comfy_quant'].numpy().tobytes())['format']


@pytest.mark.parametrize('fault', ['weight_shape', 'weight_dtype', 'scale_shape', 'scale_dtype', 'experts', 'extra_scale', 'missing_scale'])
def test_native_fp8_layout_validation(tiny_clip, fp8_state, fault, monkeypatch):
    name = 'model.layers.0.mlp.experts.gate_up_proj'
    if fault in ('weight_shape', 'scale_shape'):
        key = name + ('.weight' if fault == 'weight_shape' else '.weight_scale')
        fp8_state[key] = fp8_state[key].unsqueeze(0)
    elif fault in ('weight_dtype', 'scale_dtype'):
        key = name + ('.weight' if fault == 'weight_dtype' else '.weight_scale')
        fp8_state[key] = fp8_state[key].double() if fault == 'scale_dtype' else fp8_state[key].bfloat16()
    elif fault == 'experts':
        descriptor = json.loads(fp8_state[name + '.comfy_quant'].numpy().tobytes())
        descriptor['num_experts'] = 2
        fp8_state[name + '.comfy_quant'] = torch.tensor(list(json.dumps(descriptor).encode()), dtype=torch.uint8)
    elif fault == 'extra_scale':
        fp8_state[name + '.weight_scale_2'] = torch.ones(4)
    else:
        del fp8_state[name + '.weight_scale']
    monkeypatch.setattr(omni.sd1_clip.SDClipModel, 'load_sd', lambda *a: pytest.fail('invalid FP8 package reached state loading'))
    with pytest.raises(ValueError, match='Qwen3-Omni'):
        tiny_clip.qwen3_omni.load_sd(fp8_state)
