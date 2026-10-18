from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.glm_ocr import GlmOCR, GlmOCRTEModel, process_image, image_geometry
from transformers import GlmOcrConfig, GlmOcrForConditionalGeneration, Glm46VImageProcessor
from transformers.cache_utils import DynamicCache
from scripts.convert_glm_ocr import expected_shapes, mapped_name, inspect_checkpoint
from safetensors.torch import save_file


def models():
    torch.manual_seed(41)
    text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=128, mrope_section=[2, 3, 3])
    vision = dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=4, out_hidden_size=32)
    config = GlmOcrConfig(text_config={**{k: v for k, v in text.items() if k != 'mrope_section'},
                                     'rms_norm_eps': 1e-5,
                                     'rope_parameters': dict(rope_type='default', rope_theta=10000, mrope_section=[2, 3, 3])},
                          vision_config={**vision, 'rms_norm_eps': 1e-5})
    reference = GlmOcrForConditionalGeneration(config).eval()
    native = GlmOCR({'text': text, 'vision': vision}, torch.float32, 'cpu', manual_cast)
    native.load_state_dict({mapped_name(k): v for k, v in reference.state_dict().items()}, strict=True)
    return native, reference


def test_vision_and_prefill_and_cached_decoder_match_reference():
    native, reference = models()
    pixels, grid = process_image(torch.rand(1, 51, 71, 3))
    expected_vision = reference.model.visual(pixels.flatten(1), torch.tensor([[1, *grid]])).pooler_output
    actual_vision = native.visual(pixels, grid)
    torch.testing.assert_close(actual_vision, expected_vision, rtol=2e-4, atol=2e-5)
    embeds = native.model.embed_tokens(torch.tensor([[7, 8, 9, 10, 11]]))
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 1, 2, 3], [0, 1, 2, 1, 3]])
    cache = native.model.init_kv_cache(1, 12, 'cpu', torch.float32)
    expected_cache = DynamicCache(config=reference.config.text_config)
    actual, _, cache = native.model(None, embeds=embeds, past_key_values=cache, position_ids=positions)
    expected = reference.model.language_model(inputs_embeds=embeds, position_ids=positions[:, None], past_key_values=expected_cache).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    for position in (4, 5):
        embeds = native.model.embed_tokens(torch.tensor([[12]]))
        actual, _, cache = native.model(None, embeds=embeds, past_key_values=cache, position_ids=torch.tensor([[position]]))
        expected = reference.model.language_model(inputs_embeds=embeds, position_ids=torch.tensor([[position]]), past_key_values=expected_cache).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_padded_pixels_match_official_processor_without_resize():
    image = torch.rand(1, 51, 71, 3)
    image = (image * 255).round() / 255
    actual, grid = process_image(image)
    array = (image[0].numpy() * 255).astype('uint8')
    padded = Image.new('RGB', (84, 56), 'white')
    padded.paste(Image.fromarray(array))
    expected = Glm46VImageProcessor()(padded, do_resize=False, return_tensors='pt')
    assert expected.image_grid_thw.tolist() == [[1, *grid]]
    torch.testing.assert_close(actual.flatten(1), expected.pixel_values, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('h,w', [(77,84), (720,576), (1,1), (1400,1600)])
def test_no_enlargement(h,w):
    height, width, ph, pw = image_geometry(h,w)
    assert height <= h and width <= w
    assert height * width <= 1_000_000
    assert 0 <= ph-height < 28 and 0 <= pw-width < 28
    assert ph % 28 == pw % 28 == 0


def test_admission_includes_budget_and_expanded_pixels():
    native, _ = models()
    wrapper = SimpleNamespace(glm_ocr=SimpleNamespace(transformer=native))
    tokens = {'glm_ocr': [[(7, 1), ({'type':'image','data':torch.rand(1,720,576,3)},1), (8,1)]]}
    estimate = GlmOCRTEModel.generation_memory_estimation_function
    short = estimate(wrapper, tokens, 32, 'cpu')
    longer = estimate(wrapper, tokens, 2048, 'cpu')
    assert longer - short == (2048-32)*2*2*2*16*4
    with pytest.raises(ValueError, match='context'):
        estimate(wrapper, tokens, 131072, 'cpu')


def test_exact_checkpoint_inventory():
    original, mtp = expected_shapes()
    assert len(original) == 510 and len(mtp) == 16
    assert not set(original) & set(mtp)
    assert len({mapped_name(k) for k in original}) == len(original)


def test_checkpoint_ranges_and_dtype(tmp_path):
    path = tmp_path / 'model.safetensors'
    save_file({'x': torch.ones(2,3,dtype=torch.bfloat16)}, path)
    assert set(inspect_checkpoint(path, {'x': [2,3]})) == {'x'}
    with pytest.raises(ValueError, match='inventory'):
        inspect_checkpoint(path, {'z': [2,3]})
    with pytest.raises(ValueError, match='shape'):
        inspect_checkpoint(path, {'x': [3,2]})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(Exception):
        inspect_checkpoint(path, {'x': [2,3]})


def test_budget_reaches_admission_before_generation(monkeypatch):
    from comfy.sd import CLIP
    import comfy.model_management
    import contextlib
    events = []
    stage = SimpleNamespace(reset_clip_options=lambda: None, set_clip_options=lambda _: None,
                            generation_memory_estimation_function=lambda tokens, budget, device: budget * 100,
                            memory_estimation_function=lambda tokens, device: 300,
                            generate=lambda *a, **kw: events.append('generate') or [1])
    clip = CLIP(no_init=True)
    clip.cond_stage_model = stage
    patch = object()
    clip.patcher = SimpleNamespace(load_device=torch.device('cpu'), model_patches_models=lambda: [patch])
    monkeypatch.setattr(comfy.model_management, 'load_models_gpu', lambda models, memory_required: events.append(memory_required))
    monkeypatch.setattr(comfy.model_management, 'LoadedModel', lambda model: model)
    def free(required, device, keep_loaded):
        assert required == 4200 and device == torch.device('cpu')
        assert keep_loaded == [clip.patcher, patch]
        events.append('release_idle')
    monkeypatch.setattr(comfy.model_management, 'free_memory', free)
    monkeypatch.setattr('comfy.ops.use_quantized_matmul', lambda *a: contextlib.nullcontext())
    assert clip.generate({}, max_length=42, do_sample=False) == [1]
    assert events == [4200, 'generate']
    events.clear()
    monkeypatch.setattr(comfy.model_management, 'get_free_memory', lambda device: 100)
    with pytest.raises(RuntimeError, match='needs 4200 free bytes'):
        clip.generate({}, max_length=42, do_sample=False)
    assert events == [4200, 'release_idle']  # Still fails before generation/cache allocation.
    events.clear()
    memory = iter([100, 5000])
    monkeypatch.setattr(comfy.model_management, 'get_free_memory', lambda device: next(memory))
    assert clip.generate({}, max_length=42, do_sample=False) == [1]
    assert events == [4200, 'release_idle', 'generate']
    del stage.generation_memory_estimation_function
    events.clear()
    clip.generate({}, max_length=42, do_sample=False)
    assert events == [300, 'generate']


def test_raw_decoded_output_keeps_whitespace():
    from comfy_extras.nodes_textgen import TextGenerate
    clip = SimpleNamespace(tokenize=lambda *a, **kw: [], generate=lambda *a, **kw: [1], decode=lambda _: '  text\n\n')
    output = TextGenerate.execute(clip, 'Text Recognition:', 64, {'sampling_mode':'off'}, image=torch.zeros(1,28,28,3))
    assert output.result[0] == 'text'
    assert output.result[4] == '  text\n\n'
