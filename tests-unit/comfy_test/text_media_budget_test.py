import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.text_generation import ContextOverflowError, select_context
from comfy.text_generation_media import prepare_generation_tokens, sample_video, timestamp
from comfy.text_encoders.chat import count_chat_tokens
from comfy.text_encoders.gemma4 import Gemma4Base, Gemma4_12B_Config, Gemma4_E2B_Config, prepare_chat_image
from comfy.text_encoders.llama import BaseGenerate
from comfy.text_encoders.qwen35 import Qwen35, Qwen35Config, Qwen35TEModel
from comfy.text_encoders.qwen_vl import budget_image_size
from comfy.sd import CLIP
from comfy_extras.nodes_textgen import TextGenerate


@pytest.mark.parametrize('source_fps,fps,expected', [(24, 4, [0, 6, 12, 18]), (30, 4, [0, 7, 15, 22]), (24, 48, list(range(24)))])
def test_sampling_uses_source_seconds_without_duplicate_frames(source_fps, fps, expected):
    video = torch.arange(int(source_fps)).reshape(-1, 1, 1, 1)
    frames, times = sample_video(video, source_fps, fps)
    assert frames.flatten().tolist() == expected
    assert times == [i / source_fps for i in expected]


def test_fractional_source_rate_and_short_clips():
    video = torch.zeros(61, 1, 1, 3)
    frames, times = sample_video(video, 29.97, 4)
    assert len(frames) == 9
    assert times[-1] == 59 / 29.97
    assert len(sample_video(video[:1], 24, 0.1)[0]) == 1
    assert timestamp(0.25) == '00:00.25'
    assert timestamp(61.5) == '01:01.5'
    assert timestamp(59.9999) == '01:00'
    assert len(sample_video(video, 29.97, 1000)[0]) == 61


@pytest.mark.parametrize('source,fps', [(0, 1), (24, 0), (math.inf, 1), (24, math.nan)])
def test_invalid_frame_rates_fail_before_indexing(source, fps):
    with pytest.raises(ValueError, match='finite positive'):
        sample_video(torch.zeros(1, 1, 1, 3), source, fps)


def image_count(tokens):
    return count_chat_tokens(tokens, lambda embed: embed['data'].shape[1] * embed['data'].shape[2] // (32 * 32))


def geometry(embed, budget):
    return budget_image_size(*embed['data'].shape[1:3], budget)


def images(*sizes):
    return {'qwen': [[(7, 1.0), *[({'type': 'image', 'data': torch.rand(1, h, w, 3)}, 1.0) for h, w in sizes], (8, 1.0)]]}


def test_resize_preserves_output_and_reaches_largest_shared_ceiling():
    source = images((640, 960), (320, 320))
    pixels = source['qwen'][0][1][0]['data'].clone()
    result, info = prepare_generation_tokens(source, 2048, 512, 100, image_count, geometry, True)
    assert info['input_tokens'] + info['output_allowance'] <= 512
    assert info['output_allowance'] == 100
    assert torch.equal(source['qwen'][0][1][0]['data'], pixels)
    assert 'expected_length' not in source['qwen'][0][1][0]
    for token, _ in result['qwen'][0][1:-1]:
        assert isinstance(token['data'], torch.Tensor)
        assert token['data'].shape[1] % 32 == token['data'].shape[2] % 32 == 0
    ceiling = info['image_token_ceiling']
    larger = copy.deepcopy(result)
    for index, entry in enumerate(source['qwen'][0]):
        if isinstance(entry[0], dict):
            h, w = geometry(entry[0], ceiling + 1)
            larger['qwen'][0][index][0]['data'] = SimpleNamespace(shape=(1, h, w, 3))
    assert image_count(larger) + 100 > 512


def test_available_memory_can_limit_media_before_context():
    memory = lambda tokens, count, output: (count + output) * 1000 + count * count
    source = images((960, 960))
    large, large_info = prepare_generation_tokens(source, 32768, 0, 128, image_count, geometry, True,
                                                 4_000_000, memory)
    small, small_info = prepare_generation_tokens(source, 32768, 0, 128, image_count, geometry, True,
                                                 400_000, memory)
    assert small_info['input_tokens'] < large_info['input_tokens']
    assert small_info['estimated_working_bytes'] <= 400_000
    assert small_info['output_allowance'] == large_info['output_allowance'] == 128
    assert small_info['vram_context_limit'] < large_info['vram_context_limit'] < 32768
    assert small['qwen'][0][1][0]['data'].numel() < large['qwen'][0][1][0]['data'].numel()


def test_fixed_resolution_cannot_be_made_cheaper_by_resizing():
    count = lambda tokens: count_chat_tokens(tokens, lambda embed: 256)
    fixed = lambda embed, budget: embed['data'].shape[1:3]
    with pytest.raises(ContextOverflowError, match='minimum image sizes'):
        prepare_generation_tokens(images((32, 32)), 256, 0, 32, count, fixed, True)


def test_auto_output_stays_within_both_budgets():
    source = {'qwen': [[(1, 1.0), (2, 1.0)]]}
    _, info = prepare_generation_tokens(source, 2048, 1024, 0, image_count, geometry, True,
                                        1000, lambda tokens, count, output: (count + output) * 10)
    assert info['input_tokens'] + info['output_allowance'] == 100


@pytest.mark.parametrize('size', [(960, 1728), (1728, 960)])
def test_short_edge_floor_respects_model_grid_and_orientation(size):
    result, info = prepare_generation_tokens(images(size), 2048, 400, 64, image_count, geometry,
                                             True, min_short_edge=300)
    h, w = result['qwen'][0][1][0]['data'].shape[1:3]
    assert min(h, w) >= 300 and h % 32 == w % 32 == 0
    assert (h < w) == (size[0] < size[1])
    assert info['input_tokens'] + info['output_allowance'] <= 400
    assert info['output_allowance'] == 64 and info['min_short_edge'] == 300


@pytest.mark.parametrize('context,memory', [(128, None), (2048, 170_000)])
def test_floor_failure_happens_before_pixel_resize(context, memory, monkeypatch):
    def unexpected_resize(*args, **kwargs):
        pytest.fail('A request below its quality floor must fail before materializing pixels')
    monkeypatch.setattr('comfy.text_generation_media.F.interpolate', unexpected_resize)
    with pytest.raises(ContextOverflowError, match='min_short_edge=256px'):
        prepare_generation_tokens(images((960, 1728)), 2048, context, 64, image_count, geometry,
            True, memory, lambda tokens, count, output: (count + output) * 1000,
            min_short_edge=256)


def test_tiny_source_requires_explicit_floor_override():
    source = images((32, 32))
    with pytest.raises(ContextOverflowError, match='minimum short edge cannot be reached'):
        prepare_generation_tokens(source, 2048, 0, 64, image_count, geometry, True, min_short_edge=256)
    selected, info = prepare_generation_tokens(source, 2048, 0, 64, image_count, geometry, True, min_short_edge=0)
    assert min(selected['qwen'][0][1][0]['data'].shape[1:3]) < 256
    assert info['min_short_edge'] == 0
    unchanged, _ = prepare_generation_tokens(source, 2048, 0, 64, image_count, geometry, False, min_short_edge=256)
    assert unchanged['qwen'][0][1][0]['data'] is source['qwen'][0][1][0]['data']


def test_gemma4_uses_original_pixels_and_native_geometry_ceiling():
    embed = prepare_chat_image(torch.rand(1, 1080, 1920, 3), 70)
    model = SimpleNamespace(model=SimpleNamespace(config=Gemma4_12B_Config()))
    shape = Gemma4Base.image_geometry(model, embed, 100_000)
    assert shape == (1200, 2112)
    assert shape[0] * shape[1] // (48 * 48) <= 1120
    assert shape[0] > embed['data'].shape[1]
    native = images((1, 1))
    native['qwen'][0][1] = (embed, 1.0)
    result, info = prepare_generation_tokens(native, 4096, 2048, 128,
        lambda tokens: count_chat_tokens(tokens, lambda value: Gemma4Base.image_token_length(model, value)),
        lambda value, budget: Gemma4Base.image_geometry(model, value, budget), True)
    actual = result['qwen'][0][1][0]
    assert tuple(actual['data'].shape[1:3]) == shape
    assert 'source' not in actual and 'max_soft_tokens' not in actual
    with pytest.raises(ContextOverflowError, match='minimum short edge cannot be reached'):
        prepare_generation_tokens(native, 4096, 2048, 128,
            lambda tokens: count_chat_tokens(tokens, lambda value: Gemma4Base.image_token_length(model, value)),
            lambda value, budget: Gemma4Base.image_geometry(model, value, budget), True, min_short_edge=4096)


@pytest.mark.parametrize('config', [Gemma4_12B_Config(), Gemma4_E2B_Config()])
def test_gemma4_cache_estimate_matches_real_allocations(config):
    model = SimpleNamespace(model=SimpleNamespace(config=config, fixed_kv=True))
    capacity = 1300
    caches = Gemma4Base.init_kv_cache(model, 1, capacity, torch.device('meta'), torch.bfloat16)
    actual = sum((cache.key.numel() + cache.value.numel()) * 2 for cache in caches if hasattr(cache, 'key'))
    assert Gemma4Base.generation_cache_memory(model, capacity, 2) == actual


def test_hybrid_cache_estimate_accounts_for_linear_state_and_mtp():
    config = Qwen35Config()
    model = SimpleNamespace(model=SimpleNamespace(config=config), mtp=object())
    ordinary = Qwen35.generation_cache_memory(model, 1024, 2, False)
    assert ordinary < BaseGenerate.generation_cache_memory(model, 1024, 2)
    assert Qwen35.generation_cache_memory(model, 1024, 2, True) > ordinary


def test_history_can_drop_old_turns_when_minimum_media_cannot_fit():
    messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old'}, {'role': 'user', 'content': 'current'}]
    def prepare(tokens):
        if len(tokens[0]) > 1:
            raise ContextOverflowError('minimum cannot fit')
        return tokens, {'output_allowance': 16, 'memory_budget_bytes': 100}
    _, info = select_context(messages, lambda active: [[(1, 1.0)] * len(active)], 64, 0, 16,
                             'drop_oldest_turns', prepare_tokens=prepare)
    assert info['excluded_message_ranges'] == [[0, 2]]
    assert info['output_allowance'] == 16 and len(messages) == 3


def test_oom_retry_preserves_output_and_publishes_only_success(monkeypatch):
    class Clip:
        calls = 0
        budgets = []
        def tokenize_video(self, *args, **kwargs):
            return images((512, 512)), {'sampled_frames': 4}
        def prepare_generation_tokens(self, tokens, budget, output, auto, memory_budget=None, **kwargs):
            assert kwargs['min_short_edge'] == 256
            self.budgets.append((output, memory_budget))
            size = 512 if memory_budget is None else 256
            return images((size, size)), {'version': 1, 'input_tokens': 4, 'output_allowance': output,
                'memory_budget_bytes': memory_budget or 1000, 'estimated_working_bytes': memory_budget or 1000,
                'image_sizes': [[size, size]], 'excluded_message_ranges': []}
        def generate(self, tokens, **kwargs):
            self.calls += 1
            assert kwargs['max_length'] == 128
            if self.calls == 1:
                raise torch.cuda.OutOfMemoryError('fixture')
            return [42]
        def decode(self, ids):
            return 'caption'
    monkeypatch.setattr('comfy.model_management.soft_empty_cache', lambda: None)
    clip = Clip()
    assert next(item for item in TextGenerate.define_schema().inputs if item.id == 'auto_resize').default is True
    assert next(item for item in TextGenerate.define_schema().inputs if item.id == 'min_short_edge').default == 256
    result = TextGenerate.execute(clip, 'describe', 128, {'sampling_mode': 'off'}, video=torch.zeros(24, 1, 1, 3)).result
    assert clip.calls == 2 and clip.budgets == [(128, None), (128, 500)]
    assert result[2] == '' and json.loads(result[3])['oom_retries'] == 1


def test_oom_replanning_stops_at_floor_without_a_second_generation(monkeypatch):
    class Clip:
        calls = 0
        def tokenize_video(self, *args, **kwargs):
            return images((512, 512)), {'sampled_frames': 1}
        def prepare_generation_tokens(self, tokens, budget, output, auto, memory_budget=None, **kwargs):
            return prepare_generation_tokens(tokens, 2048, budget, output, image_count, geometry, auto,
                200_000 if memory_budget is None else memory_budget,
                lambda value, count, reserve: (count + reserve) * 1000,
                min_short_edge=kwargs['min_short_edge'])
        def generate(self, tokens, **kwargs):
            self.calls += 1
            assert min(tokens['qwen'][0][1][0]['data'].shape[1:3]) >= 256
            raise torch.cuda.OutOfMemoryError('fixture')
    monkeypatch.setattr('comfy.model_management.soft_empty_cache', lambda: None)
    clip = Clip()
    with pytest.raises(ContextOverflowError, match='min_short_edge=256px'):
        TextGenerate.execute(clip, 'describe', 128, {'sampling_mode': 'off'}, video=torch.zeros(24, 1, 1, 3))
    assert clip.calls == 1


def test_clip_memory_planning_measures_after_managed_loading(monkeypatch):
    events = []
    def prepare(tokens, cap, output, resize, **kwargs):
        events.append(('prepare', kwargs))
        return tokens, {}
    patcher = SimpleNamespace(load_device=torch.device('cuda:0'), model_patches_models=lambda: [])
    clip = SimpleNamespace(cond_stage_model=SimpleNamespace(prepare_generation_tokens=prepare),
                           patcher=patcher, load_model=lambda tokens: events.append(('load', tokens)))
    monkeypatch.setattr('comfy.model_management.LoadedModel', lambda value: value)
    monkeypatch.setattr('comfy.model_management.get_total_memory', lambda device: 10000)
    monkeypatch.setattr('comfy.model_management.free_memory', lambda total, device, keep_loaded: events.append(('free', keep_loaded)))
    monkeypatch.setattr('comfy.model_management.get_free_memory', lambda device: 4000)
    monkeypatch.setattr('comfy.model_management.minimum_inference_memory', lambda: 1000)
    _, info = CLIP.prepare_generation_tokens(clip, [], 2048, 128, True, mtp=False, min_short_edge=384)
    assert [event[0] for event in events] == ['load', 'free', 'prepare']
    assert events[1][1] == [patcher]
    assert events[2][1]['memory_budget'] == 3000
    assert events[2][1]['mtp'] is False
    assert events[2][1]['min_short_edge'] == 384
    assert info == {'memory_device': 'cuda:0', 'memory_kind': 'vram'}


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('can_reserve', [False, True])
def test_text_planning_reserves_working_memory_before_loading(monkeypatch, device, can_reserve):
    required, reserve, available = 11576251744, 1000000000, 10580000000
    tokens = {'qwen': [[(index, 1.0) for index in range(7723)]]}
    requests = []

    def estimate(value, output, device):
        assert value is tokens and output == 20000
        return required

    def load(value, memory_required=0):
        nonlocal available
        requests.append(memory_required)
        if can_reserve and memory_required >= required + reserve:
            available = required + reserve

    def prepare(value, cap, output, resize, **kwargs):
        return prepare_generation_tokens(value, 32768, cap, output,
                                         lambda value: len(value['qwen'][0]), None, resize,
                                         kwargs['memory_budget'], lambda *args: required)

    patcher = SimpleNamespace(load_device=torch.device(device), model_patches_models=lambda: [])
    clip = SimpleNamespace(cond_stage_model=SimpleNamespace(prepare_generation_tokens=prepare,
                           generation_memory_estimation_function=estimate), patcher=patcher, load_model=load)
    monkeypatch.setattr('comfy.model_management.LoadedModel', lambda value: value)
    monkeypatch.setattr('comfy.model_management.get_total_memory', lambda device: 32000000000)
    monkeypatch.setattr('comfy.model_management.free_memory', lambda *args, **kwargs: None)
    monkeypatch.setattr('comfy.model_management.get_free_memory', lambda device: available)
    monkeypatch.setattr('comfy.model_management.minimum_inference_memory', lambda: reserve)
    if not can_reserve:
        with pytest.raises(ContextOverflowError, match='available budget'):
            CLIP.prepare_generation_tokens(clip, tokens, 32768, 20000, True)
        assert requests == [required + reserve]
        assert len(tokens['qwen'][0]) == 7723
        return
    selected, info = CLIP.prepare_generation_tokens(clip, tokens, 32768, 20000, True)
    assert selected == tokens
    assert info['input_tokens'] == 7723 and info['output_allowance'] == 20000
    assert info['estimated_working_bytes'] == required
    assert requests == [required + reserve]
    assert info['memory_kind'] == ('ram' if device == 'cpu' else 'vram')


@pytest.mark.parametrize('kind,output', [('image', 128), ('audio', 128), ('text', 0)])
def test_initial_reservation_preserves_media_sizing_and_automatic_output(monkeypatch, kind, output):
    tokens = {'qwen': [[(1 if kind == 'text' else {'type': kind}, 1.0)]]}
    events = []
    def estimate(*args, **kwargs):
        pytest.fail('Unplanned media or automatic output must not reserve their full raw estimate')
    def prepare(value, cap, output, resize, **kwargs):
        events.append(('prepare', kwargs['memory_budget']))
        return value, {}
    patcher = SimpleNamespace(load_device=torch.device('cpu'), model_patches_models=lambda: [])
    clip = SimpleNamespace(cond_stage_model=SimpleNamespace(prepare_generation_tokens=prepare,
                           generation_memory_estimation_function=estimate), patcher=patcher,
                           load_model=lambda value: events.append(('load', value)))
    monkeypatch.setattr('comfy.model_management.LoadedModel', lambda value: value)
    monkeypatch.setattr('comfy.model_management.get_total_memory', lambda device: 10000)
    monkeypatch.setattr('comfy.model_management.free_memory', lambda *args, **kwargs: None)
    monkeypatch.setattr('comfy.model_management.get_free_memory', lambda device: 4000)
    monkeypatch.setattr('comfy.model_management.minimum_inference_memory', lambda: 1000)
    CLIP.prepare_generation_tokens(clip, tokens, 2048, output, True, memory_budget=2000)
    assert events == [('load', tokens), ('prepare', 2000)]


@pytest.mark.parametrize('output,expected', [(20000, 20000), (0, 32768 - 7723)])
def test_qwen_generation_reservation_uses_expanded_tokens(output, expected):
    calls = []
    tokens = {'qwen': [[(1, 1.0), ({'type': 'image'}, 1.0)]]}
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768)),
                            count_chat_tokens=lambda value: 7723,
                            generation_memory=lambda *args: calls.append(args) or 1234)
    model.resolve_max_length = lambda length, count: BaseGenerate.resolve_max_length(model, length, count)
    encoder = SimpleNamespace(clip='qwen', qwen=SimpleNamespace(transformer=model))
    assert Qwen35TEModel.generation_memory_estimation_function(encoder, tokens, output, torch.device('cpu')) == 1234
    assert calls == [(tokens, 7723, expected, torch.device('cpu'))]


def test_video_as_images_preserves_frame_association_and_fractional_timestamps():
    captured = []
    def tokenize(messages, thinking):
        captured.extend(messages)
        return [[(part, 1.0) for part in messages[-1]['content'] if part['type'] == 'image']]
    clip = SimpleNamespace(tokenizer=object(), tokenize_chat=tokenize)
    video = torch.arange(24).reshape(24, 1, 1, 1).expand(-1, 1, 1, 3)
    tokens, info = CLIP.tokenize_video(clip, 'describe', video, 24, 4, skip_template=False, thinking=False)
    content = captured[0]['content']
    assert len(tokens[0]) == info['sampled_frames'] == 4
    assert [part['data'][0, 0, 0, 0].item() for part in content if part['type'] == 'image'] == [0, 6, 12, 18]
    assert [part['text'] for part in content if part['type'] == 'text'][2] == '\nFrame at 00:00.25: '
