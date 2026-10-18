import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from comfy.cli_args import args
args.cpu = True
import comfy.memory_management
from comfy.ldm.paddleocr import Detector, Recognizer
from comfy.ops import manual_cast
from comfy.paddleocr import (decode_ctc, detection_regions, mini_box, plain_text, prepare_detector, load_model,
                            prepare_recognizer, read_assets, rectify, workspace_bytes, PaddleOCRModel)
from transformers import PPOCRV6MediumDetConfig, PPOCRV6MediumDetForObjectDetection, PPOCRV6SmallRecConfig, PPOCRV6SmallRecForTextRecognition
from transformers.models.pp_ocrv5_server_det import PPOCRV5ServerDetImageProcessor
from transformers.models.pp_ocrv6_small_rec import PPOCRV6SmallRecImageProcessor


def config(kind):
    detector = kind == 'detector'
    backbone = {'model_type': 'pp_lcnet_v4', 'stem_channels': [3, 8, 16], 'stem_type': 'large',
                'out_features': ['stage1', 'stage2', 'stage3', 'stage4'],
                'block_configs': [[[3, 16, 16, 1, True]], [[3, 16, 24, 2 if detector else 1, False]],
                                  [[3, 24, 32, 2 if detector else [2, 1], True]],
                                  [[3, 32, 40, 2 if detector else [2, 1], False]]]}
    if not detector:
        return {'model_type': 'pp_ocrv6_small_rec', 'backbone_config': backbone, 'hidden_size': 16,
                'num_attention_heads': 2, 'mlp_ratio': 2.0, 'depth': 2, 'head_out_channels': 24, 'conv_kernel_size': [1, 7]}
    branches = {'reduce_channel': [1, 1, 0], 'return_channel': [1, 1, 0]}
    for ratio, k in [('longratio', 7), ('midratio', 5), ('shortratio', 3)]:
        branches.update({f'symmetric_conv_long_{ratio}': [[k, k], [1, 1], [k // 2, k // 2]],
                         f'vertical_long_to_small_conv_{ratio}': [[k, 1], [1, 1], [k // 2, 0]],
                         f'horizontal_small_to_long_conv_{ratio}': [[1, k], [1, 1], [0, k // 2]]})
    return {'model_type': 'pp_ocrv6_medium_det', 'backbone_config': backbone, 'neck_out_channels': 16,
            'reduce_factor': 2, 'intraclass_block_number': 4, 'intraclass_block_config': branches,
            'scale_factor_list': [1, 2, 4, 8], 'kernel_list': [3, 2, 2]}


@pytest.mark.parametrize('kind', ['detector', 'recognizer'])
def test_native_math_matches_reference(kind):
    torch.manual_seed(11)
    cfg = config(kind)
    cls, refcls, cfgcls, shape = (Detector, PPOCRV6MediumDetForObjectDetection, PPOCRV6MediumDetConfig, (1, 3, 64, 96)) if kind == 'detector' else (Recognizer, PPOCRV6SmallRecForTextRecognition, PPOCRV6SmallRecConfig, (1, 3, 48, 128))
    reference = refcls(cfgcls(**copy.deepcopy(cfg))).eval()
    native = cls(cfg, manual_cast, device='cpu', dtype=torch.float32).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(shape)
    actual, expected = native(pixels), reference(pixels).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    if kind == 'recognizer':
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize('h,w', [(77, 84), (720, 576), (1, 1), (49, 19), (31, 401)])
def test_no_enlargement_and_padding(h, w):
    image = torch.randint(256, (3, h, w), dtype=torch.uint8)
    x, transform = prepare_detector(image, 'no_enlargement')
    assert transform['resized_size'] == [h, w]
    assert x.shape[-2] % 32 == x.shape[-1] % 32 == 0
    x, transform = prepare_recognizer(image, 'no_enlargement')
    rh, rw = transform['resized_size']
    assert rh <= h and rw <= w and x.shape[-2] == 48
    assert x.shape[-1] >= 320
    assert workspace_bytes('recognizer', x.shape) > x.numel() * 4


@pytest.mark.parametrize('h,w', [(77, 84), (720, 576), (31, 401)])
def test_official_preprocessing_matches(h, w):
    image = torch.randint(256, (3, h, w), dtype=torch.uint8)
    x, _ = prepare_detector(image, 'official')
    reference = PPOCRV5ServerDetImageProcessor(limit_side_len=736, limit_type='min')
    expected = reference(image, return_tensors='pt').pixel_values
    torch.testing.assert_close(x, expected, rtol=1e-5, atol=1e-6)
    x, _ = prepare_recognizer(image, 'official')
    expected = PPOCRV6SmallRecImageProcessor()(image, return_tensors='pt').pixel_values
    torch.testing.assert_close(x, expected, rtol=1e-5, atol=1e-6)


def test_ctc_blank_repeated_characters_and_score():
    probabilities = torch.tensor([[0.1, 0.9, 0], [0.2, 0.8, 0], [1, 0, 0], [0.1, 0.9, 0], [0.1, 0, 0.9]])
    text, score = decode_ctc(probabilities, ['', 'A', 'B'])
    assert text == 'AAB' and score == pytest.approx(0.9)
    assert decode_ctc(torch.tensor([[1., 0, 0]]), ['', 'A', 'B']) == ('', None)


def test_detection_thresholds_geometry_and_blank():
    prediction = np.zeros((64, 96), dtype=np.float32)
    prediction[10:25, 20:60] = 0.8
    transform = {'source_size': [32, 48]}
    regions = detection_regions(prediction, transform, 'official')
    reference = PPOCRV5ServerDetImageProcessor()
    expected, scores = reference._boxes_from_bitmap(prediction, prediction > 0.3, 48, 32, 0.6, 1.5, 3, 1000)
    np.testing.assert_array_equal(regions[0]['corners'], expected[0])
    assert regions[0]['detection_score'] == scores[0]
    assert detection_regions(prediction, transform, 'official', box_threshold=0.9) == []
    assert detection_regions(prediction * 0, transform, 'official') == []


def test_padding_does_not_create_regions_outside_source():
    prediction = np.ones((64, 96), dtype=np.float32)
    regions = detection_regions(prediction, {'source_size': [33, 65]}, 'no_enlargement')
    for region in regions:
        assert all(0 <= x <= 65 and 0 <= y <= 33 for x, y in region['corners'])


def test_rectification_rotates_tall_regions():
    image = np.full((60, 20, 3), 255, dtype=np.uint8)
    crop, transform = rectify(image, [[0, 0], [20, 0], [20, 60], [0, 60]])
    assert crop.shape == (3, 20, 60)
    assert transform['rotation_ccw_degrees'] == 90
    assert mini_box(np.array([[0, 0], [20, 0], [20, 60], [0, 60]], dtype=np.float32))[1] == 20


def test_asset_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'config.json').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        read_assets(root / 'model.safetensors', 'detector')


def test_node_checkpoint_path_containment(tmp_path, monkeypatch):
    import folder_paths
    from comfy_extras.nodes_paddleocr import checkpoint_path
    root = tmp_path / 'models'
    root.mkdir()
    outside = tmp_path / 'outside.safetensors'
    outside.touch()
    (root / 'escape.safetensors').symlink_to(outside)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'detection', ([str(root)], {'.safetensors'}))
    with pytest.raises(ValueError, match='within'):
        checkpoint_path('escape.safetensors')


def test_independent_images_and_empty_output(monkeypatch):
    model = PaddleOCRModel.__new__(PaddleOCRModel)
    model.identity = {}
    model.vocabulary = ['', 'A', 'B']
    model.recognizer = object()
    outputs = iter([torch.tensor([[[0., 1, 0]]]), torch.tensor([[[0., 0, 1]]])])
    monkeypatch.setattr(model, 'infer', lambda *args: next(outputs))
    result = model.recognize(torch.zeros(2, 15, 20, 3), operation='recognize_lines')
    assert [i['regions'][0]['text'] for i in result['images']] == ['A', 'B']
    assert plain_text(result) == '[Image 0]\nA\n\n[Image 1]\nB'
    json.dumps(result, allow_nan=False)


def test_workspace_is_admitted_before_execution(monkeypatch):
    from types import SimpleNamespace
    import comfy.model_management as mm
    events = []
    patcher = SimpleNamespace(load_device=torch.device('cpu'), model=lambda x: events.append('forward') or x)
    monkeypatch.setattr(mm, 'load_models_gpu', lambda models, **kw: events.append(kw['memory_required']))
    result = PaddleOCRModel.__new__(PaddleOCRModel).infer(patcher, torch.zeros(1, 3, 48, 320), 'recognizer')
    assert events == [workspace_bytes('recognizer', result.shape), 'forward']


@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('mutation', ['missing', 'extra', 'shape', 'dtype'])
def test_weight_inventory_is_strict(tmp_path, monkeypatch, mutation, lazy):
    import comfy.paddleocr as ocr
    from safetensors.torch import save_file
    cfg = config('recognizer')
    model = Recognizer(cfg, torch.nn, device='cpu', dtype=torch.float32)
    sd = {k: v for k, v in model.state_dict().items() if not k.endswith('.num_batches_tracked')}
    key = next(iter(sd))
    if mutation == 'missing':
        del sd[key]
    elif mutation == 'extra':
        sd['wrong'] = torch.ones(1)
    elif mutation == 'shape':
        sd[key] = torch.ones(1)
    else:
        sd[key] = sd[key].half()
    checkpoint = tmp_path / 'model.safetensors'
    save_file(sd, checkpoint)
    monkeypatch.setattr(ocr.comfy.utils, 'load_torch_file', lambda *a, **kw: sd.copy())
    monkeypatch.setattr(comfy.memory_management, 'aimdo_enabled', lazy)
    monkeypatch.setattr(ocr, 'read_assets', lambda *a: [cfg, {}])
    with pytest.raises(ValueError, match='inventory'):
        load_model(checkpoint, 'recognizer')


def test_lazy_weight_loading_accepts_complete_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    import comfy.paddleocr as ocr
    cfg = config('recognizer')
    reference = Recognizer(cfg, torch.nn, device='cpu', dtype=torch.float32)
    sd = {k: v for k, v in reference.state_dict().items() if not k.endswith('.num_batches_tracked')}
    checkpoint = tmp_path / 'model.safetensors'
    save_file(sd, checkpoint)
    monkeypatch.setattr(ocr.comfy.utils, 'load_torch_file', lambda *a, **kw: sd.copy())
    monkeypatch.setattr(comfy.memory_management, 'aimdo_enabled', True)
    monkeypatch.setattr(ocr, 'read_assets', lambda *a: [cfg, {}])
    monkeypatch.setitem(ocr.SOURCES, 'recognizer', {'weights_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    monkeypatch.setattr(ocr.comfy.model_patcher, 'CoreModelPatcher', lambda model, **kw: SimpleNamespace(model=model, is_dynamic=lambda: False))
    patcher, _, _ = load_model(checkpoint, 'recognizer')
    for key, value in sd.items():
        torch.testing.assert_close(patcher.model.state_dict()[key], value)


def test_fp32_recognizer_with_kitchen_attention(monkeypatch):
    import comfy.ldm.paddleocr as native
    from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
    monkeypatch.setattr(native, 'optimized_attention', attention_comfy_kitchen_int8)
    test_native_math_matches_reference('recognizer')


def test_cancellation_before_model_execution(monkeypatch):
    from types import SimpleNamespace
    import comfy.model_management as mm
    events = []
    def interrupt():
        raise mm.InterruptProcessingException()
    patcher = SimpleNamespace(load_device=torch.device('cpu'), model=lambda x: events.append('forward'))
    monkeypatch.setattr(mm, 'load_models_gpu', lambda *a, **kw: None)
    monkeypatch.setattr(mm, 'throw_exception_if_processing_interrupted', interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        PaddleOCRModel.__new__(PaddleOCRModel).infer(patcher, torch.zeros(1, 3, 48, 320), 'recognizer')
    assert events == []


def test_insufficient_gpu_workspace_stops_before_forward(monkeypatch):
    from types import SimpleNamespace
    import comfy.model_management as mm
    events = []
    patcher = SimpleNamespace(load_device=torch.device('cuda'), model=lambda x: events.append('forward'))
    monkeypatch.setattr(mm, 'load_models_gpu', lambda *a, **kw: events.append('admit'))
    monkeypatch.setattr(mm, 'get_free_memory', lambda *a: 0)
    with pytest.raises(RuntimeError, match='insufficient'):
        PaddleOCRModel.__new__(PaddleOCRModel).infer(patcher, torch.zeros(1, 3, 48, 320), 'recognizer')
    assert events == ['admit']
