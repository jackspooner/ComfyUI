"""Native PP-OCRv6 loading, image preparation and literal text recognition.

Postprocessing follows PaddlePaddle/Hugging Face Transformers, Apache-2.0.
Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team.
https://www.apache.org/licenses/LICENSE-2.0
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.storage
import comfy.utils
from comfy.ldm.paddleocr import Detector, Recognizer


SOURCES = {
    'detector': {'repository': 'PaddlePaddle/PP-OCRv6_medium_det_safetensors', 'revision': '4236c2b61741a259c091fd879dcc4edc339e916c',
                 'weights_sha256': 'bd393266c02e1a680b1b34c301d5d0d81e6290440b7f8ab0f5d5032276b17eb1',
                 'model_type': 'pp_ocrv6_medium_det', 'config_sha256': '3ea7e760f64255c152ba139e2f5f798303f7189b87c923cf498e2b2e830cadd3',
                 'processor_sha256': '74421569a28a78d417db320f9bc039ce0997a1defe2b35f259dfc74299c9f1ed'},
    'recognizer': {'repository': 'PaddlePaddle/PP-OCRv6_medium_rec_safetensors', 'revision': '024cad6a831de75c2c3c26e711ba8c4a82ccd24b',
                   'weights_sha256': '5f43c16f2a684b1d2284662178bdb604febd3d6bfdb5ca73828d08d0f7c0c3e9',
                   'model_type': 'pp_ocrv6_small_rec', 'config_sha256': 'aa5f351b4a7df95cc4953f013d53a6d44205f1bb0aee0ec7eb243d1d8fae1c27',
                   'processor_sha256': '2b24fa36f548893f26a931cf44f1a1a6b2b14ed7e2f8f8e28e6848801a8278db'},
}
PREPROCESSING = ('no_enlargement', 'official')
OPERATIONS = ('detect_and_recognize', 'recognize_lines')


def read_assets(checkpoint, kind):
    root = Path(checkpoint).resolve().parent
    source = SOURCES[kind]
    assets = []
    for filename, digest_key in [('config.json', 'config_sha256'), ('preprocessor_config.json', 'processor_sha256')]:
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'PaddleOCR {filename} escapes the model directory.')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source[digest_key]:
            raise ValueError(f'PaddleOCR requires the pinned official {kind} {filename}; see docs/paddleocr.md.')
        assets.append(json.loads(data))
    return assets


def load_model(checkpoint, kind):
    config, processor = read_assets(checkpoint, kind)
    cls = Detector if kind == 'detector' else Recognizer
    sd = comfy.utils.load_torch_file(str(checkpoint), safe_load=True)
    # Cast-aware linear ops can defer parameters until loading. Validate against
    # an allocation-free schema rather than that incomplete initial state dict.
    schema = cls(config, torch.nn, device='meta', dtype=torch.float32)
    expected = {k: v.shape for k, v in schema.state_dict().items() if not k.endswith('.num_batches_tracked')}
    del schema
    if set(sd) != set(expected) or any(sd[k].shape != shape or sd[k].dtype != torch.float32 for k, shape in expected.items()):
        raise ValueError(f'PaddleOCR {kind} checkpoint must have the complete supported FP32 tensor inventory and shapes.')
    model = cls(config, comfy.ops.manual_cast, device='cpu', dtype=torch.float32).eval()
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device(), fast_disk=comfy.storage.state_dict_fast_disk(sd))
    model.load_state_dict(sd, strict=False, assign=patcher.is_dynamic())
    digest = hashlib.sha256()
    with Path(checkpoint).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != SOURCES[kind]['weights_sha256']:
        raise ValueError(f'PaddleOCR {kind} checkpoint hash differs from the pinned official weights.')
    identity = {**SOURCES[kind], 'filename': Path(checkpoint).name, 'sha256': digest.hexdigest()}
    return patcher, processor, identity


def prepare_detector(image, mode):
    _, h, w = image.shape
    ph, pw = h, w
    if mode == 'official':
        scale = max(1.0, 736 / min(h, w))
        rh, rw = int(h * scale), int(w * scale)
        if max(rh, rw) > 4000:
            scale = 4000 / max(rh, rw)
            rh, rw = int(rh * scale), int(rw * scale)
        ph, pw = max(round(rh / 32) * 32, 32), max(round(rw / 32) * 32, 32)
        x = tvF.resize(image[None].float(), [ph, pw], interpolation=tvF.InterpolationMode.BILINEAR)
        resized = [ph, pw]
    else:
        ph, pw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        x = F.pad(image[None].float(), (0, pw - w, 0, ph - h), value=255)
        resized = [h, w]
    mean = x.new_tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)
    std = x.new_tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
    # Match the exported Transformers processor: normalize RGB, then reverse channels.
    x = ((x / 255 - mean) / std)[:, [2, 1, 0]]
    return x, {'source_size': [h, w], 'resized_size': resized, 'input_size': [ph, pw], 'padding_right_bottom': [pw - resized[1], ph - resized[0]]}


def prepare_recognizer(image, mode):
    _, h, w = image.shape
    scale = 48 / h if mode == 'official' else min(1.0, 48 / h)
    rh, rw = 48 if mode == 'official' else min(h, 48), max(1, math.ceil(w * scale))
    if mode == 'official':
        target = int(48 * max(w / h, 320 / 48))
        rw = 3200 if target > 3200 else min(target, math.ceil(48 * w / h))
    x = tvF.resize(image[None], [rh, rw], interpolation=tvF.InterpolationMode.BILINEAR, antialias=False) if (rh, rw) != (h, w) else image[None]
    x = (x[:, [2, 1, 0]].float() / 255 - 0.5) / 0.5
    x = F.pad(x, (0, 0, 0, 48 - rh), value=1)
    x = F.pad(x, (0, max(0, 320 - rw), 0, 0), value=0)
    return x, {'source_size': [h, w], 'resized_size': [rh, rw], 'input_size': list(x.shape[-2:]), 'padding_right_bottom': [max(0, 320 - rw), 48 - rh]}


def mini_box(contour):
    rect = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
    left = sorted(points[:2], key=lambda p: p[1])
    right = sorted(points[2:], key=lambda p: p[1])
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32), min(rect[1])


def unclip(polygon, ratio):
    distance = cv2.contourArea(polygon) * ratio / cv2.arcLength(polygon, True)
    x, y = polygon[:, 0], polygon[:, 1]
    ccw = (x @ np.roll(y, -1) - y @ np.roll(x, -1)) > 0
    edges = np.roll(polygon, -1, axis=0) - polygon
    directions = edges / np.maximum(np.linalg.norm(edges, axis=1, keepdims=True), 1e-6)
    normals = np.stack([directions[:, 1], -directions[:, 0]], axis=1) if ccw else np.stack([-directions[:, 1], directions[:, 0]], axis=1)
    shifted = polygon + distance * normals
    previous = np.roll(directions, 1, axis=0)
    cross = previous[:, 0] * directions[:, 1] - previous[:, 1] * directions[:, 0]
    parallel = np.abs(cross) < 1e-6
    delta = shifted - np.roll(shifted, 1, axis=0)
    t = (delta[:, 0] * directions[:, 1] - delta[:, 1] * directions[:, 0]) / np.where(parallel, 1.0, cross)
    vertices = np.roll(shifted, 1, axis=0) + previous * t[:, None]
    vertices[parallel] = (polygon + 0.5 * distance * (np.roll(normals, 1, axis=0) + normals))[parallel]
    return vertices.astype(np.float32)


def detection_regions(prediction, transform, mode, threshold=0.3, box_threshold=0.6, unclip_ratio=1.5):
    h, w = transform['source_size']
    if mode == 'no_enlargement':
        prediction = prediction[:h, :w]
    mh, mw = prediction.shape
    contours, _ = cv2.findContours((prediction > threshold).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours[:1000]:
        mm.throw_exception_if_processing_interrupted()
        box, side = mini_box(contour)
        if side < 3:
            continue
        x0 = max(0, math.floor(box[:, 0].min()))
        x1 = min(mw - 1, math.ceil(box[:, 0].max()))
        y0 = max(0, math.floor(box[:, 1].min()))
        y1 = min(mh - 1, math.ceil(box[:, 1].max()))
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, (box - [x0, y0]).reshape(1, -1, 2).astype(np.int32), 1)
        score = cv2.mean(prediction[y0:y1 + 1, x0:x1 + 1], mask)[0]
        if score < box_threshold:
            continue
        box, side = mini_box(unclip(box, unclip_ratio).reshape(-1, 1, 2))
        if side < 5:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] * w / mw), 0, w)
        box[:, 1] = np.clip(np.round(box[:, 1] * h / mh), 0, h)
        if cv2.contourArea(box) > 0:
            regions.append({'corners': box.tolist(), 'detection_score': score})
    return sorted(regions, key=lambda r: (min(p[1] for p in r['corners']), min(p[0] for p in r['corners']), r['corners']))


def rectify(image, corners):
    points = np.array(corners, dtype=np.float32)
    width = max(1, int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    height = max(1, int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotate = height / width >= 1.5
    if rotate:
        crop = np.rot90(crop)
    return torch.from_numpy(np.ascontiguousarray(crop)).movedim(-1, 0), {'perspective_matrix': matrix.tolist(), 'rectified_size': [height, width], 'rotation_ccw_degrees': 90 if rotate else 0, 'interpolation': 'bicubic'}


def decode_ctc(probabilities, vocabulary):
    values, ids = probabilities.max(dim=-1)
    ids, values = ids.tolist(), values.tolist()
    selected = [i for i, token in enumerate(ids) if token != 0 and (i == 0 or ids[i - 1] != token)]
    text = ''.join(vocabulary[ids[i]] for i in selected)
    return text, sum(values[i] for i in selected) / len(selected) if selected else None


def workspace_bytes(kind, shape):
    _, _, h, w = shape
    if kind == 'detector':
        return 64 * 1024 ** 2 + h * w * 1024
    sequence = max(1, w // 4)
    return 32 * 1024 ** 2 + sequence * 18710 * 4 + 8 * sequence * sequence * 4 + h * w * 256


class PaddleOCRModel:
    def __init__(self, detector_path, recognizer_path):
        self.detector, _, detector_id = load_model(detector_path, 'detector')
        self.recognizer, processor, recognizer_id = load_model(recognizer_path, 'recognizer')
        self.vocabulary = processor['character_list']
        self.identity = {'detector': detector_id, 'recognizer': recognizer_id}

    def get_models(self):
        return [self.detector, self.recognizer]

    def infer(self, patcher, pixels, kind):
        required = workspace_bytes(kind, pixels.shape)
        mm.load_models_gpu([patcher], memory_required=required, force_full_load=True)
        if not mm.is_device_cpu(patcher.load_device) and mm.get_free_memory(patcher.load_device) < required + mm.extra_reserved_memory():
            raise RuntimeError(f'PaddleOCR {kind} has insufficient free workspace after model admission; reduce the input dimensions.')
        mm.throw_exception_if_processing_interrupted()
        with comfy.model_prefetch.malloc_graph_scope(patcher.load_device):
            output = patcher.model(pixels.to(patcher.load_device))
        return output.detach().cpu()

    def recognize(self, images, operation='detect_and_recognize', preprocessing='no_enlargement', threshold=0.3, box_threshold=0.6, unclip_ratio=1.5):
        if operation not in OPERATIONS or preprocessing not in PREPROCESSING:
            raise ValueError('Unsupported PaddleOCR operation or preprocessing mode.')
        results = []
        progress = comfy.utils.ProgressBar(len(images))
        for index, image in enumerate(images):
            mm.throw_exception_if_processing_interrupted()
            pixels = (image[..., :3].detach().cpu() * 255).round().to(torch.uint8).movedim(-1, 0)
            _, h, w = pixels.shape
            array = pixels.movedim(0, -1).numpy()
            if operation == 'detect_and_recognize':
                inputs, detector_transform = prepare_detector(pixels, preprocessing)
                prediction = self.infer(self.detector, inputs, 'detector')[0, 0].numpy()
                regions = detection_regions(prediction, detector_transform, preprocessing, threshold, box_threshold, unclip_ratio)
            else:
                detector_transform = None
                regions = [{'corners': [[0, 0], [w, 0], [w, h], [0, h]], 'detection_score': None}]
            for region in regions:
                mm.throw_exception_if_processing_interrupted()
                if operation == 'detect_and_recognize':
                    crop, rectification = rectify(array, region['corners'])
                else:
                    crop, rectification = pixels, None
                inputs, recognition_transform = prepare_recognizer(crop, preprocessing)
                probabilities = self.infer(self.recognizer, inputs, 'recognizer')[0]
                text, score = decode_ctc(probabilities, self.vocabulary)
                region.update(text=text, recognition_score=score, rectification=rectification, recognition_transform=recognition_transform)
            results.append({'image_index': index, 'width': w, 'height': h, 'pixel_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                            'detector_transform': detector_transform, 'regions': regions})
            progress.update(1)
        return {'version': 1, 'operation': operation, 'preprocessing': preprocessing, 'models': self.identity, 'images': results}


def plain_text(result):
    return '\n\n'.join(f"[Image {image['image_index']}]\n" + '\n'.join(region['text'] for region in image['regions']) for image in result['images'])
