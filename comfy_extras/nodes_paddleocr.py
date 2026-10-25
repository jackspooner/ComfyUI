import json
from pathlib import Path

import folder_paths
from comfy.paddleocr import PaddleOCRModel, OPERATIONS, PREPROCESSING, plain_text
from comfy_api.latest import ComfyExtension, io


PaddleOCRType = io.Custom('PADDLEOCR_MODEL')


def checkpoint_path(name):
    path = Path(folder_paths.get_full_path_or_raise('detection', name)).resolve()
    if path.suffix != '.safetensors' or not any(path.is_relative_to(Path(root).resolve()) for root in folder_paths.get_folder_paths('detection')):
        raise ValueError('PaddleOCR checkpoints must be safetensors files within a configured detection model directory.')
    return path


class LoadPaddleOCRModels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = [name for name in folder_paths.get_filename_list('detection') if name.endswith('.safetensors')]
        return io.Schema(node_id='LoadPaddleOCRModels', display_name='Load PaddleOCR Models', category='model/loaders',
                         inputs=[io.Combo.Input('detector_name', options=models), io.Combo.Input('recognizer_name', options=models)],
                         outputs=[PaddleOCRType.Output()])

    @classmethod
    def execute(cls, detector_name, recognizer_name):
        return io.NodeOutput(PaddleOCRModel(checkpoint_path(detector_name), checkpoint_path(recognizer_name)))


class PaddleOCR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PaddleOCR', display_name='PaddleOCR', category='image/analysis',
                         inputs=[PaddleOCRType.Input('model'), io.Image.Input('images'),
                                 io.Combo.Input('operation', options=list(OPERATIONS), default='detect_and_recognize'),
                                 io.Combo.Input('preprocessing', options=list(PREPROCESSING), default='no_enlargement'),
                                 io.Float.Input('threshold', default=0.3, min=0, max=1, step=0.01, advanced=True),
                                 io.Float.Input('box_threshold', default=0.6, min=0, max=1, step=0.01, advanced=True),
                                 io.Float.Input('unclip_ratio', default=1.5, min=0, max=10, step=0.1, advanced=True),
                                 io.Int.Input('detector_batch_size', default=1, min=1, max=64, optional=True, advanced=True),
                                 io.Int.Input('recognition_batch_size', default=1, min=1, max=64, optional=True, advanced=True),
                                 io.Autogrow.Input('additional_images', template=io.Autogrow.TemplatePrefix(
                                     io.Image.Input('image'), prefix='image', min=0, max=100), optional=True)],
                         outputs=[io.String.Output(display_name='text'), io.String.Output(display_name='regions_json')])

    @classmethod
    def execute(cls, model, images, operation, preprocessing, threshold, box_threshold, unclip_ratio,
                detector_batch_size=1, recognition_batch_size=1, additional_images=None):
        frames = list(images)
        for name in sorted(additional_images or {}, key=lambda name: int(name.removeprefix('image'))):
            frames.extend(additional_images[name])
        result = model.recognize(frames, operation, preprocessing, threshold, box_threshold, unclip_ratio,
                                 detector_batch_size=detector_batch_size, recognition_batch_size=recognition_batch_size)
        return io.NodeOutput(plain_text(result), json.dumps(result, ensure_ascii=False, allow_nan=False))


class PaddleOCRExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadPaddleOCRModels, PaddleOCR]


async def comfy_entrypoint():
    return PaddleOCRExtension()
