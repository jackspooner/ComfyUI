"""Optional native captioner nodes. Install by linking this directory into custom_nodes."""
import json
import math

import folder_paths
import comfy.utils
from comfy.video_caption import VideoCaptioner
from comfy.video_caption_internvideo3 import InternVideoCaptioner
from comfy_api.latest import ComfyExtension, io
from comfy_api.input_impl import VideoFromFile


CaptionModel = io.Custom('VIDEO_CAPTION_MODEL')


class LoadVideoCaptioner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LoadVideoCaptioner', display_name='Load Audio-Visual Captioner', category='model/loaders',
            inputs=[io.Combo.Input('model_name', options=folder_paths.get_filename_list('text_encoders'))],
            outputs=[CaptionModel.Output()])

    @classmethod
    def execute(cls, model_name):
        path = folder_paths.get_full_path_or_raise('text_encoders', model_name)
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        loader = InternVideoCaptioner if metadata.get('architecture') == 'internvideo3_8b' else VideoCaptioner
        return io.NodeOutput(loader(state, metadata))


class CaptionVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='CaptionVideo', display_name='Caption Audio-Visual Video', category='text/video',
            description='Native SALMONN audio-visual questions, PadCaptioner dense captions or InternVideo3 visual questions. InternVideo3 has no audio encoder. Greedy decoding; no network, training, or automatic download.',
            inputs=[CaptionModel.Input('captioner'), io.Image.Input('images'),
                io.Float.Input('source_fps', default=24.0, min=0.01, max=1000),
                io.String.Input('prompt', multiline=True, default='Describe the visual events and the sounds in the video.'),
                io.Audio.Input('audio', optional=True),
                io.Float.Input('fps', default=2.0, min=0.1, max=30),
                io.Int.Input('max_pixels', default=61250, min=3136, max=262144),
                io.Int.Input('max_tokens', default=320, min=1, max=4096),
                io.Int.Input('max_events', default=24, min=1, max=30),
                io.Combo.Input('response_format', options=['text', 'json_schema'], default='text', optional=True),
                io.String.Input('json_schema', default='', multiline=True, optional=True)],
            outputs=[io.String.Output('text'), io.String.Output('result_json')])

    @classmethod
    def execute(cls, captioner, images, source_fps, prompt, fps=2.0, max_pixels=61250, max_tokens=320, max_events=24, audio=None, response_format='text', json_schema=''):
        selected_audio = audio if captioner.supports_audio else None
        result = captioner.caption(images, source_fps, selected_audio, prompt, fps, max_pixels, max_tokens, max_events, response_format, json_schema)
        result.update(model=captioner.family, sampling_fps=min(fps, source_fps), audio_used=selected_audio is not None)
        return io.NodeOutput(result['text'], json.dumps(result))


class VideoCaptionExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadVideoCaptioner, CaptionVideo, CaptionVideoFile]


class CaptionVideoFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='CaptionVideoFile', display_name='Caption Video File', category='text/video',
            inputs=[CaptionModel.Input('captioner'), io.String.Input('file'),
                io.String.Input('prompt', multiline=True, default='Describe the visual events and the sounds in the video.'),
                io.Boolean.Input('use_audio', default=True),
                io.Float.Input('fps', default=2.0, min=0.1, max=10),
                io.Int.Input('max_pixels', default=61250, min=3136, max=262144),
                io.Int.Input('max_tokens', default=320, min=1, max=4096),
                io.Int.Input('max_events', default=24, min=1, max=30),
                io.Combo.Input('response_format', options=['text', 'json_schema'], default='text', optional=True),
                io.String.Input('json_schema', default='', multiline=True, optional=True)],
            outputs=[io.String.Output('text'), io.String.Output('result_json')])

    @classmethod
    def execute(cls, captioner, file, prompt, use_audio=True, fps=2.0, max_pixels=61250, max_tokens=320, max_events=24, response_format='text', json_schema=''):
        if not folder_paths.exists_annotated_filepath(file):
            raise ValueError('Video file is absent from the configured input directory')
        source = VideoFromFile(folder_paths.get_annotated_filepath(file))
        duration, rate = source.get_duration(), float(source.get_frame_rate())
        if not math.isfinite(duration) or not 0 < duration <= 120 or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Captioning requires a video with valid timing and at most 120 seconds')
        stride = max(1, math.ceil(rate / fps))
        sampled_frames = math.ceil(duration * rate / stride)
        if sampled_frames > 512:
            raise ValueError('Selected video exceeds 512 sampled frames; lower fps or trim the source')
        components = source.get_components_selected(include_images=True, include_audio=use_audio and captioner.supports_audio,
            frame_stride=stride, max_size=512, compact=True)
        result = captioner.caption(components.images, float(components.frame_rate), components.audio,
            prompt, fps, max_pixels, max_tokens, max_events, response_format, json_schema)
        result.update(model=captioner.family, duration_seconds=duration,
            sampling_fps=min(fps, float(components.frame_rate)), audio_used=components.audio is not None)
        return io.NodeOutput(result['text'], json.dumps(result))


async def comfy_entrypoint():
    return VideoCaptionExtension()
