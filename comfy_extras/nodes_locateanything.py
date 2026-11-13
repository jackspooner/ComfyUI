import numpy as np
from PIL import Image

import folder_paths
from comfy.locateanything import runtime
from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS
from comfy_api.latest import ComfyExtension, io


class LocateAnythingLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LocateAnythingLoader', display_name='Load LocateAnything', category='model/loaders',
            description='Load pinned native MoonViT/projector and Qwen2 packages. Prepare offline with scripts/convert_locateanything.py.',
            inputs=[io.Combo.Input('vision_name', options=folder_paths.get_filename_list('clip_vision')),
                    io.Combo.Input('language_name', options=folder_paths.get_filename_list('text_encoders')),
                    io.Combo.Input('device', options=['auto', 'cpu'], default='auto'),
                    io.Combo.Input('attention', options=['optimized', *REGISTERED_ATTENTION_FUNCTIONS], default='pytorch')],
            outputs=[io.Custom('LOCATE_VISION').Output(), io.Custom('LOCATE_LANGUAGE').Output()])

    @classmethod
    def execute(cls, vision_name, language_name, device='auto', attention='pytorch'):
        if device not in ('auto','cpu'):
            raise ValueError('LocateAnything device must be auto or cpu')
        return io.NodeOutput(runtime.load(folder_paths.get_full_path_or_raise('clip_vision', vision_name), 'vision', device == 'cpu', attention),
            runtime.load(folder_paths.get_full_path_or_raise('text_encoders', language_name), 'language', device == 'cpu', attention))


class LocateAnythingGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LocateAnythingGenerate', display_name='LocateAnything Generate', category='image/locate',
            inputs=[io.Custom('LOCATE_VISION').Input('vision'), io.Custom('LOCATE_LANGUAGE').Input('language'),
                    io.Image.Input('image'), io.String.Input('prompt', default='', multiline=True),
                    io.Combo.Input('generation_mode', options=['hybrid','fast','slow'], default='hybrid'),
                    io.Int.Input('max_new_tokens', default=2048, min=1, max=8192),
                    io.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff)],
            outputs=[io.String.Output()])

    @classmethod
    def execute(cls, vision, language, image, prompt, generation_mode, max_new_tokens, seed):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError('image must be a non-empty ComfyUI IMAGE tensor')
        pixels = (image[0].detach().cpu().clamp(0,1).numpy()*255.).round().astype(np.uint8)
        pil = Image.fromarray(pixels[:,:,0], mode='L').convert('RGB') if pixels.shape[-1] == 1 else Image.fromarray(pixels[:,:,:3], mode='RGB')
        return io.NodeOutput(runtime.locate(vision, language, pil, prompt, generation_mode, max_new_tokens, seed))


class LocateAnythingExtension(ComfyExtension):
    async def get_node_list(self):
        return [LocateAnythingLoader, LocateAnythingGenerate]


async def comfy_entrypoint():
    return LocateAnythingExtension()
