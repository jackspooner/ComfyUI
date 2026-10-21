import torch

import comfy.model_management
from comfy_api.latest import ComfyExtension, IO


class EmptyMossSoundEffectLatent(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='EmptyMossSoundEffectLatent', display_name='Empty MOSS SoundEffect Latent',
            category='model/latent/audio',
            inputs=[IO.Float.Input('seconds', default=30.0, min=0.02, max=1000.0, step=0.1,
                                   tooltip='The reference pipeline generates 30 seconds, then trims the decoded audio.'),
                    IO.Int.Input('batch_size', default=1, min=1, max=4096)],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size):
        samples = torch.zeros((batch_size, 128, max(1, round(seconds * 50))), dtype=torch.bfloat16,
                              device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({'samples': samples, 'type': 'audio', 'downscale_ratio_temporal': 960})


class MossSoundEffectExtension(ComfyExtension):
    async def get_node_list(self):
        return [EmptyMossSoundEffectLatent]


async def comfy_entrypoint():
    return MossSoundEffectExtension()
