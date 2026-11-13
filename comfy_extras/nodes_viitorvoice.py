"""Native ViiTor model/refinement components; alignment and codecs compose separately."""
import folder_paths
from comfy_api.latest import ComfyExtension, IO
from comfy.viitorvoice import package, runtime


class ViiTorVoiceLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ViiTorVoiceLoader', display_name='Load ViiTorVoice', category='model/loaders',
            inputs=[IO.Combo.Input('model_name', options=folder_paths.get_filename_list('text_encoders'))],
            outputs=[IO.Custom('VIITORVOICE_MODEL').Output()])

    @classmethod
    def execute(cls, model_name):
        return IO.NodeOutput(package.load(folder_paths.get_full_path_or_raise('text_encoders', model_name), 'language'))


class ViiTorVoicePlan(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ViiTorVoicePlan', category='audio/edit',
            description='Build the pinned sentence-edit draft from native Qwen alignment and padded DualCodec source codes.',
            inputs=[IO.Custom('AUDIO_CODES').Input('codes'), IO.String.Input('alignment_json'),
                    IO.String.Input('original_text', multiline=True), IO.String.Input('edited_text', multiline=True),
                    IO.String.Input('language', default='en'), IO.Combo.Input('align_granularity', options=['auto','word','character']),
                    IO.Float.Input('padding_ms', default=250., min=0, max=2000),
                    IO.Float.Input('expand_mask_ratio', default=1.5, min=0, max=8),
                    IO.Float.Input('length_scale', default=1., min=.25, max=4)],
            outputs=[IO.Custom('VIITORVOICE_PLAN').Output()])

    @classmethod
    def execute(cls, codes, alignment_json, original_text, edited_text, language, align_granularity, padding_ms, expand_mask_ratio, length_scale):
        return IO.NodeOutput(runtime.plan(codes, alignment_json, original_text, edited_text, language,
            align_granularity, padding_ms, expand_mask_ratio, length_scale))


class ViiTorVoiceGenerate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ViiTorVoiceGenerate', category='audio/edit',
            inputs=[IO.Custom('VIITORVOICE_MODEL').Input('model'), IO.Custom('VIITORVOICE_PLAN').Input('plan'),
                    IO.Int.Input('num_steps', default=32, min=1, max=100), IO.Float.Input('cfg_scale', default=2., min=0, max=20),
                    IO.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff, control_after_generate=True)],
            outputs=[IO.Custom('AUDIO_CODES').Output(), IO.Custom('VIITORVOICE_EDIT').Output()])

    @classmethod
    def execute(cls, model, plan, num_steps, cfg_scale, seed):
        return IO.NodeOutput(*runtime.generate(model, plan, num_steps, cfg_scale, seed))


class ViiTorVoiceSplice(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ViiTorVoiceSplice', category='audio/edit',
            description='Restore original rate and channels; preserve exact original PCM outside the declared edit/blend ranges.',
            inputs=[IO.Audio.Input('source_audio'), IO.Audio.Input('decoded'), IO.Custom('VIITORVOICE_EDIT').Input('edit')],
            outputs=[IO.Audio.Output(), IO.String.Output()])

    @classmethod
    def execute(cls, source_audio, decoded, edit):
        return IO.NodeOutput(*runtime.splice(source_audio, decoded, edit))


class ViiTorVoiceExtension(ComfyExtension):
    async def get_node_list(self):
        return [ViiTorVoiceLoader, ViiTorVoicePlan, ViiTorVoiceGenerate, ViiTorVoiceSplice]


async def comfy_entrypoint():
    return ViiTorVoiceExtension()
