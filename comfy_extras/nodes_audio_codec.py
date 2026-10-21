import comfy.audio_codec
import folder_paths
from comfy_api.latest import ComfyExtension, IO


class AudioCodecLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='AudioCodecLoader', display_name='Load Audio Codec', category='model/loaders',
            description='Load a converted discrete audio codec from the VAE model folders. MOSS v1 requires its folded FP32 artifact.',
            inputs=[IO.Combo.Input('codec_name', options=folder_paths.get_filename_list('vae'))],
            outputs=[IO.Custom('AUDIO_CODEC').Output()],
        )

    @classmethod
    def execute(cls, codec_name):
        path = folder_paths.get_full_path_or_raise('vae', codec_name)
        return IO.NodeOutput(comfy.audio_codec.load_audio_codec(path))


class AudioCodecEncode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='AudioCodecEncode', display_name='Encode Audio Codes', category='audio/codec',
            description='Resample and mix to mono, then encode integer codes. No loudness normalization. MOSS valid lengths exclude partial final frames.',
            inputs=[IO.Audio.Input('audio'), IO.Custom('AUDIO_CODEC').Input('codec'), IO.Int.Input('codebooks', default=32, min=1, max=32)],
            outputs=[IO.Custom('AUDIO_CODES').Output()],
        )

    @classmethod
    def execute(cls, audio, codec, codebooks=32):
        return IO.NodeOutput(codec.encode(audio, codebooks))


class AudioCodecDecode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='AudioCodecDecode', display_name='Decode Audio Codes', category='audio/codec',
            description='Decode valid integer frames to raw PCM without normalization. Continuation prefixes retain causal context during decoding, then are trimmed from the returned audio.',
            inputs=[IO.Custom('AUDIO_CODES').Input('codes'), IO.Custom('AUDIO_CODEC').Input('codec')],
            outputs=[IO.Audio.Output(is_output_list=True)],
        )

    @classmethod
    def execute(cls, codes, codec):
        return IO.NodeOutput(codec.decode(codes))


class AudioCodecExtension(ComfyExtension):
    async def get_node_list(self):
        return [AudioCodecLoader, AudioCodecEncode, AudioCodecDecode]


async def comfy_entrypoint():
    return AudioCodecExtension()
