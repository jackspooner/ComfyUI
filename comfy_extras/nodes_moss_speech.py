from comfy_api.latest import ComfyExtension, IO


class MossVoiceGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='MossVoiceGenerator', display_name='MOSS VoiceGenerator', category='audio/generation',
            description='Generate discrete speech codes from text and a voice description. Decode with the separate MOSS v1 audio codec.',
            inputs=[
                IO.Clip.Input('clip'),
                IO.String.Input('text', multiline=True, default='Hello, it is nice to meet you.'),
                IO.String.Input('instruction', multiline=True, default='A warm, natural voice.'),
                IO.String.Input('language', default='en', tooltip='Language label supplied verbatim. Empty uses the reference None value.'),
                IO.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                IO.Int.Input('max_length', default=1000, min=0, max=1048576, tooltip='Outer generation steps, including delay/flush overhead. 0 uses remaining context.'),
                IO.Float.Input('audio_temperature', default=1.5, min=0, max=10, step=0.05),
                IO.Float.Input('audio_top_p', default=0.6, min=0, max=1, step=0.01),
                IO.Int.Input('audio_top_k', default=50, min=0, max=1025),
                IO.Float.Input('audio_repetition_penalty', default=1.1, min=0.01, max=10, step=0.01),
            ],
            outputs=[IO.Custom('AUDIO_CODES').Output()],
        )

    @classmethod
    def execute(cls, clip, text, instruction, language, seed, max_length, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty):
        tokens = clip.tokenize(text, instruction=instruction, language=language or None)
        return IO.NodeOutput(clip.generate_audio_codes(tokens, max_length=max_length, seed=seed,
            audio_temperature=audio_temperature, audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty))


class MossSpeechExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossVoiceGenerator]


async def comfy_entrypoint():
    return MossSpeechExtension()
