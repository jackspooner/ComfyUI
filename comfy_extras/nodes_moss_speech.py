from comfy_api.latest import ComfyExtension, IO
from comfy.text_encoders.moss_speech import encode_speech_reference


class MossSpeechReference(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='MossSpeechReference', display_name='MOSS Speech Reference', category='audio/generation',
            description='Prepare voice-reference codes with the pinned MOSS mono/resample and bounded -20 dBFS gain policy.',
            inputs=[IO.Audio.Input('audio'), IO.Custom('AUDIO_CODEC').Input('codec')],
            outputs=[IO.Custom('AUDIO_CODES').Output()])

    @classmethod
    def execute(cls, audio, codec):
        return IO.NodeOutput(encode_speech_reference(codec, audio))


class MossSpeechGenerate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='MossSpeechGenerate', display_name='MOSS Speech Generate', category='audio/generation',
            description='Native Delay and v1.5 acting speech. References are ordered speakers; decode the output with the MOSS codec. Use MOSS Dialogue Generate for TTSD defaults.',
            inputs=[IO.Clip.Input('clip'), IO.String.Input('text', multiline=True),
                    IO.String.Input('instruction', multiline=True, default=''), IO.String.Input('language', default=''),
                    IO.Int.Input('duration_tokens', default=0, min=0, max=1048576, tooltip='0 serializes None; this is a prompt hint, not an exact duration.'),
                    IO.String.Input('quality', default=''), IO.String.Input('sound_event', default=''), IO.String.Input('ambient_sound', default=''),
                    IO.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                    IO.Int.Input('max_length', default=1000, min=0, max=1048576),
                    IO.Float.Input('text_temperature', default=1.5, min=0, max=10, step=0.05),
                    IO.Float.Input('text_top_p', default=1.0, min=0, max=1, step=0.01), IO.Int.Input('text_top_k', default=50, min=0, max=155648),
                    IO.Float.Input('audio_temperature', default=1.7, min=0, max=10, step=0.05),
                    IO.Float.Input('audio_top_p', default=0.8, min=0, max=1, step=0.01), IO.Int.Input('audio_top_k', default=25, min=0, max=1025),
                    IO.Float.Input('audio_repetition_penalty', default=1.0, min=0.01, max=10, step=0.01),
                    IO.Custom('AUDIO_CODES').Input('continuation', optional=True, tooltip='Previously encoded audio to continue. Include its transcript in the text prompt before the new text.'),
                    IO.Autogrow.Input('references', template=IO.Autogrow.TemplatePrefix(
                        input=IO.Custom('AUDIO_CODES').Input('reference'), prefix='reference_', min=0, max=IO.Autogrow._MaxNames), optional=True)],
            outputs=[IO.Custom('AUDIO_CODES').Output()])

    @classmethod
    def execute(cls, clip, text, instruction, language, duration_tokens, quality, sound_event, ambient_sound,
                seed, max_length, text_temperature, text_top_p, text_top_k,
                audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty, references=None, normalize_text=False, continuation=None):
        refs = None
        if references:
            last = max(int(name.removeprefix('reference_')) for name in references)
            refs = tuple(references.get(f'reference_{i}') for i in range(last + 1))
        tokens = clip.tokenize(text, instruction=instruction or None, language=language or None, references=refs,
                              tokens=duration_tokens or None, quality=quality or None, sound_event=sound_event or None,
                              ambient_sound=ambient_sound or None, normalize_text=normalize_text, continuation=continuation)
        return IO.NodeOutput(clip.generate_audio_codes(tokens, max_length=max_length, seed=seed,
            text_temperature=text_temperature, text_top_p=text_top_p, text_top_k=text_top_k,
            audio_temperature=audio_temperature, audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty))


class MossDialogueGenerate(MossSpeechGenerate):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = 'MossDialogueGenerate'
        schema.display_name = 'MOSS Dialogue Generate'
        schema.description = 'Native TTSD dialogue with [S1], [S2], ... speaker tags and ordered references. For continuation, encode the joined reference audio and include its transcript before new dialogue. Scene and Tokens serialize as None.'
        schema.inputs = [item for item in schema.inputs if item.id != 'duration_tokens']
        defaults = dict(max_length=8192, text_temperature=1.1, text_top_p=0.9, audio_temperature=1.1,
                        audio_top_p=0.9, audio_top_k=50, audio_repetition_penalty=1.1)
        for item in schema.inputs:
            if item.id in defaults:
                item.default = defaults[item.id]
        return schema

    @classmethod
    def execute(cls, **kwargs):
        return super().execute(duration_tokens=0, **kwargs)


class MossVoiceActing(MossSpeechGenerate):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = 'MossVoiceActing'
        schema.display_name = 'MOSS v1.5 Voice Acting'
        schema.description = 'v1.5 voice acting with optional pinned text normalization. Use language names such as English or Chinese.'
        schema.inputs.append(IO.Boolean.Input('normalize_text', default=True))
        return schema

    @classmethod
    def execute(cls, normalize_text=True, **kwargs):
        return super().execute(normalize_text=normalize_text, **kwargs)


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
        return [MossVoiceGenerator, MossSpeechReference, MossSpeechGenerate, MossVoiceActing, MossDialogueGenerate]


async def comfy_entrypoint():
    return MossSpeechExtension()
