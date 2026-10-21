"""Local Drama 3 speech, references and explicit revision histories."""
from pathlib import Path
import hashlib
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from comfy.fish_audio import runtime, history as take_history

ModelType = io.Custom('DRAMA3_MODEL')
ReferenceType = io.Custom('DRAMA3_REFERENCE')
HistoryType = io.Custom('DRAMA3_HISTORY')


class Drama3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Drama3Loader', display_name='Load Drama 3', category='audio/drama3',
            description='Load an offline converted NVFP4 generator and matching BF16 Fish codec. Blackwell CUDA required.',
            inputs=[io.Combo.Input('model_name', options=[n for n in folder_paths.get_filename_list('text_encoders') if n.lower().endswith('.safetensors')]),
                    io.Combo.Input('codec_name', options=[n for n in folder_paths.get_filename_list('vae') if n.lower().endswith('.safetensors')])],
            outputs=[ModelType.Output()])

    @classmethod
    def execute(cls, model_name, codec_name):
        return io.NodeOutput(runtime.load(folder_paths.get_full_path_or_raise('text_encoders', model_name),
                                          folder_paths.get_full_path_or_raise('vae', codec_name)))


class Drama3EncodeReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Drama3EncodeReference', display_name='Drama 3 Voice Reference', category='audio/drama3',
            inputs=[ModelType.Input('model'), io.Audio.Input('audio'), io.String.Input('transcript', multiline=True),
                    io.Int.Input('speaker', default=0, min=0, max=15)], outputs=[ReferenceType.Output()])

    @classmethod
    def execute(cls, model, audio, transcript, speaker):
        return io.NodeOutput(runtime.encode_reference(model, audio, transcript, speaker))


class Drama3Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        refs = io.Autogrow.TemplatePrefix(input=ReferenceType.Input('reference'), prefix='reference_', min=0, max=100)
        return io.Schema(node_id='Drama3Generate', display_name='Drama 3 Generate / Revise Take', category='audio/drama3',
            description='Generate mono 44.1 kHz speech or revise an earlier whole take using explicit saved history.',
            inputs=[ModelType.Input('model'), io.String.Input('text', multiline=True, dynamic_prompts=True),
                    io.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                    io.Float.Input('temperature', default=.7, min=0., max=10., step=.01),
                    io.Float.Input('top_p', default=.7, min=0., max=1., step=.01),
                    io.Int.Input('top_k', default=30, min=1, max=252288),
                    io.Int.Input('max_frames', default=1024, min=0, max=262144, tooltip='Maximum new audio frames per chunk; 0 uses upstream automatic defaults.'),
                    io.Boolean.Input('single_chunk', default=False),
                    io.Autogrow.Input('references', template=refs, optional=True), HistoryType.Input('history', optional=True)],
            outputs=[io.Audio.Output(), HistoryType.Output()])

    @classmethod
    def execute(cls, model, text, seed, temperature, top_p, top_k, max_frames, single_chunk, references=None, history=None):
        refs = tuple(ref for ref in (references or {}).values() if ref is not None)
        audio, updated = runtime.synthesize(model, text, refs, history, seed, temperature, top_p, top_k, max_frames, single_chunk)
        return io.NodeOutput(audio, updated)


class Drama3SaveHistory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Drama3SaveHistory', display_name='Save Drama 3 History', category='audio/drama3',
            inputs=[HistoryType.Input('history'), io.String.Input('filename_prefix', default='drama3/take')],
            outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, history, filename_prefix):
        directory, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        file = f'{filename}_{counter:05}.drama3.safetensors'
        path = Path(directory) / file
        if not path.resolve().is_relative_to(Path(folder_paths.get_output_directory()).resolve()):
            raise ValueError('History destination must stay inside the output directory')
        take_history.save(history, path)
        return io.NodeOutput(ui={'files': [ui.SavedResult(file, subfolder, io.FolderType.output)]})


def history_files():
    result = []
    for kind, directory in (('input', folder_paths.get_input_directory()), ('output', folder_paths.get_output_directory())):
        root = Path(directory)
        for path in root.rglob('*.drama3.safetensors'):
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                result.append(f'{path.relative_to(root).as_posix()} [{kind}]')
    return sorted(result)


class Drama3LoadHistory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Drama3LoadHistory', display_name='Load Drama 3 History', category='audio/drama3',
                         inputs=[io.Combo.Input('history_name', options=history_files())], outputs=[HistoryType.Output()])

    @classmethod
    def execute(cls, history_name):
        return io.NodeOutput(take_history.load(resolve_history(history_name)))

    @classmethod
    def fingerprint_inputs(cls, history_name):
        digest = hashlib.sha256()
        with resolve_history(history_name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()


def resolve_history(history_name):
    if history_name not in history_files():
        raise ValueError('Select an admitted input/output Drama 3 history file')
    path = Path(folder_paths.get_annotated_filepath(history_name))
    roots = (folder_paths.get_input_directory(), folder_paths.get_output_directory())
    if not any(path.resolve().is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError('History file must stay inside input or output')
    return path


class Drama3Extension(ComfyExtension):
    async def get_node_list(self):
        return [Drama3Loader, Drama3EncodeReference, Drama3Generate, Drama3SaveHistory, Drama3LoadHistory]


async def comfy_entrypoint():
    return Drama3Extension()
