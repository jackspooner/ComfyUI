import json
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.audio
import comfy.model_management
import comfy.model_prefetch
import folder_paths
from comfy.soulx_singer import metadata, runtime
from comfy.soulx_singer.pronunciation import Pronunciation


def filenames(category, fragment):
    return [name for name in folder_paths.get_filename_list(category) if fragment in name.lower()]


class SoulXSingerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "singer_name": (filenames("diffusion_models", "soulx"),),
            "vocoder_name": (filenames("vae", "soulx"),),
            "precision": (["float16", "float32", "bfloat16"],),
        }, "optional": {"whisper_name": (filenames("audio_encoders", "soulx_singer_whisper"),)}}

    RETURN_TYPES = ("SOULX_SINGER",)
    FUNCTION = "load"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Load offline converted SoulX SVS or SVC and its matching Vocos through ComfyUI ModelPatcher. SVC also needs the local Whisper-base content encoder."

    def load(self, singer_name, vocoder_name, precision, whisper_name=None):
        dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[precision]
        if comfy.model_management.get_torch_device().type == "cpu":
            dtype = torch.float32
        bundle = {"singer": runtime.load_component(folder_paths.get_full_path_or_raise("diffusion_models", singer_name), "soulx_singer", dtype),
                  "vocoder": runtime.load_component(folder_paths.get_full_path_or_raise("vae", vocoder_name), "soulx_vocos", dtype)}
        kind = bundle["singer"].metadata["kind"]
        if bundle["vocoder"].metadata["kind"] != kind:
            raise ValueError("Select the Vocos file converted from the same SVS/SVC checkpoint as the singer.")
        if kind == "svc":
            if not whisper_name:
                raise ValueError("SVC requires a converted local Whisper-base encoder in the optional whisper_name input.")
            bundle["whisper"] = runtime.load_component(folder_paths.get_full_path_or_raise("audio_encoders", whisper_name), "soulx_whisper", dtype)
        return (bundle,)


class SoulXAudioModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (filenames("audio_encoders", "soulx_singer"),)}}

    RETURN_TYPES = ("SOULX_AUDIO_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Load converted RMVPE (CPU) or karaoke/dereverb Roformer (native selected device). No downloads."

    def load(self, model_name):
        path = folder_paths.get_full_path_or_raise("audio_encoders", model_name)
        with safe_open(path, framework="pt", device="cpu") as saved:
            kind = (saved.metadata() or {}).get("format")
        if kind not in ("soulx_rmvpe", "soulx_roformer"):
            raise ValueError("Choose the converted RMVPE, karaoke or dereverb audio model.")
        return (runtime.load_component(path, kind, torch.float32, cpu=kind == "soulx_rmvpe"),)


class SoulXExtractPitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pitch_model": ("SOULX_AUDIO_MODEL",), "audio": ("AUDIO",),
                             "voicing_threshold": ("FLOAT", {"default": .03, "min": 0., "max": 1., "step": .01})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pitch_json",)
    FUNCTION = "extract"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Extract one monophonic 50 Hz RMVPE F0 contour and editable estimated MIDI notes. Feed isolated lead vocals; backing harmonies require separate stems."

    def extract(self, pitch_model, audio, voicing_threshold):
        if pitch_model.metadata["format"] != "soulx_rmvpe":
            raise ValueError("Pitch extraction requires the RMVPE audio model.")
        model, device, _ = runtime.activate(pitch_model)
        f0 = model(runtime.mono(audio, 16000).to(device), 16000, voicing_threshold)[0].tolist()
        return (json.dumps({"frame_rate": 50, "duration": audio["waveform"].shape[-1] / audio["sample_rate"],
                            "f0": f0, "notes": metadata.pitch_notes(f0),
                            "note_estimator": "median-filtered RMVPE; inspect/correct boundaries; monophonic only"}),)


class SoulXSeparateVocals:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"separator": ("SOULX_AUDIO_MODEL",), "audio": ("AUDIO",),
                             "chunk_seconds": ("FLOAT", {"default": 5., "min": .1, "max": 60., "step": .1})}}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "residual",)
    FUNCTION = "separate"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Estimate vocals (karaoke model) or dry vocals (dereverb model). Residual is the original audio minus that estimate. Keep backing singers/reverb outside the lead's scalar pitch path."

    def separate(self, separator, audio, chunk_seconds):
        if separator.metadata["format"] != "soulx_roformer":
            raise ValueError("Separation requires the karaoke or dereverb Roformer audio model.")
        try:
            return runtime.separate(separator, audio, chunk_seconds)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()


class SoulXAlignLyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pitch_json": ("STRING", {"multiline": True}),
                             "timing_json": ("STRING", {"multiline": True, "default": '[{"start":0,"end":5,"text":"reference words"}]'}),
                             "lyrics": ("STRING", {"multiline": True, "default": ""}),
                             "pronunciations_json": ("STRING", {"multiline": True, "default": "{}"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "align"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Map replacement syllables to explicit phrase times and retain continuous F0. Use one lyric phrase per timed region (| or newline). Empty lyrics transcribes the timing text into metadata. Edit the returned rows before synthesis. Pronunciation overrides are ARPAbet arrays, or zh_/yue_ tokens for Chinese."

    def align(self, pitch_json, timing_json, lyrics, pronunciations_json):
        roots = [Path(root) / "soulx_singer" for root in folder_paths.get_folder_paths("text_encoders")]
        root = next((p for p in roots if (p / "cmudict.dict").is_file()), None)
        if root is None:
            raise FileNotFoundError("Prepare the local SoulX pronunciation assets under text_encoders/soulx_singer with scripts/prepare_soulx_pronunciation.py.")
        pronounce = Pronunciation(root)
        aligned = metadata.align_lyrics(json.loads(pitch_json), json.loads(timing_json), lyrics,
                                         pronounce, json.loads(pronunciations_json))
        return (json.dumps(metadata.segment_metadata(aligned), ensure_ascii=False),)


class SoulXSynthesize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"singer": ("SOULX_SINGER",), "reference_audio": ("AUDIO",),
                             "reference_metadata": ("STRING", {"multiline": True}),
                             "target_metadata": ("STRING", {"multiline": True}),
                             "control": (["melody", "score"],),
                             "pitch_shift": ("INT", {"default": 0, "min": -120, "max": 120}),
                             "steps": ("INT", {"default": 32, "min": 1, "max": 1000}),
                             "guidance": ("FLOAT", {"default": 3., "min": 0., "max": 100.}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Synthesize supplied aligned lyrics/notes or F0 using an isolated singer reference. Metadata may be a single segment or upstream segment list with millisecond placement times. Zero pitch shift preserves the supplied melody."

    def generate(self, singer, reference_audio, reference_metadata, target_metadata, control, pitch_shift, steps, guidance, seed):
        prompt = metadata.read_metadata(reference_metadata)
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        target = metadata.read_metadata(target_metadata)
        targets = target if isinstance(target, list) else [target]
        duration = max(item.get("time", [0, sum(metadata.values(item["duration"], float)) * 1000])[1] for item in targets) / 1000
        output = torch.zeros(1, 1, round(duration * 24000))
        try:
            for index, item in enumerate(targets):
                comfy.model_management.throw_exception_if_processing_interrupted()
                result = runtime.generate_svs(singer, reference_audio, prompt, item, control, pitch_shift, steps, guidance, (seed + index) % (2 ** 64))
                offset = round(item.get("time", [0])[0] * 24)
                count = min(output.shape[-1] - offset, result["waveform"].shape[-1])
                output[..., offset:offset + count] = result["waveform"][..., :count]
            return ({"waveform": output, "sample_rate": 24000},)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()


class SoulXVoiceConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"singer": ("SOULX_SINGER",), "reference_audio": ("AUDIO",), "source_audio": ("AUDIO",),
                             "reference_pitch": ("STRING", {"multiline": True}), "source_pitch": ("STRING", {"multiline": True}),
                             "pitch_shift": ("INT", {"default": 0, "min": -120, "max": 120}),
                             "steps": ("INT", {"default": 32, "min": 1, "max": 1000}),
                             "guidance": ("FLOAT", {"default": 3., "min": 0., "max": 100.}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert"
    CATEGORY = "audio/soulx"
    DESCRIPTION = "Convert source singing into the reference singer without transcribing lyrics. Extract both F0 inputs from isolated vocals. Long sources use overlapping windows; source words are retained."

    def convert(self, singer, reference_audio, source_audio, reference_pitch, source_pitch, pitch_shift, steps, guidance, seed):
        try:
            return (runtime.convert_svc(singer, reference_audio, source_audio, json.loads(reference_pitch),
                                        json.loads(source_pitch), pitch_shift, steps, guidance, seed),)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SoulXSingerLoader, SoulXAudioModelLoader, SoulXExtractPitch,
                                                   SoulXSeparateVocals, SoulXAlignLyrics, SoulXSynthesize, SoulXVoiceConvert)}
NODE_DISPLAY_NAME_MAPPINGS = {key: key.replace("SoulX", "SoulX ") for key in NODE_CLASS_MAPPINGS}
