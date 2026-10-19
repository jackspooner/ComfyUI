"""Native offline Applio/RVC file conversion."""

import folder_paths
from comfy.rvc import runtime


class RVCModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "voice_name": (folder_paths.get_filename_list("audio_models"),),
            "content_name": (folder_paths.get_filename_list("audio_encoders"),),
            "pitch_name": (["none"] + folder_paths.get_filename_list("audio_encoders"),),
        }}

    RETURN_TYPES = ("RVC_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio/rvc"
    DESCRIPTION = "Load converted HiFi-GAN RVC v1/v2, matching ContentVec, and optional shared RMVPE. FP32 native ComfyUI managed weights; no downloads."

    def load(self, voice_name, content_name, pitch_name):
        bundle = {
            "voice": runtime.load(folder_paths.get_full_path_or_raise("audio_models", voice_name), "rvc_voice"),
            "content": runtime.load(folder_paths.get_full_path_or_raise("audio_encoders", content_name), "rvc_contentvec"),
        }
        if pitch_name != "none":
            bundle["pitch"] = runtime.load(folder_paths.get_full_path_or_raise("audio_encoders", pitch_name), "soulx_rmvpe")
        return (bundle,)


class RVCVoiceConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("RVC_MODEL",), "audio": ("AUDIO",),
            "semitones": ("FLOAT", {"default": 0., "min": -48., "max": 48., "step": .1}),
            "speaker": ("INT", {"default": 0, "min": 0, "max": 65535}),
            "protect": ("FLOAT", {"default": .33, "min": 0., "max": .5, "step": .01}),
            "rms_mix": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": .01}),
            "index_name": (["none"] + folder_paths.get_filename_list("rvc_indexes"),),
            "index_rate": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert"
    CATEGORY = "audio/rvc"
    DESCRIPTION = "Convert one recording using a trained RVC voice. Mixes channels to mono, preserves timing at the model rate, and processes bounded segments. Use the voice's training embedder and matching index. Lower protect retains more unvoiced source features."

    def convert(self, model, audio, semitones, speaker, protect, rms_mix, index_name, index_rate, seed):
        path = folder_paths.get_full_path_or_raise("rvc_indexes", index_name) if index_name != "none" else None
        return (runtime.convert(model["voice"], model["content"], audio, semitones, speaker, protect, rms_mix, seed,
                                pitch_model=model.get("pitch"), index_path=path, index_rate=index_rate),)


NODE_CLASS_MAPPINGS = {"RVCModelLoader": RVCModelLoader, "RVCVoiceConvert": RVCVoiceConvert}
NODE_DISPLAY_NAME_MAPPINGS = {"RVCModelLoader": "Load Applio / RVC Models", "RVCVoiceConvert": "Applio / RVC Voice Convert"}
