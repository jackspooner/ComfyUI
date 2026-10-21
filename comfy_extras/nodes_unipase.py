"""Offline native UniPASE speech restoration."""
import folder_paths
from comfy.unipase import runtime


class UniPASEModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("audio_models"),)}}

    RETURN_TYPES = ("UNIPASE_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio/enhancement"
    DESCRIPTION = "Load the four locally packaged UniPASE networks with ComfyUI model management. No downloads."

    def load(self, model_name):
        return (runtime.load(folder_paths.get_full_path_or_raise("audio_models", model_name)),)


class UniPASEEnhanceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("UNIPASE_MODEL",), "audio": ("AUDIO",),
            "output_sample_rate": ("INT", {"default": 0, "min": 0, "max": 384000, "step": 1}),
            "enable_plc": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "enhance"
    CATEGORY = "audio/enhancement"
    DESCRIPTION = "Restore speech, processing each channel independently in bounded segments. 0 keeps the source sample rate. Optional packet loss concealment detects near-zero 20 ms packets. This is a generative speech model, with no guarantee of music preservation or removal of every synthetic artifact."

    def enhance(self, model, audio, output_sample_rate=0, enable_plc=True):
        return (runtime.enhance(model, audio, output_sample_rate, enable_plc),)


NODE_CLASS_MAPPINGS = {"UniPASEModelLoader": UniPASEModelLoader, "UniPASEEnhanceAudio": UniPASEEnhanceAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"UniPASEModelLoader": "Load UniPASE", "UniPASEEnhanceAudio": "UniPASE Speech Restoration"}
