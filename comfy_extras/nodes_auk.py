import math
import json

import torch

import comfy.audio
import comfy.model_management
from comfy_api.latest import ComfyExtension, io
from comfy.text_encoders import auk_prompt
from comfy_extras.nodes_textgen import TextGenerate


class EmptyAuKLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="EmptyAuKLatentAudio", display_name="Empty AuK Latent Audio", category="audio/auk",
                         inputs=[io.Float.Input("seconds", default=5.0, min=0.02, max=1000.0, step=0.02),
                                 io.Int.Input("batch_size", default=1, min=1, max=4096)],
                         outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, seconds, batch_size):
        samples = torch.zeros((batch_size, 64, math.ceil(seconds * 50)), device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": samples, "type": "audio", "sample_rate": 24000, "downscale_ratio_temporal": 480})


class AuKConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="AuKConditioning", display_name="AuK Conditioning", category="audio/auk",
                         inputs=[io.Clip.Input("clip"), io.String.Input("instruction", multiline=True),
                                 io.Audio.Input("reference_audio", optional=True), io.Vae.Input("vae", optional=True),
                                 io.Int.Input("reference_seed", default=0, min=0, max=0xffffffffffffffff,
                                              tooltip="Seed for AuK's stochastic reference-audio VAE encoding.")],
                         outputs=[io.Conditioning.Output()])

    @classmethod
    def execute(cls, clip, instruction, reference_seed, reference_audio=None, vae=None):
        reference = None
        if reference_audio is not None:
            if vae is None:
                raise ValueError("Connect the AuK VAE when supplying reference audio.")
            waveform = comfy.audio.resample(reference_audio["waveform"].mean(dim=1, keepdim=True), reference_audio["sample_rate"], vae.audio_sample_rate)
            reference_length = waveform.shape[-1] // 480
            if waveform.shape[-1] < 480:
                waveform = torch.nn.functional.pad(waveform, (0, 480 - waveform.shape[-1]))
                reference_length = 1
            # Restore caller RNG states after stochastic reference encoding.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(reference_seed)
                reference = vae.encode(waveform.movedim(1, -1))
        tokens = clip.tokenize(instruction, audio=reference_audio)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if reference is not None:
            conditioning = [[c, {**options, "auk_reference": reference, "auk_reference_length": reference_length}] for c, options in conditioning]
        return io.NodeOutput(conditioning)


class AuKPromptHelper(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="AuKPromptHelper", display_name="AuK Prompt Helper", category="audio/auk",
                         description="Optional local preparation using a separately loaded generative CLIP, such as MOSS-Audio. No downloads or cloud calls.",
                         inputs=[io.Clip.Input("clip"), io.String.Input("request", multiline=True),
                                 io.Audio.Input("reference_audio", optional=True),
                                 io.String.Input("reference_transcript", multiline=True, default="", optional=True),
                                 io.Float.Input("duration", optional=True, force_input=True, min=0.02, max=1000.0,
                                                tooltip="Explicit duration in seconds takes precedence over the estimate."),
                                 io.Int.Input("max_length", default=2048, min=1, max=1048576)],
                         outputs=[io.String.Output(display_name="instruction"), io.Float.Output(display_name="seconds")])

    @classmethod
    def execute(cls, clip, request, max_length, reference_audio=None, reference_transcript="", duration=None):
        reference_seconds = None if reference_audio is None else reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
        payload = json.dumps({"request": request, "reference_seconds": reference_seconds,
                              "reference_transcript": reference_transcript, "explicit_duration": duration}, ensure_ascii=False)
        system = auk_prompt.SYSTEM_PROMPT + json.dumps(auk_prompt.TASKS, ensure_ascii=False)
        generated = TextGenerate.execute(clip, payload, max_length, {"sampling_mode": "off"}, audio=reference_audio,
                                         thinking=False, system_prompt=system)
        result = auk_prompt.parse_preparation(generated.result[0])
        instruction, seconds = auk_prompt.prepare(result, request, reference_seconds, reference_transcript, duration)
        return io.NodeOutput(instruction, seconds)


class AuKExtension(ComfyExtension):
    async def get_node_list(self):
        return [EmptyAuKLatentAudio, AuKConditioning, AuKPromptHelper]


async def comfy_entrypoint():
    return AuKExtension()
