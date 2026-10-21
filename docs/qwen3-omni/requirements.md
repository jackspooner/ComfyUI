# Requirements

[Project index](README.md)

## Required result

Load a converted artifact from the selected community NVFP4 checkpoint through
Load CLIP and connect it to Generate Text. Preserve the standard CLIP contract
and the node's generated_text/thinking string outputs. Populate reasoning only
when the model emits it; do not imply the Instruct checkpoint has a switchable
reasoning mode solely because the node exposes a thinking input.

| ID | Requirement | Example acceptance case |
| --- | --- | --- |
| R1 | Text-only input remains valid | Answer a factual question with no connected media |
| R2 | Audio plus an instruction | Describe foreground/background sounds; transcribe a spoken phrase |
| R3 | Image plus an instruction | Describe objects and read clearly legible text |
| R4 | Image, audio and instruction in one request | Compare a pictured object with an audible event, using both sources |
| R5 | Native CLIP ownership | Load, generate, cancel, offload and reload through ComfyUI |
| R6 | Reproducible conversion | Pin source revision; account for every selected weight and required scale |
| R7 | Honest precision support | Document the execution actually tested, including activation and KV-cache precision |

Media sockets are optional because text-only is valid. Media-only requests with
an empty prompt should execute when the model's template permits them. Every
supplied media input must reach the model; silent omission is a failure. Preserve
input order and use existing repeated-image handling where applicable. Do not
invent quality-based limits or silently truncate media. Context overflow must
give actionable feedback after accounting for expanded media tokens.

## Scope boundaries

Audio means input understanding, not speech output. Retain the audio encoder
even when excluding Talker and Code2Wav. Full image and audio support is part of
the first completed release; a text-only milestone is an internal checkpoint.

Video synchronization, speech synthesis, streaming chat, persistent conversation
memory, training and compatibility with arbitrary diffusion conditioning are
outside this project. No new external service, runtime downloader or inference
worker is planned. Preserve existing nodes and other model loaders.

## Completion

R1–R7 must pass the [validation matrix](validation.md). Deliver a converter, native
integration, conversion provenance, usage/recovery documentation, saved example
workflow and measured resource results. Source integration, runtime activation
and GPU acceptance have distinct status fields. Update authoritative shared
operating guidance when the reusable capability is implemented.
