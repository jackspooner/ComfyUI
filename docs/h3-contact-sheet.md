# H3 Contact Sheet

`MiniMaxH3ContactSheet` and `MiniMaxH3ContactSheetDecode` implement the five-slot
Turnaround recipe inside native ComfyUI. Use the REF2VA H3 diffusion model,
the `minimax` CLIP loader, H3 video VAE, stock model-only LoRA loader and stock
sampler. The recipe needs the separate Turnaround LoRA; an ordinary H3 video
workflow with that LoRA does not implement contact-sheet generation.

## Source and files

The implementation is adapted from [matlowai/ComfyUI-H3-ContactSheet](https://github.com/matlowai/ComfyUI-H3-ContactSheet/tree/3c8c8c66e69362e5abf9cf4c4e693baddab90ca3),
under GPL-3.0, matching this repository's licence. Modifications on 2026-10-29
reuse the native H3 resizing and conditioning owners, register native names,
and require exactly one five-slot packet at the decoder boundary.

The [Turnaround model card](https://huggingface.co/matlod/minimax-h3-turnaround)
declares the base MiniMax licence. Default weights are
`minimax_h3_five_view_512_s1500.safetensors` at revision
`791cceae539e19b297c08e26ff9d8246d231b89a`, 63,103,776 bytes,
SHA-256 `9bc12ae97d2076b3449f8b1feca1fff52252576d608a84e4625497644f265d35`.
Provision separately through the download owner into `loras/minimax/`.
ComfyUI does not download these files automatically.

## Graph and outputs

Connect CLIP, video VAE and one reference image to Contact Sheet. Mention
`<Picture 1>` in the prompt and describe the subject's starting direction and
desired orbit. The default per-view size is 512; the publisher also evaluated
1024 and 2048. Apply the LoRA through `LoraLoaderModelOnly` at strength 1,
then `BasicGuider`, `res_multistep` / `simple`, 28 steps and full denoise.
Connect the sampler's denoised latent to Contact Sheet Decode with the video VAE.
Save both outputs using ordinary `SaveImage` nodes.

The preparation node creates video `[1,24,5,H/16,W/16]` and audio `[1,32,2,28]`
in the native nested latent. Existing model packing and timeline positions
remain authoritative. The decoder duplicates each slot into a legal two-token
VAE clip, keeps its first pixel frame, and releases temporary decode outputs.
It returns five images in slot order and a single horizontal `[view0..view4]`
sheet. Whole-video VAE decoding is incorrect because it blends neighbouring
views. The reference is not included in the sheet. Audio is bookkeeping and is
not returned as a generated soundtrack.

Views are uncalibrated: never assign numeric camera angles from slot indices.
The rotation axis can follow the subject's pose, views can under-rotate or
re-pose, and larger resolutions can concentrate rotation in late slots. Review
individual images before selecting them for RefMod creation. This is not a
photogrammetry input guarantee. Publisher timings used RTX PRO 6000 hardware;
they are not local RTX 5090 benchmarks.

## Verification and activation

Run the documented worktree setup, then the prepared interpreter with
`-m pytest tests-unit/comfy_extras_test/minimax_contact_sheet_test.py -q`.
The isolated fixtures check packing, reference geometry, legal per-slot decoding,
ordering and rejection of ordinary video packets. They do not establish learned
view consistency, checkpoint parity, GPU memory behaviour or performance.
Source integration alone does not activate the nodes: discover the serving
schemas after an authorized idle deployment and retain the normal queue prompt
ID for representative inference. RouterImage's dedicated
`minimax-h3-contact-sheet` profile preserves the five outputs and sheet separately.
