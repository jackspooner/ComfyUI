# Image conversation acceptance, 2026-10-21

Local normal-queue tests exercised an image-only first turn, a text follow-up,
a differently sized second image, comparison, and continuation from saved history
after restarting ComfyUI. Every successful response retained the full version 2
history with immutable attachment IDs and reported version 1 context information
including visual tokens. No turns were excluded. The working budget was 4096
tokens with a 128-token output allowance, greedy decoding and thinking disabled.

The references were a 320×320 image labeled RED SQUARE and a 420×300 image labeled
BLUE CIRCLE. The drawn red shape is rectangular, despite its label; these tests
establish image association, replay and counting, not calibrated recognition
quality. All six models distinguished the red first image from the blue second
image and identified the first image after restart.

The ordinary Load Image → Generate Text socket path also returned version 2
history automatically, without `history`, `prompt_content` or `context_budget`:
Qwen3.8 prompt `63f3ba2f-47fd-495d-8a09-c27d8c57b60f`, 114 input tokens.

| Checkpoint | First turn | Comparison | Restart continuation |
| --- | --- | --- | --- |
| `qwen3.8_27b_w4a8.safetensors` | `bc55edd2-12ee-4631-9482-2a2f806a10a1` | `b75ae3a9-6952-4d91-9088-43b2272e3681` | `4484533c-eb73-4969-875f-fa82217b6c45` |
| `qwen3vl_4b_bf16.safetensors` | `b23df007-e231-4275-8232-3eeee62e0b69` | `b929a140-685d-4214-83b3-53ebb3234a56` | `b5044d6d-d7ff-48b0-b738-674b518e61cb` |
| `qwen3_omni_30b_a3b_nvfp4.safetensors` | `ae8b75b5-2bea-4b81-94f5-af2bdb7697d0` | `1fc407e7-2513-49f7-a435-92daf2642037` | `f97ffc54-3c61-4fc6-a208-7ab95eb55315` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `7ee64766-ef2b-4eaf-bc2b-912e2e5dea59` | `e852e8e8-9fd8-45fb-80c4-3c7552d7ef97` | `3e356399-5129-455f-848f-f4bc0d337e41` |
| `gemma4_12b_bf16.safetensors` | `12c49beb-b5a7-4e6b-94e4-510cbec4e12d` | `c88fddf8-1eb7-449b-85c9-d462d8063f4b` | `360d6665-1543-4f0a-987b-9b14e6003be2` |
| `glm_ocr_bf16.safetensors` | `9ea89807-8c42-4f0a-a21b-b3d830fd4085` | `00b74ce3-1ac2-4429-a681-6c7b2f513ddb` | `1be63349-9b22-4fd4-92a0-7ca23309fd11` |

Generate Text outputs were saved beneath `output/image-history-acceptance` through
SaveText. Snapshots remain beneath the configured system-user `text_history/images`
directory. Copy the history JSON and snapshots together; see
[the storage contract](text-conversations.md#image-attachments).

The initial standalone Gemma3 run exposed a direct-list token accounting mismatch.
The corrected run above passed after preserving that tokenizer's existing return
contract; the regression now covers both direct lists and named token streams.
The combined history, generation, vision, Gemma, MOSS, Audio Flamingo, Omni,
GLM-OCR and shared-attention suites passed 192 tests with four external-reference
tests skipped. A concurrent shared-attention update exposed a redundant Omni
wrapper; reusing the shared Attention implementation removed duplicate container
wrapping and preference arguments while preserving the selected backend.

No complete Qwen2.5-VL checkpoint was installed, so its live acceptance remains
unverified. Unit tests exercise its preprocessing geometry and forwarding of
multimodal position IDs. No models were downloaded. Separate Qwen3.5 weights,
maximum-context memory capacity, other quantizations and other devices were not
live-tested. GLM-OCR retains its task-specific status and existing production
activation restrictions; these small labeled examples do not resolve earlier
OCR-quality limitations or establish general conversational ability.

## RouterText acceptance

RouterText commit `c9ee15f31a8f213e6f932be6f6e9cd37764f8d94` passed its full build,
405 tests (one skipped), lint, formatting and OpenAPI checks. Live acceptance
used the existing Qwen3.8 and Gemma4 routes with native history enabled. An
image-only saved chat retained its exact preview bytes, survived a RouterText
restart, accepted a second differently sized image and compared both. Switching
the same chat to Gemma4 retained the images; five successful turns were saved
exactly once. Cancelling prompt `4f90112f-34c2-4df0-8c64-058f6672c9dc` interrupted
only that job and saved no extra turn. The test chat was then deleted; ComfyUI's
independently reusable snapshots remained. Legacy conversation mode still
returned `CONVERSATION_IMAGE_NOT_SUPPORTED` for image input. Direct `/v1/infer`
prompt `e9cf0f0d-437c-45fd-bfd5-033d51852ca0` reported 131 actual input tokens.

Local JSON receipts and reproduction scripts are retained beneath
`output/image-history-acceptance/verification-2026-10-21`.
