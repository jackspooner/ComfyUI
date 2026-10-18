# SoulX-Singer native integration plan

Source baseline: ComfyUI `a3d3d379481bacb448a12a22c386cde27c0d2f48`.
Upstream Soul-AILab/SoulX-Singer: `81aeb3ae772c70093c3de74dc23c92d983801ae4` (Apache-2.0; preserve Amphion/MIT attribution).
SVS/SVC assets: `40493ad90286056c7a9095035164434a79daa8c9`.
Preprocessing assets: `83dc50289d22a81b1e9998f5b9e111aef7c1fdcd`.
SVC's separately constructed Whisper-base: `e37978b90ca9030d5170a5c07aadb050351a65bb`.

1. Prepare isolated worktree and exact graph, inventory local models, obtain pinned assets through Download Manager 2, record receipts and verify published hashes.
2. Convert tensor-only checkpoints offline, preserving precision/configuration/phone inventory. Split the native flow model, Vocos and SVC Whisper encoder into managed components. Round-trip every exported tensor and record exclusions.
3. Port released noncausal adaptive-RMS Llama flow model and text/note/F0 encoder to native operations, selected attention, prefetch and ModelPatcher. Reuse existing ConvNeXt/Whisper components where their exact contracts fit. Keep upstream midpoint integration and CFG rescaling; no Transformers model, subprocess or internet in runtime.
4. Provide reference-audio F0 extraction, editable JSON notes/word timing and explicit replacement lyric timing. Reuse installed offline transcription where possible. Keep lead, backing harmonies and accompaniment distinct. Provide SVS melody/score and transcription-free SVC nodes with standard AUDIO inputs/outputs.
5. Compare controlled model math, preprocessing, converted tensors, vocoder and complete inference against pinned source. Verify actual RTX 5090 repeated execution, cancellation, model transitions and offload; retain metrics/settings/outputs separately from the read-only reference project. Separate pitch, timing, words, identity and listening evidence.
6. Deliver clearly arranged editor graphs and matching prompts, update README/setup and authoritative existing operating skills, run relevant tests and graph diff checks, review/commit/push/integrate safely. Service activation requires a separate idle-window operational decision; do not interrupt active jobs.

Upstream has released SVS melody/score conditioning and SVC raw-audio conditioning. Its documentation also lists roadmap/demo material; those listings are not acceptance evidence. Its automatic transcription is explicitly imperfect. The SVC Whisper wrapper is not an nn.Module and is absent from the released SVC state dict; it requires an explicit additional local artifact. Upstream dependency pins include Torch 2.2/Transformers 4.41/NumPy<2, which must not replace the serving environment. Native inference will not import the upstream application or its dependency bundle.
