# UniPASE acceptance record

20 October 2026, Linux x86_64, Python 3.13.16, PyTorch 2.14.1+cu130.
Source and checkpoint revisions and hashes are in [sources](unipase-sources.json)
and the local converter manifest. Code MIT, model-card Apache-2.0; component
notices are retained in the model directory.

## Verified offline

All four originals matched published SHA256s. Conversion validated 1,036 native
tensors, complete key/shape inventories, finite FP32 values and bit-exact saved
round trips after documented key mapping and positional weight-normalization
folding. Packaged SHA256:
`f4073752f4fbaa63c038401a3f4b05aa014a045851c8386c62a27269afba5157`.

79 targeted tests passed: UniPASE boundaries/math plus RVC and SoulX regressions.
UniPASE's 52 checks cover mono/stereo batches, 8/16/22.05/44.1/48 kHz source
rates, tiny inputs and 8/10/12/17/25-second boundaries, changed-rate sample count,
identity stitching, early cancellation, relative attention and PostNet normalization.
Fixture tests do not establish model quality or forced offload behavior.

`scripts/check_unipase.py` compared all three official noisy speech fixtures,
at 16 and 48 kHz output. Whole-clip CPU comparisons passed the predeclared maximum
absolute `.003` and relative-RMS `.015` gates. Observed maximum absolute difference
was `.000265`; worst relative RMS `.0000765` (0.00765%). Inputs were 4–14.61 seconds,
at 44.1/48 kHz. Retained WAVs and metrics:
`/home/jack/renders/unipase-acceptance/` (`*-before.wav`, `*-native.wav`,
`*-upstream.wav`, `cpu-parity.json`). The reference uses unchanged pinned numerical
code; only ESPnet's abstract base and PReLU lookup are replaced with PyTorch
identities. CPU comparison process peak RSS was 7,436,504 KiB with both complete
pipelines loaded; six comparisons took 54.8 seconds, not a serving RAM/VRAM bound.

The official short inference was used for parity, independently of our long-file
chunk orchestration. Neural speech parity therefore does not establish identical
long-form boundaries. Upstream's long script has short-input and changed-rate bugs;
our corrected orchestration has isolated boundary tests. Independent per-channel
processing extends upstream mono behavior; stereo spatial fidelity is unverified.

## Activation and remaining gates

GPU queue inference, cancellation/reload, explicit cleanup, measured GPU/RAM resource
usage, facade end-to-end delivery and listening acceptance are pending until recorded
below. No general music enhancement, universal artifact removal, low-VRAM capacity,
other-accelerator support or complete URGENT hybrid reproduction is claimed.

The first GPU prompt `ecd3fe38-4aa6-4310-84d4-f6a84d5ad711` failed allocation
compilation in the encoder loop. Its failed history is retained. The corrective
change captures layer-one features before the repeated scope and reuses stable
loop buffers, following existing native model patterns; compiler/VRAM flags remain
enabled. GPU revalidation is required before acceptance.
