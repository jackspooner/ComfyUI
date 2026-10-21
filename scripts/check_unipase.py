#!/usr/bin/env python3
"""Offline CPU real-checkpoint fidelity against pinned UniPASE; never downloads or starts services."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import types

import soundfile as sf
import torch


def errors(actual, expected):
    difference = actual - expected
    rms = difference.square().mean().sqrt().item()
    return {"max": difference.abs().max().item(), "rms": rms,
            "relative_rms": rms / max(expected.square().mean().sqrt().item(), 1e-12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "sources", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    settings = parser.parse_args()
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(settings.upstream))
    from comfy.cli_args import args
    args.cpu = True
    from comfy.unipase import runtime
    from scripts.convert_unipase import FILES, CODE_REVISION, WEIGHT_REVISION, digest
    for _, (filename, expected) in FILES.items():
        if digest(settings.sources / filename) != expected:
            raise ValueError(f"Reference source mismatch: {filename}")
    # This harness substitutes only ESPnet's abstract base and activation lookup.
    # The pinned reference uses no ESPnet execution/math beyond nn.Module/PReLU.
    for name in ("espnet2", "espnet2.enh", "espnet2.enh.separator", "espnet2.enh.separator.abs_separator", "espnet2.torch_utils", "espnet2.torch_utils.get_layer_from_string"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["espnet2.enh.separator.abs_separator"].AbsSeparator = torch.nn.Module
    sys.modules["espnet2.torch_utils.get_layer_from_string"].get_layer = lambda name: {"prelu": torch.nn.PReLU}[name]
    from models.unipase import UniPASE
    # Hash every imported upstream model source so reports bind to exact code.
    code_hashes = {str(p.relative_to(settings.upstream)): digest(p) for p in sorted((settings.upstream / "models").rglob("*.py"))}
    expected_code = json.loads((Path(__file__).resolve().parents[1] / "docs/unipase-sources.json").read_text())["model_code_sha256"]
    if code_hashes != expected_code:
        raise ValueError("Reference model code differs from pinned upstream")
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    torch.manual_seed(12)
    start = time.monotonic()
    names = {"encoder": "dewavlm", "adapter": "adapter", "vocoder": "vocoder", "postnet": "postnet"}
    reference = UniPASE(**{names[k] + "_ckpt_path": str(settings.sources / v[0]) for k, v in FILES.items()}).eval()
    native = runtime.load(settings.model)
    settings.output.mkdir(parents=True, exist_ok=True)
    report = {"code_revision": CODE_REVISION, "weight_revision": WEIGHT_REVISION, "model_sha256": digest(settings.model),
              "torch": torch.__version__, "device": "cpu", "reference_framework_substitutions": ["ESPnet AbsSeparator -> torch.nn.Module", "get_layer('prelu') -> torch.nn.PReLU"],
              "thresholds": {"waveform_max": .003, "waveform_relative_rms": .015}, "fixtures": []}
    for file in sorted((settings.upstream / "audio/noisy").glob("*")):
        audio, rate = sf.read(file, dtype="float32", always_2d=True)
        x = torch.from_numpy(audio[:, 0].copy())[None]
        for output_rate in (16000, 48000):
            t = time.monotonic()
            expected = reference(x, rate, output_rate, True)
            actual = runtime.segment(native, x, rate, output_rate, True)
            measurement = errors(actual, expected)
            entry = {"input": file.name, "input_sha256": digest(file), "input_rate": rate, "output_rate": output_rate,
                     "samples": actual.shape[-1], "elapsed_seconds": time.monotonic() - t, **measurement}
            report["fixtures"].append(entry)
            assert actual.shape == expected.shape
            assert measurement["max"] < .003 and measurement["relative_rms"] < .015, entry
            # Match upstream file-level peak restoration for reviewable examples.
            scale = x.abs().amax()
            for label, waveform in (("native", actual), ("upstream", expected)):
                waveform = waveform / (waveform.abs().amax() + 1e-8) * scale
                sf.write(settings.output / f"{file.stem}-{output_rate}-{label}.wav", waveform[0].numpy(), output_rate, subtype="FLOAT")
        sf.write(settings.output / f"{file.stem}-before.wav", x[0].numpy(), rate, subtype="FLOAT")
    report.update(elapsed_seconds=time.monotonic()-start, process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (settings.output / "cpu-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
