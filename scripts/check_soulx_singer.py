#!/usr/bin/env python3
"""Native real-checkpoint parity check against a retained upstream reference run."""

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from comfy.soulx_singer.metadata import preprocess


def run(args):
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = args.device == "cpu"
    from comfy.soulx_singer import runtime
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.reference / "request.json").read_text())
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float16 if settings["precision"] == "float16" else torch.float32
    kind = settings.get("task", "svs")
    bundle = {"singer": runtime.load_component(args.models / f"diffusion_models/soulx_singer_{kind}.safetensors", "soulx_singer", dtype),
              "vocoder": runtime.load_component(args.models / f"vae/soulx_singer_{kind}_vocos.safetensors", "soulx_vocos", dtype)}
    if kind == "svc":
        bundle["whisper"] = runtime.load_component(args.models / "audio_encoders/soulx_singer_whisper_base.safetensors", "soulx_whisper", dtype)
    example = Path(settings["upstream"]) / "example/audio"
    prompt = json.loads((example / "en_prompt.json").read_text())[0]
    target = json.loads((example / "en_target.json").read_text())[0]
    waveform, rate = sf.read(example / "en_prompt.mp3", dtype="float32", always_2d=True)
    reference = {"waveform": torch.from_numpy(waveform.T)[None], "sample_rate": rate}
    captured = {}
    flow, mel = runtime.sample_flow, runtime.reference_mel
    def sample(*pos, **kw):
        result = flow(*pos, **kw)
        captured["generated_mel"] = result.float().cpu().numpy()
        return result
    def extract(*pos, **kw):
        result = mel(*pos, **kw)
        captured["reference_mel"] = result.float().cpu().numpy()
        return result
    started = time.monotonic()
    with patch.object(runtime, "sample_flow", sample), patch.object(runtime, "reference_mel", extract):
        if kind == "svc":
            phone_set = json.loads(bundle["singer"].metadata["phones"])
            reference_frames = len(preprocess(prompt, phone_set)["mel2note"])
            reference_waveform = runtime.mono(reference)[..., :reference_frames * 480]
            reference = {"waveform": reference_waveform[:, None], "sample_rate": 24000}
            source, rate = sf.read(example / "en_target.mp3", dtype="float32", always_2d=True)
            source = runtime.mono({"waveform": torch.from_numpy(source.T)[None], "sample_rate": rate})[..., :len(target["f0"].split()) * 480]
            result = runtime.convert_svc(bundle, reference, {"waveform": source[:, None], "sample_rate": 24000},
                                          {"f0": list(map(float, prompt["f0"].split()))[:reference_frames]}, {"f0": list(map(float, target["f0"].split()))},
                                          0, settings["steps"], 3, settings["seed"])
        else:
            result = runtime.generate_svs(bundle, reference, prompt, target, settings["control"], 0,
                                           settings["steps"], 3, settings["seed"])
    captured["waveform"] = result["waveform"].squeeze().numpy()
    sf.write(args.output / "native.wav", captured["waveform"], 24000, subtype="FLOAT")
    np.savez(args.output / "tensors.npz", **captured)
    expected = np.load(args.reference / "tensors.npz")
    report = {"settings": settings, "seconds": time.monotonic() - started, "device": args.device, "metrics": {}}
    tolerances = {"reference_mel": (2e-6, 1e-5), "generated_mel": (2e-4, 2e-3), "waveform": (1e-3, 1e-2)}
    for key, value in captured.items():
        reference_value = expected[key]
        if value.shape != reference_value.shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} != {reference_value.shape}")
        error = value - reference_value
        atol, rtol = tolerances[key]
        passed = bool(np.allclose(value, reference_value, atol=atol, rtol=rtol))
        report["metrics"][key] = {"max_abs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                                   "relative_rmse": float(np.sqrt(np.mean(error ** 2) / max(1e-12, np.mean(reference_value ** 2)))),
                                   "atol": atol, "rtol": rtol, "passed": passed}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(metric["passed"] for metric in report["metrics"].values()):
        raise SystemExit("Native/upstream numerical parity failed; inspect retained tensors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("models", "reference", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run(parser.parse_args())
