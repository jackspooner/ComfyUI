#!/usr/bin/env python3
"""Offline CPU comparisons against pinned Applio and Transformers; no services."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def errors(actual, expected):
    delta = actual - expected
    return {"max": delta.abs().max().item(), "rms": delta.square().mean().sqrt().item(),
            "relative_rms": (delta.square().mean() / expected.square().mean()).sqrt().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "sources", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    settings = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(settings.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "5379cf261fa7d8085c3b077aa42dabca7c85310b":
        raise ValueError("Use pinned Applio 5379cf261fa7d8085c3b077aa42dabca7c85310b")
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(settings.upstream))
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    from comfy.rvc import runtime
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from transformers import HubertConfig, HubertModel

    torch.set_num_threads(6)
    report = {"source_revision": revision, "torch": torch.__version__,
              "thresholds": {"generator_max": 2e-5, "generator_relative_rms": 1e-4, "contentvec_max": 2e-5}}
    for name in ("kikiV1", "zhanzhanv2-xi"):
        checkpoint = torch.load(settings.sources / (name + ".pth"), map_location="cpu", weights_only=True)
        config = list(checkpoint["config"])
        config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
        version = checkpoint.get("version", "v1")
        reference = Synthesizer(*config, use_f0=True, text_enc_hidden_dim=256 if version == "v1" else 768).eval()
        del reference.enc_q
        reference.load_state_dict(checkpoint["weight"], strict=True)
        native = runtime.load(settings.models / "audio_models" / "rvc" / (name + ".safetensors"), "rvc_voice").patcher.model
        torch.manual_seed(20)
        features = torch.randn(1, 35, 256 if version == "v1" else 768)
        f0 = torch.full((1, 35), 180.)
        pitch, speaker, length = runtime.coarse_f0(f0), torch.tensor([0]), torch.tensor([35])
        with torch.inference_mode():
            mean, logs, _ = reference.enc_p(features, pitch, length)
            native_mean, native_logs = native.enc_p(features, pitch, {})
            report[name + "_encoder"] = {"mean_max": (mean - native_mean).abs().max().item(),
                                         "log_max": (logs - native_logs).abs().max().item()}
            torch.manual_seed(123)
            expected = reference.infer(features, length, pitch, f0, speaker)[0]
            actual = native(features, pitch, f0, speaker, torch.Generator().manual_seed(123), {})
            report[name] = errors(actual, expected)
        del reference, native

    config = HubertConfig(**json.loads((settings.sources / "config.json").read_text()))
    reference = HubertModel(config).eval()
    state = torch.load(settings.sources / "contentvec.bin", map_location="cpu", weights_only=True)
    reference.load_state_dict({k: v for k, v in state.items() if not k.startswith("final_proj.")}, strict=True)
    native = runtime.load(settings.models / "audio_encoders" / "rvc_contentvec.safetensors", "rvc_contentvec").patcher.model
    with torch.inference_mode():
        torch.manual_seed(22)
        audio = torch.randn(1, 16000) * .1
        report["contentvec"] = errors(native(audio, "v2", {}), reference(audio).last_hidden_state)
    settings.output.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("kikiV1", "zhanzhanv2-xi"):
        assert report[name]["max"] <= 2e-5 and report[name]["relative_rms"] <= 1e-4, report[name]
    assert report["contentvec"]["max"] <= 2e-5, report["contentvec"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
