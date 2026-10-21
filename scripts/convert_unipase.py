#!/usr/bin/env python3
"""Package the four pinned local UniPASE checkpoints. Never downloads or executes pickle globals."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
import comfy.ops
from comfy.ldm.unipase.wavlm import DeWavLM
from comfy.ldm.unipase.vocos import Adapter, Vocoder
from comfy.ldm.unipase.postnet import TFGridNet

CODE_REVISION = "857b60ad05d37a2cf6d7a89883ec9fc4fc164b45"
WEIGHT_REVISION = "f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0"
FILES = {
    "encoder": ("DeWavLM-Omni.pt", "16c94d05ff6ef2bdf08b0be2853826afda687c2d63c833382198db7682476689"),
    "adapter": ("Adapter.pt", "3f3f9f8076e304bc740fef5496f91528ca9ea6be7f5a63deec71cdc33471c8c7"),
    "vocoder": ("Vocoder_DWO-L1.pt", "7595d48962c296bab2f7cb307f299acf5a02ec6a972d9ab7da3ed71b4a54a3d5"),
    "postnet": ("PostNet.pt", "1bf950dd2432e3cd2ef7055226b259bb3ac3fba8d7ce1f5f2424581a3d19cf63"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoder_state(state):
    state = dict(state)
    v = state.pop("encoder.pos_conv.0.weight_v")
    g = state.pop("encoder.pos_conv.0.weight_g")
    state["encoder.pos_conv.0.weight"] = torch._weight_norm(v, g, 2)
    result = {}
    for key, value in state.items():
        key = re.sub(r"(feature_extractor.conv_layers.\d+)\.0\.", r"\1.conv.", key)
        key = re.sub(r"(feature_extractor.conv_layers.\d+)\.2\.1\.", r"\1.layer_norm.", key)
        result[key] = value.contiguous()
    return result


def build(kind, config, device):
    if kind == "postnet":
        return TFGridNet(**config, dtype=torch.float32, device=device, operations=comfy.ops.manual_cast)
    return {"encoder": DeWavLM, "adapter": Adapter, "vocoder": Vocoder}[kind](config, dtype=torch.float32, device=device, operations=comfy.ops.manual_cast)


def convert(source, output):
    manifest, temporary = output.with_suffix(".manifest.json"), output.with_suffix(".partial")
    if any(p.exists() for p in (output, temporary, manifest)):
        raise FileExistsError("Inspect existing output/partial/manifest, then select a fresh destination; originals are never overwritten.")
    states, configs, sources = {}, {}, {}
    for kind, (filename, expected_hash) in FILES.items():
        path = source / filename
        actual_hash = digest(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Incomplete or unsupported {filename}: SHA256 mismatch against pinned release")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state, config = checkpoint["model"], checkpoint["cfg"]
        if kind == "encoder":
            state = encoder_state(state)
        with torch.device("meta"):
            model = build(kind, config, "meta")
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"{kind} tensor inventory mismatch: missing={sorted(set(expected)-set(state))}; extra={sorted(set(state)-set(expected))}")
        for key, tensor in state.items():
            if tensor.shape != expected[key].shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid {kind}.{key}: shape/dtype/payload differs")
            states[f"{kind}.{key}"] = tensor.contiguous()
        configs[kind] = config
        sources[filename] = actual_hash
        del checkpoint, state, model
    metadata = {"format": "unipase", "version": "1", "configs": json.dumps(configs), "code_revision": CODE_REVISION,
                "weight_revision": WEIGHT_REVISION, "source_hashes": json.dumps(sources), "weight_license": "Apache-2.0"}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(states, temporary, metadata=metadata)
    with safe_open(temporary, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(states) or saved.metadata() != metadata:
            raise ValueError("Saved inventory/metadata mismatch")
        for key, tensor in states.items():
            if not torch.equal(tensor, saved.get_tensor(key)):
                raise ValueError(f"Saved value mismatch: {key}")
    evidence = {"code_revision": CODE_REVISION, "weight_revision": WEIGHT_REVISION, "sources": sources,
                "output_sha256": digest(temporary), "tensors": len(states), "excluded": [],
                "transforms": ["encoder convolution keys mapped to native Wav2Vec2 layers", "positional convolution weight normalization folded with dim=2"],
                "verification": "complete native shape inventory, finite FP32, saved tensors bit-exact after declared transforms"}
    temporary.rename(output)
    manifest.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"output": str(output), "manifest": str(manifest), **evidence}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    settings = parser.parse_args()
    torch.set_num_threads(6)
    convert(settings.source, settings.output)
