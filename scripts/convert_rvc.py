#!/usr/bin/env python3
"""Package local Applio ContentVec or RVC HiFi-GAN exports. Never downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.rvc.checkpoint import fold_weights, voice_config
from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
import comfy.ops
from comfy.ldm.rvc.content import ContentEncoder
from comfy.ldm.rvc.model import Synthesizer


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def convert(source, output, kind, revision, expected_sha256=None):
    manifest = output.with_suffix(".manifest.json")
    temporary = output.with_suffix(".partial")
    if output.exists() or temporary.exists() or manifest.exists():
        raise FileExistsError("Output or partial output exists; use a fresh destination after inspecting it.")
    source_hash = digest(source)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if kind == "voice":
        config, version, use_f0 = voice_config(checkpoint)
        original = checkpoint["weight"]
        excluded = {k: "training posterior encoder" for k in original if k.startswith("enc_q.")}
        state, transforms = fold_weights({k: v for k, v in original.items() if k not in excluded})
        metadata = dict(format="rvc_voice", config=json.dumps(config), version=version, use_f0=json.dumps(use_f0), vocoder="HiFi-GAN")
    else:
        if source_hash != "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e":
            raise ValueError("Only the pinned Applio ContentVec export is supported; custom embedders require separate architecture acceptance.")
        original = checkpoint
        excluded = {k: "training mask embedding" for k in original if k == "masked_spec_embed"}
        state, transforms = fold_weights({k: v for k, v in original.items() if k not in excluded}, positional=True)
        metadata = dict(format="rvc_contentvec", architecture="hubert-base-applio", sample_rate="16000")
        if state.get("final_proj.weight", torch.empty(0)).shape != (256, 768):
            raise ValueError("Expected Applio HuBERT-base ContentVec with its v1 final projection.")
    with torch.device("meta"):
        model = Synthesizer(config, version, use_f0, comfy.ops.manual_cast) if kind == "voice" else ContentEncoder(torch.float32, "meta", comfy.ops.manual_cast)
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError(f"Incomplete/unsupported tensor inventory: missing={sorted(set(expected) - set(state))}, unexpected={sorted(set(state) - set(expected))}")
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid tensor shape/dtype/payload: {key}")
    metadata.update(source_revision=revision, source_sha256=source_hash)
    state = {k: v.contiguous() for k, v in state.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, temporary, metadata=metadata)
    with safe_open(temporary, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(state) or saved.metadata() != metadata:
            raise ValueError("Saved tensor inventory or metadata mismatch")
        for key, value in state.items():
            if not torch.equal(value, saved.get_tensor(key)) or value.dtype != saved.get_tensor(key).dtype:
                raise ValueError(f"Saved tensor mismatch: {key}")
    evidence = dict(source=str(source), source_sha256=source_hash, source_revision=revision,
                    output_sha256=digest(temporary), tensors=len(state), excluded=excluded, transforms=transforms,
                    verification="all saved tensors bit-exact after documented normalization/key transforms")
    # The artifact is self-contained; publish the receipt only after verified payload publication.
    temporary.rename(output)
    manifest.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), manifest=str(manifest), tensors=len(state))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("voice", "contentvec"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    convert(args.source, args.output, args.kind, args.source_revision, args.sha256)
