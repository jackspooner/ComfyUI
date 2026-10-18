#!/usr/bin/env python3
"""Split an already downloaded Singer Plus checkpoint for native ComfyUI loaders."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

IMPLEMENTATION_REVISION = "baa409c2e7e5e775f09b4e92a220808f4827d2cc"


def convert(source, source_code, destination):
    config = json.loads((source / "config.json").read_text())
    if config.get("is_distilled") is not False or config.get("use_ema") is not True:
        raise ValueError("This converter supports the non-distilled YingMusic Singer Plus EMA checkpoint.")
    output_paths = [destination / part / "yingmusic_singer_plus.safetensors" for part in ("diffusion_models", "audio_encoders", "vae")]
    assets = destination / "text_encoders" / "yingmusic_singer_plus"
    manifest_path = destination / "yingmusic_singer_plus.manifest.json"
    if any(path.exists() for path in [*output_paths, assets, manifest_path]):
        raise FileExistsError("YingMusic destination already exists; select a new output directory.")
    checkpoint = source / "model.safetensors"
    g2p = source_code / "src/YingMusicSinger/utils/f5_tts/g2p"
    required_assets = [g2p / "g2p/vocab.json", g2p / "sources/chinese_lexicon.txt", g2p / "sources/pinyin_2_bpmf.txt",
                       *[g2p / "sources/g2p_chinese_model" / name for name in ("polychar.txt", "vocab.txt", "polydict.json", "poly_bert_model.onnx")]]
    for path in required_assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    with checkpoint.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"implementation_revision": IMPLEMENTATION_REVISION, "source_sha256": source_hash, "files": {}}
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        for path, prefix in zip(output_paths, ("model.transformer.", "midi_teacher.midi_conform.", "vae.model.")):
            state = {key.removeprefix(prefix): tensors.get_tensor(key) for key in tensors.keys() if key.startswith(prefix)}
            if not state:
                raise ValueError(f"Missing checkpoint component: {prefix}")
            if prefix.startswith("midi_teacher"):
                state["mel_basis"] = tensors.get_tensor("melody_spectrogram_extract.mel_basis")
            expected_counts = {"model.transformer.": 366, "midi_teacher.midi_conform.": 635, "vae.model.": 365}
            if len(state) != expected_counts[prefix]:
                raise ValueError(f"Unexpected Singer Plus tensor inventory for {prefix}: {len(state)}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            save_file(state, temporary, metadata={"source": "ASLP-lab/YingMusic-Singer-Plus", "implementation_revision": IMPLEMENTATION_REVISION, "source_sha256": source_hash})
            temporary.rename(path)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"][str(path.relative_to(destination))] = {"sha256": digest, "bytes": path.stat().st_size, "tensors": len(state)}
            del state
    assets.mkdir(parents=True)
    for path in required_assets:
        relative = path.relative_to(g2p / "sources") if path.is_relative_to(g2p / "sources") else Path(path.name)
        output = assets / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        with output.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["files"][str(output.relative_to(destination))] = {"sha256": digest, "bytes": output.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-code", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.source_code, args.destination)
