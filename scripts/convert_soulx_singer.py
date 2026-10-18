#!/usr/bin/env python3
"""Convert pinned, already downloaded SoulX-Singer artifacts. No network or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SOURCE_REVISION = "81aeb3ae772c70093c3de74dc23c92d983801ae4"
WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def verify_source(path, expected):
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"Incomplete source: {path}")
    digest = sha256(path)
    if expected.get("lfs"):
        if digest != expected["lfs"]["sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {path}")
    else:
        blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0" + path.read_bytes()).hexdigest()
        if blob != expected["blobId"]:
            raise ValueError(f"Source Git blob mismatch: {path}")
    return digest


def write_component(destination, relative, state, metadata, manifest):
    path = destination / relative
    if path.with_suffix(".partial").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: tensor.clone() if "rotary_embed.freqs" in key else tensor.contiguous() for key, tensor in state.items()}
    temporary = path.with_suffix(".partial")
    encoded = {key: value if isinstance(value, str) else json.dumps(value) for key, value in metadata.items()}
    if not path.exists():
        save_file(state, temporary, metadata=encoded)
    # Fidelity gate: read every tensor, including buffers, before publication.
    with safe_open(path if path.exists() else temporary, framework="pt", device="cpu") as saved:
        if saved.metadata() != encoded:
            raise ValueError(f"Output provenance changed: {relative}")
        if set(saved.keys()) != set(state):
            raise ValueError(f"Tensor inventory changed: {relative}")
        for key, value in state.items():
            actual = saved.get_tensor(key)
            if actual.dtype != value.dtype or not torch.equal(actual, value):
                raise ValueError(f"Tensor changed: {relative}:{key}")
    if not path.exists():
        temporary.rename(path)
    manifest["outputs"][relative] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                                      "tensors": len(state), "round_trip": "all tensors bit-exact"}


def convert(source, preprocess, source_code, whisper, destination, resume=False):
    manifest_path = destination / "soulx_singer.manifest.json"
    outputs = [f"diffusion_models/soulx_singer_{kind}.safetensors" for kind in ("svs", "svc")]
    outputs += [f"vae/soulx_singer_{kind}_vocos.safetensors" for kind in ("svs", "svc")]
    outputs += [f"audio_encoders/soulx_singer_{kind}.safetensors" for kind in ("whisper_base", "rmvpe", "karaoke", "dereverb")]
    if (any((destination / path).exists() for path in outputs) or manifest_path.exists()) and not resume:
        raise FileExistsError("SoulX-Singer output exists; choose a new destination and preserve prior evidence.")
    sources = json.loads((Path(__file__).resolve().parents[1] / "docs/soulx-singer-sources.json").read_text())
    manifest = {"source_revision": SOURCE_REVISION, "model_revision": sources["SoulX-Singer"]["revision"],
                "preprocess_revision": sources["SoulX-Singer-Preprocess"]["revision"], "inputs": {}, "outputs": {}, "excluded": {}}
    for name, root in [("SoulX-Singer", source), ("SoulX-Singer-Preprocess", preprocess)]:
        for item in sources[name]["files"]:
            path = root / item["rfilename"]
            # Optional ASR/ROSVOT source assets are retained for research but unused by this native path.
            if path.exists():
                manifest["inputs"][str(path)] = verify_source(path, item)
    config = yaml.safe_load((source / "config.yaml").read_text())
    phones_path = source_code / "soulxsinger/utils/phoneme/phone_set.json"
    phones = json.loads(phones_path.read_text())
    manifest["inputs"][str(phones_path)] = sha256(phones_path)
    for kind, filename in [("svs", "model.pt"), ("svc", "model-svc.pt")]:
        checkpoint = torch.load(source / filename, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict"]
        singer, vocoder, omitted = {}, {}, {}
        for key, value in state.items():
            if key.startswith("vocoder.model."):
                vocoder[key.removeprefix("vocoder.model.")] = value
            elif key.startswith("mel."):
                omitted[key] = {"shape": list(value.shape), "reason": "Mel frontend buffers retained in synthesis metadata"}
            else:
                singer[key.removeprefix("cfm_decoder.model.")] = value
        metadata = {"format": "soulx_singer", "kind": kind, "config": config, "phones": phones,
                    "source_revision": SOURCE_REVISION, "source_sha256": manifest["inputs"][str(source / filename)]}
        # Keep the exact trained mel filter and window instead of regenerating them at inference.
        for key, value in state.items():
            if key.startswith("mel.model."):
                metadata[key] = value.tolist()
        write_component(destination, f"diffusion_models/soulx_singer_{kind}.safetensors", singer, metadata, manifest)
        write_component(destination, f"vae/soulx_singer_{kind}_vocos.safetensors", vocoder,
                        {"format": "soulx_vocos", "kind": kind, "source_revision": SOURCE_REVISION}, manifest)
        manifest["excluded"][filename] = omitted
        del singer, vocoder, state, checkpoint
    whisper_file = whisper / "model.safetensors"
    if sha256(whisper_file) != "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b":
        raise ValueError("Whisper-base source hash mismatch")
    manifest["inputs"][str(whisper_file)] = sha256(whisper_file)
    whisper_state = load_file(whisper_file)
    encoder = {k.removeprefix("model.encoder."): v for k, v in whisper_state.items() if k.startswith("model.encoder.")}
    if not encoder:
        encoder = {k.removeprefix("encoder."): v for k, v in whisper_state.items() if k.startswith("encoder.")}
    if not encoder:
        raise ValueError("Whisper-base encoder tensors missing")
    write_component(destination, "audio_encoders/soulx_singer_whisper_base.safetensors", encoder,
                    {"format": "soulx_whisper", "source_revision": WHISPER_REVISION}, manifest)
    manifest["excluded"]["whisper-base"] = "Decoder tensors are unused by SVC"
    pitch_state = torch.load(preprocess / "rmvpe/rmvpe.pt", map_location="cpu", weights_only=True, mmap=True)
    pitch_state = {"model." + k: v for k, v in pitch_state.items()}
    pitch_state["mel_basis"] = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True))
    write_component(destination, "audio_encoders/soulx_singer_rmvpe.safetensors", pitch_state,
                    {"format": "soulx_rmvpe", "source_revision": SOURCE_REVISION}, manifest)
    bank = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60)
    bank[0, 0] = bank[-1, -1] = 1
    bands = [np.flatnonzero(row > 0).tolist() for row in bank]
    for kind, filename in [("karaoke", "mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"),
                           ("dereverb", "dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")]:
        state = torch.load(preprocess / filename, map_location="cpu", weights_only=True, mmap=True)
        write_component(destination, f"audio_encoders/soulx_singer_{kind}.safetensors", state,
                        {"format": "soulx_roformer", "kind": kind, "bands": bands, "source_revision": SOURCE_REVISION}, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "preprocess", "source-code", "whisper", "destination"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Verify existing outputs against every source tensor/provenance field and continue; never overwrite weights.")
    args = parser.parse_args()
    convert(args.source, args.preprocess, args.source_code, args.whisper, args.destination, args.resume)
