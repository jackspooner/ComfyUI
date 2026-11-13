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
MODEL_REVISION = '9b3f444f2bccd77fbc03e32eb7c334fc96040b8d'
SEPARATOR_SHA256 = '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
SEPARATOR_CONFIG_SHA256 = 'f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926'
SEPARATOR_SOURCE_FILES = {
    'inference_api.py': '60165ec145aa99bff2d079af4327d0dc3244c2c0a805ae2cce3d90a374686583',
    'utils/model_utils.py': '5dcd1fc07d6fea131205ad0dd60d758b002846d43ae7a58f9583de4c2e3238b3',
    'models/bs_roformer/mel_band_roformer.py': '922051cbb9c6540c8b2eaeff415286ac46f1df99442234d86a2f4dff7942ac1b',
    'models/bs_roformer/attend.py': 'c2ccbe9179dcbc6ff4af0fe61141ddb69a4744a4f2114bf5a5c307579c7155e2',
}


def convert_separator(source, source_code, destination):
    """Reuse the native Roformer geometry and verified writer, not source Python."""
    import librosa
    import numpy as np
    import torch
    from comfy.cli_args import args
    args.cpu = True  # Offline header/shape validation must not initialize CUDA.
    from comfy.soulx_singer.separation import MelBandRoformer
    from scripts.convert_soulx_singer import sha256, write_component

    relative = 'audio_encoders/yingmusic_separator.safetensors'
    manifest_path = (destination / relative).with_suffix('.manifest.json')
    if any(path.exists() for path in (destination / relative, manifest_path, (destination / relative).with_suffix('.partial'))):
        raise FileExistsError('YingMusic separator destination exists; preserve it and choose a fresh destination.')
    checkpoint = source / 'ckpts/MelBandRoformer.ckpt'
    config = source / 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
    roots = {checkpoint: SEPARATOR_SHA256, config: SEPARATOR_CONFIG_SHA256}
    source_root = source_code / 'src/third_party/MusicSourceSeparationTraining'
    roots.update({source_root / name: digest for name, digest in SEPARATOR_SOURCE_FILES.items()})
    snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in roots}
    for path, expected in roots.items():
        if sha256(path) != expected:
            raise ValueError(f'YingMusic separator pinned source mismatch: {path}')
    bank = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60)
    bank[0, 0] = bank[-1, -1] = 1
    bands = [np.flatnonzero(row > 0).tolist() for row in bank]
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    with torch.device('meta'):
        expected = MelBandRoformer(bands, stems=1, dtype=torch.float32, device='meta', operations=torch.nn).state_dict()
    expected = {key: value for key, value in expected.items() if key not in ('freq_indices', 'num_bands_per_freq')}
    if (state.keys() != expected.keys() or len(state) != 684 or any(
            state[key].shape != value.shape or state[key].dtype != torch.float32 for key, value in expected.items())):
        raise ValueError('YingMusic separator requires the exact complete 684-tensor FP32 checkpoint')
    metadata = dict(format='yingmusic_roformer', kind='vocals', bands=bands,
        source_revision=IMPLEMENTATION_REVISION, model_revision=MODEL_REVISION,
        source_sha256=SEPARATOR_SHA256, config_sha256=SEPARATOR_CONFIG_SHA256,
        preprocessing='yingmusic_demix_v1', config=config.read_text())
    manifest = dict(source_revision=IMPLEMENTATION_REVISION, model_revision=MODEL_REVISION,
        sources={str(path.relative_to(source) if path.is_relative_to(source) else path.relative_to(source_code)): digest for path, digest in roots.items()},
        tensors={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in state.items()}, outputs={})
    if any((path.stat().st_size, path.stat().st_mtime_ns) != snapshot for path, snapshot in snapshots.items()):
        raise ValueError('YingMusic separator source changed during conversion')
    write_component(destination, relative, state, metadata, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(manifest_path)
    return manifest


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
    parser.add_argument('--component', choices=('singer', 'separator'), default='singer')
    args = parser.parse_args()
    if args.component == 'separator':
        convert_separator(args.source, args.source_code, args.destination)
    else:
        convert(args.source, args.source_code, args.destination)
