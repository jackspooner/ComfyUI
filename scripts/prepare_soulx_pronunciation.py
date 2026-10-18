#!/usr/bin/env python3
"""Extract offline G2p-en/CMU assets without importing NLTK or installing packages."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from safetensors.torch import save_file


def prepare(source, destination):
    hashes = {"g2p_en-2.1.0-py3-none-any.whl": "2a7aabf1fc7f270fcc3349881407988c9245173c2413debbe5432f4a4f31319f",
              "cmudict-1.1.3-py3-none-any.whl": "e4d421341bf9fa774bcded8e7d6c5d73a1bf8f88edbe129207713850abac4995"}
    if destination.exists():
        raise FileExistsError(destination)
    for name, expected in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source wheel hash mismatch: {name}")
    with zipfile.ZipFile(source / "g2p_en-2.1.0-py3-none-any.whl") as archive:
        tree = ast.parse(archive.read("g2p_en/g2p.py"))
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in init.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "phonemes" for t in n.targets))
        phonemes = ast.literal_eval(assignment.value.left) + ast.literal_eval(assignment.value.right)
        weights = np.load(io.BytesIO(archive.read("g2p_en/checkpoint20.npz")), allow_pickle=False)
        state = {key: torch.from_numpy(weights[key].copy()) for key in weights.files}
        destination.mkdir(parents=True)
        save_file(state, destination / "pronunciation.safetensors", metadata={"phonemes": json.dumps(phonemes), "source": "g2p_en 2.1.0"})
        (destination / "G2P-LICENSE.txt").write_bytes(archive.read("g2p_en-2.1.0.dist-info/LICENSE.txt"))
    with zipfile.ZipFile(source / "cmudict-1.1.3-py3-none-any.whl") as archive:
        (destination / "cmudict.dict").write_bytes(archive.read("cmudict/data/cmudict.dict"))
        (destination / "CMU-LICENSE.txt").write_bytes(archive.read("cmudict/data/LICENSE"))
    manifest = {"sources": hashes, "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination)
