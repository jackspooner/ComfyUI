import json
import shutil
import uuid

import numpy as np
from PIL import Image
import pytest
import torch

from comfy_extras import moge_geometry_io as geometry_io
from comfy_extras.nodes_moge import MoGeRender, MoGePointMapToMesh


def geometry(panorama=False):
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(6), indexing="ij")
    packet = {"points": torch.stack([xx * .1, yy * .1, torch.ones_like(xx)], -1).float()[None],
              "depth": torch.ones(1, 5, 6), "mask": torch.ones(1, 5, 6, dtype=torch.bool), "image": torch.ones(1, 5, 6, 3) * .5}
    packet["mask"][0, 0, 0] = False
    packet["points"][0, 0, 0] = float("inf")
    if not panorama:
        packet["intrinsics"] = torch.tensor([[[1., 0, .5], [0, 1.2, .5], [0, 0, 1.]]])
    return packet


@pytest.mark.parametrize("panorama", [False, True])
def test_geometry_numeric_roundtrip_and_native_mesh(tmp_path, monkeypatch, panorama):
    monkeypatch.setattr(geometry_io.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(geometry_io.folder_paths, "get_input_directory", lambda: str(tmp_path))
    packet = geometry(panorama)
    preview = MoGeRender.execute(packet, "normal_opengl").result[0]
    mesh = MoGePointMapToMesh.execute(packet, 0, 1, 0, True).result[0]
    descriptor = geometry_io.save_geometry(packet, preview, str(uuid.uuid4()), mesh)
    directory = tmp_path / descriptor["subfolder"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["depth"]["units"] == ("relative" if panorama else "estimated_meters")
    assert manifest["camera"] is None if panorama else manifest["camera"]["focalPixels"] == pytest.approx(6.)
    assert len(manifest["files"]) == 5
    with Image.open(directory / "depth.png") as image:
        assert image.mode == "I;16"
        assert np.asarray(image)[0, 0] == 0
    with np.load(directory / "geometry.npz", allow_pickle=False) as archive:
        for key in ["points", "depth", "normal", "image"]:
            assert np.isfinite(archive[key]).all()
        assert archive["points"][0, 0].tolist() == [0, 0, 0]
    shutil.copyfile(directory / "geometry.npz", tmp_path / "input.npz")
    restored = geometry_io.load_geometry("input.npz")
    assert torch.isinf(restored["points"][0, 0, 0]).all()
    assert ("intrinsics" in restored) == (not panorama)
    converted = MoGePointMapToMesh.execute(restored, 0, 1, 0, True).result[0]
    assert torch.equal(converted.faces, mesh.faces)
    torch.testing.assert_close(converted.vertices, mesh.vertices)
    assert (directory / "mesh.glb").read_bytes()[:4] == b"glTF"


def test_geometry_rejects_untrusted_archives_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry_io.folder_paths, "get_input_directory", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="path|inside"):
        geometry_io.load_geometry(str(tmp_path.parent / "outside.npz"))
    np.savez(tmp_path / "unsafe.npz", points=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        geometry_io.load_geometry("unsafe.npz")
    np.savez(tmp_path / "unknown.npz", points=np.zeros(1))
    with pytest.raises(ValueError, match="Unsupported"):
        geometry_io.load_geometry("unknown.npz")


def test_export_requires_a_valid_pixel_and_uuid(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry_io.folder_paths, "get_output_directory", lambda: str(tmp_path))
    packet = geometry()
    with pytest.raises(ValueError):
        geometry_io.save_geometry(packet, torch.zeros(1, 5, 6, 3), "../escape")
    packet["mask"][:] = False
    with pytest.raises(ValueError, match="no valid depth"):
        geometry_io.save_geometry(packet, torch.zeros(1, 5, 6, 3), str(uuid.uuid4()))
