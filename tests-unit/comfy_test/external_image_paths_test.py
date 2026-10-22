import ast
import hashlib
import json
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from PIL import Image

import folder_paths
from comfy.image_paths import args, resolve_image_path


def source_definition(filename, name, namespace):
    tree = ast.parse((Path(__file__).resolve().parents[2] / filename).read_text())
    definition = next(n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)) and n.name == name)
    definition.decorator_list = []
    exec(compile(ast.Module(body=[definition], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def external(tmp_path, monkeypatch):
    monkeypatch.setattr(args, "allow_external_image_paths", True)
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path / "input"))
    (tmp_path / "input").mkdir()
    filename = tmp_path / 'original ü & #.png'
    Image.new("RGBA", (4, 3), (120, 40, 30, 80)).save(filename)
    return filename


def test_external_resolution_and_generic_containment(external, monkeypatch):
    assert resolve_image_path(str(external)) == str(external)
    with pytest.raises(ValueError):
        folder_paths.get_annotated_filepath(str(external))
    with pytest.raises(ValueError):
        resolve_image_path("../" + external.name)
    alias = external.with_name("alias.png")
    alias.symlink_to(external)
    assert resolve_image_path(str(alias)) == str(external)
    monkeypatch.setattr(args, "allow_external_image_paths", False)
    with pytest.raises(PermissionError):
        resolve_image_path(str(external))


def test_node_validation_and_content_change(external):
    loader = source_definition("nodes.py", "LoadImage", {
        "resolve_image_path": resolve_image_path, "os": os, "hashlib": hashlib,
    })
    assert loader.VALIDATE_INPUTS(str(external)) is True
    before = loader.IS_CHANGED(str(external))
    Image.new("RGB", (4, 3), "blue").save(external)
    assert loader.IS_CHANGED(str(external)) != before
    external.unlink()
    assert "Invalid image file" in loader.VALIDATE_INPUTS(str(external))


def test_rejects_non_images_and_directories(external):
    with pytest.raises(ValueError):
        resolve_image_path(str(external.parent))
    external.write_text("<html>not an image</html>")
    with pytest.raises(OSError):
        resolve_image_path(str(external))


def test_load_save_metadata_restores_live_source(external, monkeypatch):
    monkeypatch.setattr(args, "cpu", True)
    import nodes
    import torch

    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_dtype", lambda: torch.float32)
    output = external.parent / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "output_directory", str(output))
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": str(external)}}}
    images, mask = nodes.LoadImage().load_image(str(external))
    assert images.shape == (1, 3, 4, 3)
    assert mask.shape == (1, 3, 4)
    saved = nodes.SaveImage().save_images(images, prompt=prompt)["ui"]["images"][0]
    with Image.open(output / saved["filename"]) as image:
        restored = json.loads(image.info["prompt"])["1"]["inputs"]["image"]
    assert restored == str(external)
    assert nodes.LoadImage.VALIDATE_INPUTS(restored) is True
    Image.new("RGB", (4, 3), "blue").save(external)
    updated, _ = nodes.LoadImage().load_image(restored)
    assert not torch.equal(images, updated)
    assert list((external.parent / "input").iterdir()) == []


@pytest.mark.asyncio
async def test_view_external_channels_and_failures(external, monkeypatch):
    # Exercise the actual route closure without starting a server or importing GPU services.
    view = source_definition("server.py", "view_image", {
        "resolve_image_path": resolve_image_path, "folder_paths": folder_paths,
        "os": os, "Image": Image, "web": web, "BytesIO": BytesIO, "mimetypes": mimetypes,
    })
    async def request(**extra):
        return await view(make_mocked_request("GET", "/view?" + urlencode({"filename": str(external), **extra})))
    for query in ({"preview": "webp;80"}, {"channel": "rgb"}, {"channel": "a"}):
        response = await request(**query)
        assert response.status == 200
        assert Image.open(BytesIO(response.body)).size == (4, 3)
    assert isinstance(await request(), web.FileResponse)
    monkeypatch.setattr(args, "allow_external_image_paths", False)
    assert (await request()).status == 403
    monkeypatch.setattr(args, "allow_external_image_paths", True)
    external.write_text("not an image")
    assert (await request()).status == 400
    external.unlink()
    assert (await request()).status == 404
