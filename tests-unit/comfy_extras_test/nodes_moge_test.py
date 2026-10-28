import json

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_extras import nodes_moge
from comfy_extras.nodes_moge import SaveMoGeDepthPNG, _encode_metric_depth


def test_encode_metric_depth_uses_runtime_host_contract():
    depth = torch.tensor([[[1.0, 2.0, 3.0, float("inf")]]])
    mask = torch.tensor([[[True, True, True, False]]])

    encoded, minimum, maximum = _encode_metric_depth(depth, mask)

    assert encoded.dtype == np.uint16
    assert encoded.tolist() == [[[1, 32768, 65535, 0]]]
    assert minimum == 1.0
    assert maximum == 3.0


def test_encode_metric_depth_constant_field_uses_nonzero_sentinel():
    depth = torch.tensor([[[2.0, 2.0, 2.0]]])
    mask = torch.tensor([[[True, True, False]]])

    encoded, minimum, maximum = _encode_metric_depth(depth, mask)

    assert encoded.tolist() == [[[1, 1, 0]]]
    assert minimum == maximum == 2.0


def test_encode_metric_depth_rejects_empty_valid_mask():
    with pytest.raises(ValueError, match="no valid depth pixels"):
        _encode_metric_depth(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool))


def test_save_moge_depth_png_writes_true_16_bit_output(tmp_path, monkeypatch):
    depth = torch.tensor([[[1.0, 2.0], [3.0, float("inf")]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    geometry = {"depth": depth, "mask": mask}

    monkeypatch.setattr(nodes_moge.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        nodes_moge.folder_paths,
        "get_save_image_path",
        lambda prefix, output_dir, width, height: (str(tmp_path), "depth", 7, "moge_depth", prefix),
    )

    output = SaveMoGeDepthPNG.execute(geometry, "moge_depth/runtime")
    png_path = tmp_path / "depth_00007_.png"
    sidecar_path = tmp_path / "depth_00007_.json"

    with Image.open(png_path) as image:
        pixels = np.asarray(image)
        assert image.mode == "I;16"
        assert image.text["moge_depth_min"] == "1"
        assert image.text["moge_depth_max"] == "3"
    assert pixels.dtype == np.uint16
    assert pixels.tolist() == [[1, 32768], [65535, 0]]

    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == {
        "format": "moge-depth-v1",
        "png": "depth_00007_.png",
        "width": 2,
        "height": 2,
        "min": 1.0,
        "max": 3.0,
        "dtype": "uint16",
        "invalid_value": 0,
        "encoding": "nonzero_linear",
    }
    assert output.ui["moge_depth"][0]["min"] == 1.0
    assert output.ui["moge_depth"][0]["max"] == 3.0
