"""Pickle-free, bounded MoGe geometry interchange and native artifact export."""

import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch

import folder_paths


MAX_PIXELS = 4_194_304
MAX_BYTES = 512 * 1024 * 1024


def load_geometry(filename):
    path = folder_paths.get_annotated_filepath(filename)
    root = Path(folder_paths.get_input_directory()).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > MAX_BYTES:
        raise ValueError("Geometry must be a bounded file inside the input directory")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 10 or sum(f.file_size for f in archive.infolist()) > MAX_BYTES:
            raise ValueError("Geometry archive exceeds decoded bounds")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected = {"schema_version", "projection", "normal_source", "points", "depth", "mask", "normal", "image"}
    if set(arrays) not in (expected, expected | {"intrinsics"}):
        raise ValueError("Unsupported MoGe geometry fields")
    version = arrays.pop("schema_version")
    if version.shape != () or version.item() != 1:
        raise ValueError("Unsupported geometry version")
    projection = arrays.pop("projection").item()
    normal_source = arrays.pop("normal_source").item()
    if normal_source not in ("predicted", "point_gradients"):
        raise ValueError("Invalid normal source")
    if projection not in ("perspective", "panorama") or ("intrinsics" in arrays) != (projection == "perspective"):
        raise ValueError("Projection and intrinsics disagree")
    depth = arrays["depth"]
    if depth.ndim != 2 or min(depth.shape) < 3 or depth.size > MAX_PIXELS:
        raise ValueError("Geometry dimensions exceed bounds")
    height, width = depth.shape
    for key, array in arrays.items():
        shape = (3, 3) if key == "intrinsics" else (height, width, 3) if key in ("points", "normal", "image") else (height, width)
        dtype = np.dtype("uint8") if key == "mask" else np.dtype("float32")
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"Invalid geometry array: {key}")
    mask = arrays["mask"].astype(bool)
    if not np.isin(arrays["mask"], [0, 1]).all() or not mask.any() or (depth[mask] <= 0).any():
        raise ValueError("Invalid geometry mask or depth")
    if (arrays["image"] < 0).any() or (arrays["image"] > 1).any():
        raise ValueError("Invalid geometry image")
    if projection == "perspective":
        k = arrays["intrinsics"]
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid camera intrinsics")
    result = {key: torch.from_numpy(np.array(value, copy=True)).unsqueeze(0) for key, value in arrays.items()}
    result["mask"] = result["mask"].bool()
    result["points"] = result["points"].masked_fill(~result["mask"].unsqueeze(-1), float("inf"))
    result["depth"] = result["depth"].masked_fill(~result["mask"], float("inf"))
    result["normal_source"] = normal_source
    return result


def save_geometry(geometry, normal_preview, run_id, mesh=None):
    from comfy_extras.nodes_moge import _encode_metric_depth, _normals_from_points, MoGeGeometryToFOV
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

    run_id = uuid.UUID(run_id).hex
    points = geometry["points"].detach().float().cpu()
    depth = geometry["depth"].detach().float().cpu()
    if depth.ndim != 3 or depth.shape[0] != 1 or min(depth.shape[1:]) < 3 or depth[0].numel() > MAX_PIXELS:
        raise ValueError("Export requires one geometry map of at most 4 megapixels")
    mask = geometry["mask"].detach().bool().cpu() & torch.isfinite(points).all(-1) & torch.isfinite(depth) & (depth > 0)
    encoded, minimum, maximum = _encode_metric_depth(depth[0], mask[0])
    normal = geometry.get("normal")
    normal = normal.detach().float().cpu() if normal is not None else _normals_from_points(points)
    normal = torch.nn.functional.normalize(torch.nan_to_num(normal), dim=-1)
    normal_valid = mask & torch.isfinite(normal).all(-1) & (normal.norm(dim=-1) > 0.5)
    projection = "perspective" if "intrinsics" in geometry else "panorama"
    normal_source = geometry.get("normal_source", "predicted" if "normal" in geometry else "point_gradients")
    arrays = {"schema_version": np.array(1, dtype=np.uint8), "projection": np.array(projection), "normal_source": np.array(normal_source),
              "points": torch.where(mask[..., None], points, 0)[0].numpy(),
              "depth": torch.where(mask, depth, 0)[0].numpy(), "mask": mask[0].numpy().astype(np.uint8),
              "normal": torch.where(normal_valid[..., None], normal, 0)[0].numpy(),
              "image": geometry["image"][0].detach().float().cpu().numpy()}
    height, width = encoded.shape
    camera = None
    if projection == "perspective":
        k = geometry["intrinsics"][0].detach().float().cpu().numpy()
        arrays["intrinsics"] = k
        camera = {"intrinsicsNormalized": k.tolist(), "intrinsicsPixels": (k * np.array([[width], [height], [1]])).tolist(),
                  "horizontalFovDegrees": MoGeGeometryToFOV.execute(geometry, "horizontal", "degrees").result[0],
                  "verticalFovDegrees": MoGeGeometryToFOV.execute(geometry, "vertical", "degrees").result[0],
                  "focalPixels": float(k[1, 1]) * height}
    if any(not np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in "fiu"):
        raise ValueError("Geometry contains nonfinite numeric data")
    output_root = Path(folder_paths.get_output_directory()).resolve()
    directory = output_root / "moge_geometry" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(output_root):
        raise ValueError("Geometry export escapes output root")
    files = []

    def record(name, role, mime):
        path = directory / name
        files.append({"filename": name, "role": role, "mimeType": mime, "byteSize": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    np.savez_compressed(directory / "geometry.npz", **arrays)
    record("geometry.npz", "point_map", "application/octet-stream")
    Image.fromarray(encoded).save(directory / "depth.png")
    record("depth.png", "depth", "image/png")
    Image.fromarray(mask[0].numpy().astype(np.uint8) * 255).save(directory / "mask.png")
    record("mask.png", "mask", "image/png")
    preview = normal_preview[0].detach().float().cpu().numpy().copy()
    if preview.shape != (height, width, 3) or not np.isfinite(preview).all():
        raise ValueError("Invalid normal preview")
    preview[~normal_valid[0].numpy()] = 0
    Image.fromarray((preview.clip(0, 1) * 255).round().astype(np.uint8)).save(directory / "normal.png")
    record("normal.png", "normal", "image/png")
    if mesh is not None:
        data = mesh_item_to_glb_bytes(mesh, 0)
        if not data:
            raise ValueError("MoGe mesh is empty")
        (directory / "mesh.glb").write_bytes(data)
        record("mesh.glb", "mesh", "model/gltf-binary")
    manifest = {"schemaVersion": 1, "kind": "moge_geometry", "projection": projection, "width": width, "height": height,
                "depth": {"min": minimum, "max": maximum, "encoding": "0=invalid;1..65535=linear", "quantity": "camera_z" if camera else "radial_distance",
                          "units": "estimated_meters" if camera else "relative"}, "camera": camera,
                "validPixels": int(mask.sum()), "normalSource": normal_source,
                "arrays": {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in arrays.items()}, "files": files}
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    os.replace(temporary, directory / "manifest.json")
    return {"filename": "manifest.json", "subfolder": f"moge_geometry/{run_id}", "type": "output"}
