import pytest
import torch
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.trellis2 import mesh_encoder
from comfy.ldm.trellis2.vae import SparseTensor, SparseChannel2Spatial


def box_mesh():
    vertices = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-2., 2.) for z in (-3., 3.)])
    faces = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return vertices, faces


def test_subdivisions_reconstruct_exact_surface_coordinates():
    spatial = torch.randint(0, 32, (200, 3), generator=torch.Generator().manual_seed(7), dtype=torch.int32)
    coords = torch.unique(torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=1), dim=0)
    subs = mesh_encoder.shape_subdivisions(coords)
    current = subs[0].coords
    up = SparseChannel2Spatial(2)
    for sub in subs:
        assert torch.equal(current, sub.coords)
        x = SparseTensor(torch.zeros((len(current), 8)), current)
        current = up(x, sub).coords
    assert torch.equal(torch.unique(current, dim=0), coords)


@pytest.mark.skipif(mesh_encoder._mesh_voxel is None, reason="Build the optional CPU voxel extension")
def test_dual_grid_normalization_and_features():
    vertices, faces = box_mesh()
    original = vertices.clone()
    normalized, features, coords = mesh_encoder.mesh_to_dual_grid(vertices + 10, faces, 32)
    assert torch.equal(vertices, original)
    assert normalized.abs().max() < .5
    assert torch.allclose(normalized.amin(0), -normalized.amax(0))
    assert torch.allclose(normalized.amax(0), torch.tensor([1., 2., 3.]) * (.99999 / 6))
    assert torch.isfinite(features).all()
    assert features.shape == (len(coords), 6)
    assert set(features[:, 3:].unique().tolist()) == {-.5, .5}
    assert coords[:, 1:].min() >= 0 and coords[:, 1:].max() < 32
    assert len(torch.unique(coords, dim=0)) == len(coords)
    # Y-up dimensions (1,2,3) become Z-up (1,3,2).
    extent = coords[:, 1:].amax(0) - coords[:, 1:].amin(0)
    assert extent[1] > extent[2] > extent[0]


@pytest.mark.skipif(mesh_encoder._mesh_voxel is None, reason="Build the optional CPU voxel extension")
def test_invalid_mesh_is_rejected_before_native_memory_access():
    vertices, faces = box_mesh()
    faces[0, 0] = len(vertices)
    with pytest.raises(ValueError, match="valid triangle indices"):
        mesh_encoder.mesh_to_dual_grid(vertices, faces, 32)


def test_missing_extension_has_actionable_error(monkeypatch):
    monkeypatch.setattr(mesh_encoder, "_mesh_voxel", None)
    with pytest.raises(RuntimeError, match="build_trellis2_voxel.py"):
        mesh_encoder.mesh_to_dual_grid(*box_mesh(), 32)


@pytest.mark.skipif(mesh_encoder._mesh_voxel is None, reason="Build the optional CPU voxel extension")
def test_non_triangle_faces_rejected_before_native_memory_access():
    vertices, faces = box_mesh()
    with pytest.raises(ValueError, match="Mx3 triangle indices"):
        mesh_encoder.mesh_to_dual_grid(vertices, faces[:, :2], 32)
