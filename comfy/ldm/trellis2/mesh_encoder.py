"""TRELLIS.2 shape encoder, adapted from Microsoft's MIT-licensed sc_vaes.

Uses the same native sparse operations as the existing shape/texture decoders.
See voxelize/LICENSE and docs/trellis2-mesh-texturing.md for source attribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.latent_formats
import comfy.model_prefetch
from .vae import SparseTensor, SparseLinear, SparseConv3d, SparseConvNeXtBlock3d, SparseSpatial2Channel, ops

try:
    from . import _mesh_voxel
except ImportError:
    _mesh_voxel = None


def mesh_to_dual_grid(vertices, faces, resolution):
    if _mesh_voxel is None:
        raise RuntimeError("Trellis mesh encoder needs its CPU extension. Run scripts/build_trellis2_voxel.py with EIGEN_INCLUDE_DIR set; see docs/trellis2-mesh-texturing.md.")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Trellis mesh encoder requires Nx3 vertices and Mx3 triangle indices.")
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("Trellis mesh encoder requires a positive integer resolution.")
    vertices = vertices.float().cpu().contiguous()
    faces = faces.to(device="cpu", dtype=torch.int32).contiguous()
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError("Trellis mesh encoder received an empty mesh.")
    if not torch.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Trellis mesh encoder requires finite vertices and valid triangle indices.")
    lo, hi = vertices.amin(0), vertices.amax(0)
    extent = (hi - lo).max()
    if extent <= 0:
        raise ValueError("Trellis mesh encoder received a mesh with zero extent.")
    normalized = (vertices - (lo + hi) * 0.5) * (0.99999 / extent)
    # glTF Y-up -> TRELLIS training Z-up. Return the normalized Y-up mesh for baking.
    z_up = torch.stack((normalized[:, 0], -normalized[:, 2], normalized[:, 1]), dim=-1)
    coords, dual, intersected = _mesh_voxel.mesh_to_flexible_dual_grid(
        (z_up + 0.5).contiguous(), faces, torch.full((3,), 1.0 / resolution),
        torch.tensor([[0, 0, 0], [resolution] * 3], dtype=torch.int32),
        1.0, 0.2, 0.01, False,
    )
    if coords.shape[0] == 0:
        raise ValueError("Mesh has no surface intersections at the selected Trellis resolution.")
    features = torch.cat((dual * resolution - coords - 0.5, intersected.float() - 0.5), dim=-1)
    coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=-1)
    return normalized, features, coords


def shape_subdivisions(coords):
    """Exact occupancy at each encoder scale, in texture decoder order."""
    subs = []
    for _ in range(4):
        parents = coords.clone()
        parents[:, 1:] //= 2
        parents, inverse = torch.unique(parents, dim=0, return_inverse=True)
        child = coords[:, 1:] % 2
        child = child[:, 0] + 2 * child[:, 1] + 4 * child[:, 2]
        flags = torch.zeros((len(parents), 8), dtype=torch.bool, device=coords.device)
        flags[inverse, child.long()] = True
        subs.append(SparseTensor(feats=flags, coords=parents))
        coords = parents
    # Decoder expansion emits parent-major, x-fastest children, not lexicographic
    # coordinates. Align each following mask with that exact emitted row order.
    subs.reverse()
    current = subs[0].coords
    aligned = []
    for sub in subs:
        _, mapping = torch.unique(torch.cat((sub.coords, current)), dim=0, return_inverse=True)
        inverse = torch.argsort(mapping[:len(sub.coords)])
        flags = sub.feats[inverse[mapping[len(sub.coords):]]]
        aligned.append(SparseTensor(feats=flags, coords=current))
        parent, child = flags.nonzero(as_tuple=True)
        current = current[parent].clone()
        current[:, 1:] *= 2
        for axis in range(3):
            current[:, axis + 1] += (child // (2 ** axis) % 2).to(current.dtype)
    return aligned


class SparseResBlockS2C3d(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.norm1 = ops.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = ops.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels // 8, 3)
        self.conv2 = SparseConv3d(out_channels, out_channels, 3)
        self.updown = SparseSpatial2Channel(2)

    def forward(self, x):
        h = x.replace(F.silu(self.norm1(x.feats)))
        h = self.updown(self.conv1(h))
        skip = self.updown(x)
        h = h.replace(F.silu(self.norm2(h.feats)))
        h = self.conv2(h)
        return h.replace(h.feats + skip.feats.reshape(len(skip.feats), self.out_channels, -1).mean(-1))


class ShapeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [64, 128, 256, 512, 1024]
        self.input_layer = SparseLinear(6, channels[0])
        self.to_latent = SparseLinear(channels[-1], 64)
        self.blocks = nn.ModuleList()
        for i, count in enumerate([0, 4, 8, 16, 4]):
            stage = nn.ModuleList(SparseConvNeXtBlock3d(channels[i]) for _ in range(count))
            if i < 4:
                stage.append(SparseResBlockS2C3d(channels[i], channels[i + 1]))
            self.blocks.append(stage)

    def encode_shape_slat(self, features, coords, resolution, transformer_options):
        h = self.input_layer(SparseTensor(feats=features, coords=coords))
        blocks = [block for stage in self.blocks for block in stage]
        prefetch = comfy.model_prefetch.make_prefetch_queue(blocks, features.device, transformer_options)
        # Follow the native sparse VAE execution path. Its coordinate caches and
        # VRAM-dependent gathers do not form a repeatable transformer-block plan.
        for block in blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, features.device, block, features.dtype)
            h = block(h)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, features.device, None)
        h = self.to_latent(h.replace(F.layer_norm(h.feats, h.feats.shape[-1:])))
        mean = h.feats[:, :32].float()
        mean = comfy.latent_formats.Trellis2ShapeSLAT().process_in(mean)
        return {"samples": mean.T[None, :, :, None], "coords": h.coords,
                "coord_resolution": resolution // 16, "model_frame": "z_up", "type": "trellis2"}
