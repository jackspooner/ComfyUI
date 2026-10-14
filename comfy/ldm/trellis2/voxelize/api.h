/*
 * O-Voxel Convertion API
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>


/**
 * Extract flexible dual grid from a triangle mesh.
 *
 * @param vertices: Tensor of shape (N, 3) containing vertex positions.
 * @param faces: Tensor of shape (M, 3) containing triangle vertex indices.
 * @param voxel_size: Tensor of shape (3,) containing the voxel size in each dimension.
 * @param grid_range: Tensor of shape (2, 3) containing the minimum and maximum coordinates of the grid range.
 * @param face_weight: Weight for the face edges in the QEM computation.
 * @param boundary_weight: Weight for the boundary edges in the QEM computation.
 * @param regularization_weight: Regularization factor to apply to the QEM matrices.
 * @param timing: Boolean flag to indicate whether to print timing information.
 *
 * @return a tuple ((x, y, z), vertices, intersected, faces) containing the remeshed vertices and the corresponding voxel grid.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_to_flexible_dual_grid_cpu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    bool timing
);
