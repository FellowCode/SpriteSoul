"""Float32 CUDA implementation of the normal-guided AO algorithm.

Imported only when CUDA is selected. Component labelling/input validation
stay on CPU; relief reconstruction and horizon integration stay on GPU.
No runtime compiler, extra models, half precision, or CUDA toolkit required.
"""

import numpy as np
import torch


# Each horizon chunk uses O(chunk_size) memory, independent of atlas size.
RECEIVER_CHUNK_SIZE = 131072


def _edges(labels):
    return ((labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != 0),
            (labels[1:] == labels[:-1]) & (labels[1:] != 0))


def _gradient(height, labels):
    ex, ey = _edges(labels)
    dx = torch.diff(height, dim=1) * ex
    dy = torch.diff(height, dim=0) * ey
    gx, gy = torch.zeros_like(height), torch.zeros_like(height)
    cx, cy = torch.zeros_like(height), torch.zeros_like(height)
    gx[:, :-1] += dx
    gx[:, 1:] += dx
    cx[:, :-1] += ex
    cx[:, 1:] += ex
    gy[:-1] += dy
    gy[1:] += dy
    cy[:-1] += ey
    cy[1:] += ey
    return gx / cx.clamp_min(1), gy / cy.clamp_min(1)


def _guided_relief(height, normals, labels, opacity, radius, component_count):
    ex, ey = _edges(labels)
    z = normals[..., 2].clamp_min(0.2)
    sx = (-normals[..., 0] / z).clamp(-3, 3)
    sy = (normals[..., 1] / z).clamp(-3, 3)
    dx, dy = torch.diff(height, dim=1), torch.diff(height, dim=0)
    rx = (sx[:, :-1] + sx[:, 1:]) * 0.5 - dx
    ry = (sy[:-1] + sy[1:]) * 0.5 - dy
    wx = ex * torch.minimum(opacity[:, :-1], opacity[:, 1:]) / (1 + (dx / 0.75) ** 4)
    wy = ey * torch.minimum(opacity[:-1], opacity[1:]) / (1 + (dy / 0.75) ** 4)
    for residual, weights, edge_labels in ((rx, wx, labels[:, :-1]),
                                            (ry, wy, labels[:-1])):
        # Match NumPy's double-precision component accumulation. The relief
        # and final AO still use float32, without AMP/autograd allocations.
        sums = torch.bincount(edge_labels.flatten(), (residual * weights).double().flatten(),
                              minlength=component_count)
        counts = torch.bincount(edge_labels.flatten(), weights.double().flatten(),
                                minlength=component_count)
        means = (sums / counts.clamp_min(1e-8)).float()
        residual -= means[edge_labels]

    rhs = torch.zeros_like(height)
    diagonal = torch.full_like(height, 1 / np.clip(radius * 0.15, 1, 4) ** 2)
    rhs[:, :-1] -= wx * rx
    rhs[:, 1:] += wx * rx
    rhs[:-1] -= wy * ry
    rhs[1:] += wy * ry
    diagonal[:, :-1] += wx
    diagonal[:, 1:] += wx
    diagonal[:-1] += wy
    diagonal[1:] += wy
    correction = torch.zeros_like(height)
    for _ in range(48):
        total = rhs.clone()
        total[:, :-1] += wx * correction[:, 1:]
        total[:, 1:] += wx * correction[:, :-1]
        total[:-1] += wy * correction[1:]
        total[1:] += wy * correction[:-1]
        correction = total / diagonal
    return height + correction


def _integral(angle, horizontal, vertical):
    return (horizontal * (angle * 0.5 + torch.sin(2 * angle) * 0.25)
            + vertical * torch.sin(angle) ** 2 * 0.5)


def _ray_tables(radius, h, w):
    """Prepare a small sampling table; all 16 directions run in parallel."""
    angles = np.arange(16) * (2 * np.pi / 16)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    distances = np.unique(np.r_[np.arange(1, min(radius, 8) + 1),
                                np.rint(np.geomspace(min(radius, 9), radius, 12)).astype(int)])
    offsets = np.zeros((len(distances), 16, 2), np.int64)
    geometry = np.ones((len(distances), 16, 2), np.float32)
    active = np.zeros((len(distances), 16), bool)
    for direction, (ux, uy) in enumerate(directions):
        used = set()
        for index, step in enumerate(distances):
            ox, oy = int(round(float(ux) * step)), int(round(float(uy) * step))
            if (ox, oy) in used or (ox == 0 and oy == 0):
                continue
            used.add((ox, oy))
            distance = float(np.hypot(ox, oy))
            if abs(ox) >= w or abs(oy) >= h or distance > radius + 0.5:
                continue
            offsets[index, direction] = (ox, oy)
            footprint = 0.5 * (abs(ox) + abs(oy)) / distance
            geometry[index, direction] = (distance, max(0.5, distance - footprint))
            active[index, direction] = True
    keep = active.any(axis=1)
    return (torch.as_tensor(directions.astype(np.float32), device="cuda"),
            torch.as_tensor(offsets[keep], device="cuda"),
            torch.as_tensor(geometry[keep], device="cuda"),
            torch.as_tensor(active[keep], device="cuda"))


def ao_cuda(height, normals, labels, coverage, radius, strength):
    """Consume validated/scaled CPU inputs; return a CPU NumPy visibility map."""
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        return _ao_cuda(height, normals, labels, coverage, radius, strength)


def _ao_cuda(height, normals, labels, coverage, radius, strength):
    h, w = height.shape
    component_count = int(labels.max()) + 1
    # CPU indices are already sorted and avoid a CUDA nonzero/synchronization.
    receivers = torch.as_tensor(np.flatnonzero(labels), device="cuda")
    height = torch.as_tensor(height, device="cuda")
    labels = torch.as_tensor(labels.astype(np.int64), device="cuda")
    coverage = torch.as_tensor(coverage, device="cuda")
    if normals is not None:
        normals = torch.as_tensor(np.ascontiguousarray(normals), device="cuda")
        height = _guided_relief(height, normals, labels, coverage, radius, component_count)
    gx, gy = _gradient(height, labels)
    if normals is None:
        normals = torch.stack((-gx, gy, torch.ones_like(height)), dim=-1)
        normals /= torch.linalg.vector_norm(normals, dim=-1, keepdim=True)

    height, labels, coverage = height.flatten(), labels.flatten(), coverage.flatten()
    gx, gy, normals = gx.flatten(), gy.flatten(), normals.reshape(-1, 3)
    result = torch.ones(h*w, device="cuda", dtype=torch.float32)
    directions, offsets, geometry, active = _ray_tables(radius, h, w)
    ux, uy = directions[:, 0:1], directions[:, 1:2]
    for first in range(0, receivers.numel(), RECEIVER_CHUNK_SIZE):
        indices = receivers[first:first + RECEIVER_CHUNK_SIZE]
        rows, columns = indices // w, indices % w
        point_height, point_labels = height[indices], labels[indices]
        point_gx, point_gy, point_normal = gx[indices], gy[indices], normals[indices]
        nd = point_normal[:, 0] * ux - point_normal[:, 1] * uy
        nz = point_normal[:, 2]
        tangent = torch.atan(point_gx * ux + point_gy * uy)
        horizon = torch.maximum(tangent, torch.atan2(-nd, nz))
        horizon_integral = _integral(horizon, nd, nz)
        occlusion = torch.zeros_like(horizon)
        for sample in range(offsets.shape[0]):
            ox, oy = offsets[sample, :, 0:1], offsets[sample, :, 1:2]
            distance, face_distance = geometry[sample, :, 0:1], geometry[sample, :, 1:2]
            there = (indices + oy * w + ox).clamp(0, height.numel() - 1)
            valid = ((rows + oy >= 0) & (rows + oy < h)
                     & (columns + ox >= 0) & (columns + ox < w)
                     & (point_labels == labels[there]) & active[sample, :, None])
            dz = height[there] - point_height
            local_slope = (point_gx * ox + point_gy * oy) / distance
            local_tangent = torch.atan(local_slope)
            face_height = dz - local_slope * (distance - face_distance)
            # atan(x/d) equals atan2(x,d) for our strictly positive distance.
            elevation = torch.atan(face_height / face_distance)
            elevation -= local_tangent - tangent + 0.01
            raises_horizon = valid & (elevation > horizon)
            integral = _integral(elevation, nd, nz)
            blocked = (integral - horizon_integral).clamp_min(0)
            falloff = torch.exp(-2 * (face_distance ** 2 + face_height ** 2) / radius ** 2)
            occlusion += torch.where(raises_horizon, blocked * falloff * coverage[there], 0)
            horizon = torch.where(raises_horizon, elevation, horizon)
            horizon_integral = torch.where(raises_horizon, integral, horizon_integral)
        result[indices] = (1 - occlusion.sum(dim=0) * (2 / 16)).clamp(0, 1) ** strength
    # This transfer also waits for every queued CUDA operation to complete.
    return result.reshape(h, w).cpu().numpy()
