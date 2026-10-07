"""Normal-guided, cosine-weighted horizon AO for orthographic sprite relief.

Depth is a height field (larger = nearer). Normals are float OpenGL XYZ,
not encoded RGB. This is a finite-radius 2.5D approximation, not mesh AO.
"""

import cv2
import numpy as np

from smg.normal import normalize_vectors


GPU_MIN_PIXELS = 4096


def _cuda_runtime():
    """Optional, lazy PyTorch import: CPU-only installations keep working."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch if torch.cuda.is_available() else None


def _edges(labels):
    return ((labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != 0),
            (labels[1:] == labels[:-1]) & (labels[1:] != 0))


def _gradient(height, labels):
    """One-sided at alpha boundaries, central inside; never read background."""
    ex, ey = _edges(labels)
    dx = np.diff(height, axis=1) * ex
    dy = np.diff(height, axis=0) * ey
    gx, gy = np.zeros_like(height), np.zeros_like(height)
    cx, cy = np.zeros_like(height), np.zeros_like(height)
    gx[:, :-1] += dx
    gx[:, 1:] += dx
    cx[:, :-1] += ex
    cx[:, 1:] += ex
    gy[:-1] += dy
    gy[1:] += dy
    cy[:-1] += ey
    cy[1:] += ey
    return gx / np.maximum(cx, 1), gy / np.maximum(cy, 1)


def _guided_relief(height, normals, labels, opacity, radius):
    """Screened Poisson correction: depth anchors shape, normals supply detail.

    Remove each sprite's mean slope mismatch so a constant tilted normal on
    flat depth cannot introduce a ramp. Depth discontinuities reduce the edge
    weights to preserve overlaps that normals alone cannot reconstruct.
    Neumann boundaries prevent relief leaking across alpha or atlas islands.
    """
    ex, ey = _edges(labels)
    z = np.maximum(normals[..., 2], 0.2)
    sx = np.clip(-normals[..., 0] / z, -3, 3)
    sy = np.clip(normals[..., 1] / z, -3, 3)
    dx, dy = np.diff(height, axis=1), np.diff(height, axis=0)
    rx = (sx[:, :-1] + sx[:, 1:]) * 0.5 - dx
    ry = (sy[:-1] + sy[1:]) * 0.5 - dy
    wx = ex * np.minimum(opacity[:, :-1], opacity[:, 1:]) / (1 + (dx / 0.75) ** 4)
    wy = ey * np.minimum(opacity[:-1], opacity[1:]) / (1 + (dy / 0.75) ** 4)
    for residual, weights, edge_labels in ((rx, wx, labels[:, :-1]),
                                            (ry, wy, labels[:-1])):
        sums = np.bincount(edge_labels.ravel(), (residual * weights).ravel())
        counts = np.bincount(edge_labels.ravel(), weights.ravel(), minlength=len(sums))
        means = sums / np.maximum(counts, 1e-8)
        residual -= means[edge_labels]

    rhs = np.zeros_like(height)
    diagonal = np.full_like(height, 1 / np.clip(radius * 0.15, 1, 4) ** 2)
    rhs[:, :-1] -= wx * rx
    rhs[:, 1:] += wx * rx
    rhs[:-1] -= wy * ry
    rhs[1:] += wy * ry
    diagonal[:, :-1] += wx
    diagonal[:, 1:] += wx
    diagonal[:-1] += wy
    diagonal[1:] += wy
    correction = np.zeros_like(height)
    for _ in range(48):
        total = rhs.copy()
        total[:, :-1] += wx * correction[:, 1:]
        total[:, 1:] += wx * correction[:, :-1]
        total[:-1] += wy * correction[1:]
        total[1:] += wy * correction[:-1]
        correction = total / diagonal
    return height + correction


def _integral(angle, horizontal, vertical):
    # Integral of (n_d cos(e) + n_z sin(e)) cos(e) over elevation e.
    return (horizontal * (angle * 0.5 + np.sin(2 * angle) * 0.25)
            + vertical * np.sin(angle) ** 2 * 0.5)


def ao_from_depth(depth: np.ndarray, alpha: np.ndarray, radius: int = 24,
                  strength: float = 2.0, *, normals: np.ndarray | None = None,
                  height_scale: float | None = None, device: str = "auto") -> np.ndarray:
    """Return float32 visibility in [0, 1]; transparent pixels stay white.

    Optional OpenGL normals refine relief and orient the occlusion hemisphere.
    Without them, geometric normals come from depth. ``height_scale`` converts
    normalized depth to pixel units; by default each sprite uses 1/4 of its
    bounding-box span, making relief independent of atlas padding. Strength is
    a final visibility exponent; it does not change geometry or sampling.
    ``device`` is ``auto``, ``cpu`` or ``cuda``. Auto uses CUDA for maps of
    at least 4096 pixels when available, and falls back to CPU on GPU OOM.
    """
    height = np.asarray(depth, np.float32)
    opacity = np.asarray(alpha)
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Устройство AO должно быть auto, cpu или cuda")
    if height.ndim != 2 or opacity.shape != height.shape or not height.size:
        raise ValueError("Depth и alpha должны иметь одинаковый двухмерный размер")
    if (not isinstance(radius, (int, np.integer)) or radius < 1
            or strength < 0 or not np.isfinite(strength)):
        raise ValueError("Неверные параметры AO")
    if height_scale is not None and (height_scale <= 0 or not np.isfinite(height_scale)):
        raise ValueError("Масштаб глубины AO должен быть положительным")
    if not np.isfinite(opacity).all() or np.any(opacity < 0) or np.any(opacity > 255):
        raise ValueError("Alpha должна быть в диапазоне 0–255")
    visible = opacity > 0
    if not np.isfinite(height[visible]).all():
        raise ValueError("Depth содержит нечисловые значения")
    if normals is not None:
        normals = np.asarray(normals, np.float32)
        if normals.shape != (*height.shape, 3) or not np.isfinite(normals[visible]).all():
            raise ValueError("Normal должна содержать конечные OpenGL XYZ-векторы размера Depth")
        normals = normalize_vectors(np.where(visible[..., None], normals, (0, 0, 1)))
        normals = normalize_vectors(np.dstack((normals[..., :2], np.maximum(normals[..., 2], 0.05))))
    result = np.ones(height.shape, np.float32)
    if not np.any(visible) or strength == 0:
        return result

    _, labels, stats, _ = cv2.connectedComponentsWithStats(visible.astype(np.uint8), connectivity=8)
    scale = np.maximum(1, np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) * 0.25)
    height = np.where(visible, height, 0) * (scale[labels] if height_scale is None else height_scale)
    height = height.astype(np.float32)
    coverage = opacity.astype(np.float32) / 255
    if device == "cuda" or (device == "auto" and height.size >= GPU_MIN_PIXELS):
        torch = _cuda_runtime()
        if torch is None and device == "cuda":
            raise RuntimeError("Для AO на GPU необходимы CUDA-сборка PyTorch и доступная NVIDIA GPU")
        if torch is not None:
            from smg.ao_gpu import ao_cuda

            try:
                return ao_cuda(height, normals, labels, coverage, radius, strength)
            except torch.cuda.OutOfMemoryError as exc:
                if device == "cuda":
                    raise
                # Release failed GPU-frame temporaries before the CPU fallback.
                exc.__traceback__ = None
                torch.cuda.empty_cache()
    if normals is not None:
        height = _guided_relief(height, normals, labels, coverage, radius)
    gx, gy = _gradient(height, labels)
    if normals is None:
        normals = normalize_vectors(np.dstack((-gx, gy, np.ones_like(height))))

    h, w = height.shape
    # Evaluate only visible receivers, avoiding expensive trigonometry over
    # transparent atlas padding. Sorted indices keep gathers spatially local.
    receivers = np.flatnonzero(visible)
    rows, columns = np.divmod(receivers, w)
    receiver_height = height.ravel()[receivers]
    receiver_labels = labels.ravel()[receivers]
    gx, gy = gx.ravel()[receivers], gy.ravel()[receivers]
    normals = normals.reshape(-1, 3)[receivers]
    height, labels, coverage = height.ravel(), labels.ravel(), coverage.ravel()
    # Dense contact samples plus a logarithmic far field. No randomized noise.
    distances = np.unique(np.r_[np.arange(1, min(radius, 8) + 1),
                                np.rint(np.geomspace(min(radius, 9), radius, 12)).astype(int)])
    occlusion = np.zeros(receivers.size, np.float32)
    direction_count = 16
    for angle in np.arange(direction_count) * (2 * np.pi / direction_count):
        ux, uy = float(np.cos(angle)), float(np.sin(angle))
        nd = normals[:, 0] * ux - normals[:, 1] * uy
        nz = normals[:, 2]
        tangent = np.arctan(gx * ux + gy * uy)
        normal_tangent = np.arctan2(-nd, nz)
        horizon = np.maximum(tangent, normal_tangent)
        horizon_integral = _integral(horizon, nd, nz)
        used = set()
        for step in distances:
            ox, oy = int(round(ux * step)), int(round(uy * step))
            if (ox, oy) in used or (ox == 0 and oy == 0):
                continue
            used.add((ox, oy))
            distance = float(np.hypot(ox, oy))
            if abs(ox) >= w or abs(oy) >= h or distance > radius + 0.5:
                continue
            there = np.clip(receivers + oy * w + ox, 0, height.size - 1)
            valid = ((rows + oy >= 0) & (rows + oy < h)
                     & (columns + ox >= 0) & (columns + ox < w)
                     & (receiver_labels == labels[there]))
            dz = height[there] - receiver_height
            # Rounded rays need their actual tangent to keep inclined planes white.
            local_slope = (gx * ox + gy * oy) / distance
            local_tangent = np.arctan(local_slope)
            # A sampled texel has a footprint, not zero area: its near face is
            # half a texel closer along each ray axis. Extend the local tangent
            # by the same amount so this cannot create plane self-occlusion.
            footprint = 0.5 * (abs(ox) + abs(oy)) / distance
            face_distance = max(0.5, distance - footprint)
            face_height = dz - local_slope * (distance - face_distance)
            elevation = np.arctan2(face_height, face_distance)
            elevation -= local_tangent - tangent + 0.01
            raises_horizon = valid & (elevation > horizon)
            integral = _integral(elevation, nd, nz)
            blocked = np.maximum(0, integral - horizon_integral)
            falloff = np.exp(-2 * (face_distance ** 2 + face_height ** 2) / radius ** 2)
            occlusion += np.where(raises_horizon, blocked * falloff * coverage[there], 0)
            horizon = np.where(raises_horizon, elevation, horizon)
            horizon_integral = np.where(raises_horizon, integral, horizon_integral)

    visibility = np.clip(1 - occlusion * (2 / direction_count), 0, 1)
    result.ravel()[receivers] = visibility ** strength
    return result
