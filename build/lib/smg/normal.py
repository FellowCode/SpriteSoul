import cv2
import numpy as np


def normals_from_depth(depth: np.ndarray, alpha: np.ndarray, strength: float = 3.0,
                       convention: str = "OpenGL") -> np.ndarray:
    """Return RGBA normal pixels. Image Y increases downward; OpenGL green points up."""
    if depth.shape != alpha.shape:
        raise ValueError("Depth and alpha must have the same dimensions")
    if convention not in ("OpenGL", "DirectX"):
        raise ValueError("Unknown normal convention")
    mask = (alpha > 0).astype(np.uint8)
    if not np.any(mask):
        flat = np.zeros((*depth.shape, 4), np.uint8)
        flat[..., :3] = (128, 128, 255)
        flat[..., 3] = alpha
        return flat

    # Copy the nearest opaque value into transparent pixels before taking derivatives.
    # This avoids interpreting transparent background as a zero-height wall.
    transparent = (mask == 0).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        transparent, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    opaque_indices = np.flatnonzero(mask)
    filled = depth.astype(np.float32, copy=True)
    if np.any(transparent):
        nearest = opaque_indices[np.maximum(labels[transparent != 0] - 1, 0)]
        filled[transparent != 0] = depth.flat[nearest]

    dx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    dy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    nx = -dx * strength
    ny = dy * strength * (1 if convention == "OpenGL" else -1)
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack((nx / length, ny / length, nz / length), axis=-1)
    rgba = np.empty((*depth.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint((rgb * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba
