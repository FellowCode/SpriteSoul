import cv2
import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize float XYZ normals; invalid vectors face the viewer."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Expected H x W x 3 normal vectors")
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(lengths > 1e-6, values / np.maximum(lengths, 1e-6),
                    np.array((0, 0, 1), np.float32)).astype(np.float32)


def dsine_to_opengl(vectors: np.ndarray) -> np.ndarray:
    """Convert DSINE camera XYZ (right, down, front) to OpenGL texture XYZ."""
    converted = np.asarray(vectors, np.float32).copy()
    converted[..., 1] *= -1
    return normalize_vectors(converted)


def orient_ai_vectors(vectors: np.ndarray, invert_x: bool = False,
                      invert_y: bool = False) -> np.ndarray:
    """Adjust AI relief directions without changing the cached OpenGL vectors."""
    corrected = np.asarray(vectors, np.float32).copy()
    if invert_x:
        corrected[..., 0] *= -1
    if invert_y:
        corrected[..., 1] *= -1
    return corrected


def encode_normals(vectors: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if vectors.shape[:2] != alpha.shape:
        raise ValueError("Normal and alpha dimensions differ")
    result = np.empty((*alpha.shape, 4), np.uint8)
    result[..., :3] = np.clip(np.rint((normalize_vectors(vectors) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    result[..., 3] = alpha
    return result


def ai_normal(vectors: np.ndarray, alpha: np.ndarray, convention: str = "OpenGL") -> np.ndarray:
    """Encode cached OpenGL AI vectors, optionally flipping Y for DirectX."""
    if convention not in ("OpenGL", "DirectX"):
        raise ValueError("Unknown normal convention")
    mapped = np.asarray(vectors, np.float32).copy()
    if convention == "DirectX":
        mapped[..., 1] *= -1
    return encode_normals(mapped, alpha)


def hybrid_normal(depth_normal: np.ndarray, ai_vectors: np.ndarray, alpha: np.ndarray,
                  influence: float, convention: str = "OpenGL") -> np.ndarray:
    if not 0 <= influence <= 1:
        raise ValueError("AI influence must be between 0 and 1")
    if depth_normal.shape != (*alpha.shape, 4) or ai_vectors.shape != (*alpha.shape, 3):
        raise ValueError("Normal and alpha dimensions differ")
    if influence == 0:
        return depth_normal.copy()
    if influence == 1:
        return ai_normal(ai_vectors, alpha, convention)
    depth_xyz = normalize_vectors(depth_normal[..., :3].astype(np.float32) / 127.5 - 1)
    ai_xyz = ai_vectors.astype(np.float32).copy()
    if convention == "DirectX":
        ai_xyz[..., 1] *= -1
    return encode_normals((1 - influence) * depth_xyz + influence * normalize_vectors(ai_xyz), alpha)


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
