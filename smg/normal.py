import cv2
import numpy as np


LUMA = np.array((0.2126, 0.7152, 0.0722), np.float32)


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


def smooth_ai_vectors(vectors: np.ndarray, alpha: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smooth AI normals without bleeding transparent background into silhouettes."""
    values = normalize_vectors(vectors)
    if values.shape[:2] != alpha.shape:
        raise ValueError("Normal and alpha dimensions differ")
    if sigma < 0:
        raise ValueError("AI smoothing must not be negative")
    if sigma == 0:
        return values

    weights = alpha.astype(np.float32) / 255
    blurred_weights = cv2.GaussianBlur(
        weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    smoothed = np.empty_like(values)
    denominator = np.maximum(blurred_weights, 1e-6)
    for channel in range(3):
        numerator = cv2.GaussianBlur(
            values[..., channel] * weights, (0, 0), sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        smoothed[..., channel] = numerator / denominator
    smoothed[alpha == 0] = (0, 0, 1)
    return normalize_vectors(smoothed)


def _alpha_aware_gaussian(values: np.ndarray, weights: np.ndarray,
                          sigma: float) -> np.ndarray:
    """Blur a scalar field without treating transparent pixels as black."""
    blurred_weights = cv2.GaussianBlur(
        weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    numerator = cv2.GaussianBlur(
        values * weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    return numerator / np.maximum(blurred_weights, 1e-6)


def postprocess_ai_vectors(vectors: np.ndarray, rgba: np.ndarray,
                           smoothing: float = 1.5,
                           detail_strength: float = 0.35,
                           detail_radius: float = 2.5) -> np.ndarray:
    """Keep DSINE's broad slopes while restoring source-resolution relief.

    DSINE supplies the low-frequency surface orientation.  A band-pass height
    signal derived from the original sprite supplies only small-scale detail;
    its gradients are composed with the AI normal as height-field slopes.  All
    filtering is alpha-aware, so the silhouette never becomes a height edge.

    ``detail_strength`` is the approximate slope of the 90th-percentile detail
    and is therefore comparable across sprites with different contrast.
    """
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Normal post-processing expects RGBA uint8")
    if vectors.shape[:2] != rgba.shape[:2]:
        raise ValueError("Normal and source dimensions differ")
    if detail_strength < 0:
        raise ValueError("AI detail strength must not be negative")
    if detail_radius < 0.5:
        raise ValueError("AI detail radius must be at least 0.5")

    alpha = rgba[..., 3]
    base = smooth_ai_vectors(vectors, alpha, smoothing)
    if detail_strength == 0 or not np.any(alpha):
        return base

    weights = alpha.astype(np.float32) / 255
    # Perceptual (sRGB) luma preserves drawn grooves and highlights better than
    # linear luma here.  The difference of two alpha-aware scales removes broad
    # painted lighting, leaving details that DSINE tends to blur away.
    luma = (rgba[..., :3].astype(np.float32) / 255) @ LUMA
    fine = _alpha_aware_gaussian(luma, weights, 0.6)
    coarse = _alpha_aware_gaussian(luma, weights, float(detail_radius))
    height = fine - coarse
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    detail = np.stack((-dx, dy), axis=-1)
    magnitude = np.linalg.norm(detail, axis=-1)
    valid = alpha >= 32
    active = magnitude[valid & (magnitude > 1e-5)]
    if active.size == 0:
        return base

    reference = float(np.percentile(active, 90))
    detail *= float(detail_strength) / max(reference, 1e-5)
    # Isolated high-contrast pixels should not turn into nearly tangent normals.
    magnitude = np.linalg.norm(detail, axis=-1, keepdims=True)
    limit = max(float(detail_strength) * 3, 1e-5)
    detail *= np.minimum(1, limit / np.maximum(magnitude, 1e-6))
    detail *= np.clip(weights[..., None] * 2, 0, 1)

    # Add slopes rather than RGB values.  This exactly preserves the AI normal
    # when detail is zero and keeps both inputs in a physically meaningful basis.
    z = np.maximum(base[..., 2:3], 0.05)
    slopes = base[..., :2] / z + detail
    combined = normalize_vectors(np.dstack((slopes, np.ones(alpha.shape, np.float32))))
    combined[alpha == 0] = (0, 0, 1)
    return combined


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
