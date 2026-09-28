"""Replace a crown's broad AI normal with a tapered cylindrical surface."""

import cv2
import numpy as np

from smg.normal import normalize_vectors


def _smooth_rows(values: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        values[np.newaxis, :].astype(np.float32), (0, 0), sigmaX=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )[0]


def _masked_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weights = mask.astype(np.float32)
    denominator = cv2.GaussianBlur(
        weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101,
    )
    result = np.empty_like(values)
    for channel in range(2):
        numerator = cv2.GaussianBlur(
            values[..., channel] * weights, (0, 0), sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        result[..., channel] = numerator / np.maximum(denominator, 1e-6)
    return result


def compose_crown_normals(vectors: np.ndarray, crown_mask: np.ndarray,
                          alpha: np.ndarray) -> np.ndarray:
    """Keep AI fine relief while giving each crown a convex cylindrical base.

    Each connected crown gets a smooth horizontal radius profile. A constant
    profile is a cylinder; a tapered profile supplies the cone's vertical
    normal. The low-frequency AI slope is replaced, while its local residual
    is added to the geometric slope. A smooth transition inside the mask joins
    crown and non-crown pixels without changing the latter.
    """
    values = np.asarray(vectors, np.float32)
    mask = np.asarray(crown_mask, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Normal vectors must have shape H x W x 3")
    if mask.shape != values.shape[:2] or alpha.shape != mask.shape:
        raise ValueError("Crown mask, normal, and alpha dimensions differ")
    result = values.copy()
    valid = mask & (alpha > 0)
    if not np.any(valid):
        return result

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8,
    )
    for component_id in range(1, count):
        x0, y0, width, height, area = stats[component_id]
        if area < 16 or width < 3 or height < 3:
            continue
        region = np.s_[y0:y0 + height, x0:x0 + width]
        component = labels[region] == component_id
        ys, xs = np.nonzero(component)
        left = np.full(height, width, np.float32)
        right = np.full(height, -1, np.float32)
        np.minimum.at(left, ys, xs)
        np.maximum.at(right, ys, xs)
        center = _smooth_rows((left + right) * 0.5, max(1.5, height * 0.04))
        radius = _smooth_rows((right - left + 1) * 0.5, max(1.5, height * 0.04))
        radius = np.maximum(radius, 1)
        center_slope = np.gradient(center)
        radius_slope = np.gradient(radius)

        horizontal = np.arange(width, dtype=np.float32)[np.newaxis, :]
        u = np.clip((horizontal - center[:, None]) / radius[:, None], -0.98, 0.98)
        front = np.sqrt(1 - u * u)
        vertical = np.clip(
            radius_slope[:, None] + center_slope[:, None] * u, -1.2, 1.2,
        )
        base = normalize_vectors(np.stack((u, vertical, front), axis=-1))

        original = normalize_vectors(values[region])
        ai_slopes = original[..., :2] / np.maximum(original[..., 2:3], 0.25)
        ai_slopes = np.clip(ai_slopes, -3, 3)
        detail_sigma = max(2.0, min(12.0, min(width, height) * 0.05))
        detail = ai_slopes - _masked_blur(ai_slopes, component, detail_sigma)
        detail_length = np.linalg.norm(detail, axis=-1, keepdims=True)
        detail *= np.minimum(1, 1.5 / np.maximum(detail_length, 1e-6))

        base_slopes = base[..., :2] / np.maximum(base[..., 2:3], 0.2)
        combined = normalize_vectors(np.dstack((
            base_slopes + detail, np.ones((height, width), np.float32),
        )))
        distance = cv2.distanceTransform(
            np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 3,
        )[1:-1, 1:-1]
        feather = max(4.0, min(14.0, min(width, height) * 0.06))
        transition = np.clip((distance - 0.5) / feather, 0, 1)
        weight = (transition * transition * (3 - 2 * transition))[..., None]
        blended = normalize_vectors(original * (1 - weight) + combined * weight)
        result_region = result[region]
        result_region[component] = blended[component]
    return result
