from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSettings:
    invert: bool = False
    strength: float = 1.0
    contrast: float = 1.0
    smooth: float = 0.0


def normalize_depth(raw: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    valid = raw[alpha > 0]
    if valid.size == 0:
        return np.zeros(raw.shape, np.float32)
    lo, hi = np.percentile(valid, [2, 98])
    if hi - lo < 1e-6:
        return np.full(raw.shape, 0.5, np.float32)
    return np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0, 1).astype(np.float32)


def process_depth(base: np.ndarray, alpha: np.ndarray, settings: DepthSettings) -> np.ndarray:
    depth = np.asarray(base, np.float32).copy()
    if settings.invert:
        depth = 1 - depth
    depth = (depth - 0.5) * settings.contrast * settings.strength + 0.5
    if settings.smooth > 0:
        sigma = settings.smooth
        # Normalized convolution prevents transparent background from darkening edges.
        weight = cv2.GaussianBlur((alpha > 0).astype(np.float32), (0, 0), sigma)
        blurred = cv2.GaussianBlur(depth * (alpha > 0), (0, 0), sigma) / np.maximum(weight, 1e-6)
        depth[alpha > 0] = blurred[alpha > 0]
    return np.clip(depth, 0, 1)
