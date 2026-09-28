"""Approximate ambient occlusion from an edited, normalized height/depth map."""

import cv2
import numpy as np


def ao_from_depth(depth: np.ndarray, alpha: np.ndarray, radius: int = 24,
                  strength: float = 2.0) -> np.ndarray:
    """Return AO in [0, 1], with 1 meaning unoccluded.

    Depth Anything's normalized output is treated as a height field: larger
    values are closer to the viewer. Samples never occlude a different
    alpha-connected sprite in an atlas.
    """
    height = np.asarray(depth, np.float32)
    opacity = np.asarray(alpha)
    if height.ndim != 2 or opacity.shape != height.shape:
        raise ValueError("Depth и alpha должны иметь одинаковый двухмерный размер")
    if radius < 1 or strength < 0 or not np.isfinite(strength):
        raise ValueError("Неверные параметры AO")

    visible = opacity > 0
    result = np.ones(height.shape, np.float32)
    if not np.any(visible) or strength == 0:
        return result

    _, labels = cv2.connectedComponents(visible.astype(np.uint8), connectivity=8)
    h, w = height.shape
    distances = sorted(set(np.rint(np.geomspace(1, radius, min(radius, 10))).astype(int)))
    occlusion = np.zeros_like(height)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))

    for dx, dy in directions:
        horizon = np.zeros_like(height)
        for step in distances:
            ox, oy = dx * step, dy * step
            if abs(ox) >= w or abs(oy) >= h:
                continue
            y0, y1 = max(0, -oy), min(h, h - oy)
            x0, x1 = max(0, -ox), min(w, w - ox)
            here = np.s_[y0:y1, x0:x1]
            there = np.s_[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
            same_sprite = (labels[here] != 0) & (labels[here] == labels[there])
            slope = np.maximum(height[there] - height[here], 0)
            slope *= (8 * strength / np.hypot(ox, oy))
            np.maximum(horizon[here], np.where(same_sprite, slope, 0),
                       out=horizon[here])
        occlusion += np.arctan(horizon)

    result[visible] = 1 - occlusion[visible] / (len(directions) * (np.pi / 2))
    return np.clip(result, 0, 1)
