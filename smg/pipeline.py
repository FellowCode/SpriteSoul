from pathlib import Path

import numpy as np
from PIL import Image

from smg.depth.processing import normalize_depth
from smg.normal import normals_from_depth


def open_png(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Нужен PNG-файл")
        return np.array(image.convert("RGBA"))


def generate_depth(rgba: np.ndarray, model, progress=None) -> np.ndarray:
    raw = model.generate(rgba, progress)
    if raw.shape != rgba.shape[:2]:
        raise ValueError("Модель вернула неверный размер Depth")
    return normalize_depth(raw, rgba[..., 3])


def generate_normal(depth: np.ndarray, rgba: np.ndarray, strength: float = 3.0,
                    convention: str = "OpenGL") -> np.ndarray:
    return normals_from_depth(depth, rgba[..., 3], strength, convention)
