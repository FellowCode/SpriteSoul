from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from smg.depth.processing import normalize_depth
from smg.normal import normals_from_depth


def open_png(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Нужен PNG-файл")
        return np.array(image.convert("RGBA"))


def open_depth_png(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Read an 8/16-bit grayscale PNG (including exported grayscale RGBA)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Карта глубины должна быть PNG-файлом")
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError(f"Не удалось прочитать карту глубины: {path}")
    if pixels.shape[:2] != shape:
        height, width = pixels.shape[:2]
        expected_height, expected_width = shape
        raise ValueError(
            f"Размер карты глубины {path}: {width}x{height}; "
            f"размер спрайта: {expected_width}x{expected_height}"
        )
    if pixels.dtype not in (np.uint8, np.uint16):
        raise ValueError("Карта глубины должна быть 8- или 16-битной")
    if pixels.ndim == 3:
        if pixels.shape[2] not in (3, 4) or not np.array_equal(pixels[..., 0], pixels[..., 1]) or not np.array_equal(pixels[..., 0], pixels[..., 2]):
            raise ValueError("Карта глубины должна быть серой, без цветных каналов")
        pixels = pixels[..., 0]
    return pixels.astype(np.float32) / np.iinfo(pixels.dtype).max


def generate_depth(rgba: np.ndarray, model, progress=None) -> np.ndarray:
    raw = model.generate(rgba, progress)
    if raw.shape != rgba.shape[:2]:
        raise ValueError("Модель вернула неверный размер Depth")
    return normalize_depth(raw, rgba[..., 3])


def generate_normal(depth: np.ndarray, rgba: np.ndarray, strength: float = 3.0,
                    convention: str = "OpenGL") -> np.ndarray:
    return normals_from_depth(depth, rgba[..., 3], strength, convention)
