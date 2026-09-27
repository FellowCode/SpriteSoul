from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from smg.pipeline import open_png


def export_maps(source_path: str | Path, depth: np.ndarray, normal: np.ndarray,
                alpha: np.ndarray,
                directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem
    depth_path = directory / f"{stem}_depth.png"
    normal_path = directory / f"{stem}_normal.png"
    depth_16 = np.rint(np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    depth_rgba = np.empty((*depth.shape, 4), np.uint16)
    depth_rgba[..., :3] = depth_16[..., None]
    depth_rgba[..., 3] = alpha.astype(np.uint16) * 257
    encoded, png = cv2.imencode(".png", depth_rgba)
    if not encoded:
        raise OSError(f"Не удалось сохранить {depth_path}")
    depth_path.write_bytes(png.tobytes())
    Image.fromarray(normal).save(normal_path)
    return depth_path, normal_path


def save_project(path: str | Path, source_path: str | Path, depth: np.ndarray,
                 normal_strength: float, convention: str) -> None:
    path = Path(path)
    relative_source = str(Path(source_path).resolve().relative_to(path.parent.resolve())) if Path(source_path).resolve().is_relative_to(path.parent.resolve()) else str(Path(source_path).resolve())
    with path.open("wb") as file:
        np.savez_compressed(file, source=relative_source, depth=depth.astype(np.float32),
                            normal_strength=normal_strength, convention=convention)


def load_project(path: str | Path) -> tuple[Path, np.ndarray, float, str]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as project:
        source = Path(str(project["source"]))
        if not source.is_absolute():
            source = path.parent / source
        depth = np.asarray(project["depth"], np.float32)
        strength = float(project["normal_strength"])
        convention = str(project["convention"])
    rgba = open_png(source)
    if depth.shape != rgba.shape[:2]:
        raise ValueError("Размер проекта не соответствует исходному PNG")
    return source, depth, strength, convention
