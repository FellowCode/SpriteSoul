from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from smg.pipeline import open_png


def export_maps(source_path: str | Path, depth: np.ndarray, normal: np.ndarray,
                alpha: np.ndarray,
                directory: str | Path) -> tuple[Path, Path]:
    depth_path = export_depth(source_path, depth, alpha, directory)
    normal_path = export_normal(source_path, normal, directory)
    return depth_path, normal_path


def export_depth(source_path: str | Path, depth: np.ndarray, alpha: np.ndarray,
                 directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem
    depth_path = directory / f"{stem}_depth.png"
    depth_16 = np.rint(np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    depth_rgba = np.empty((*depth.shape, 4), np.uint16)
    depth_rgba[..., :3] = depth_16[..., None]
    depth_rgba[..., 3] = alpha.astype(np.uint16) * 257
    encoded, png = cv2.imencode(".png", depth_rgba)
    if not encoded:
        raise OSError(f"Не удалось сохранить {depth_path}")
    depth_path.write_bytes(png.tobytes())
    return depth_path


def export_normal(source_path: str | Path, normal: np.ndarray,
                  directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normal_path = directory / f"{Path(source_path).stem}_normal.png"
    Image.fromarray(normal).save(normal_path)
    return normal_path


def export_albedo(source_path: str | Path, albedo: np.ndarray,
                  directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    albedo_path = directory / f"{Path(source_path).stem}_albedo.png"
    Image.fromarray(albedo).save(albedo_path)
    return albedo_path


def _validated_foliage_mask(foliage_mask: np.ndarray | None,
                            shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if foliage_mask is None:
        return None
    mask = np.asarray(foliage_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Маска листвы должна быть двухмерной")
    if shape is not None and mask.shape != shape:
        raise ValueError("Размер маски листвы не соответствует исходному PNG")
    return mask


def save_project(path: str | Path, source_path: str | Path, depth: np.ndarray | None,
                 normal_strength: float, convention: str,
                 foliage_mask: np.ndarray | None = None) -> None:
    path = Path(path)
    relative_source = str(Path(source_path).resolve().relative_to(path.parent.resolve())) if Path(source_path).resolve().is_relative_to(path.parent.resolve()) else str(Path(source_path).resolve())
    if depth is not None:
        depth = np.asarray(depth, np.float32)
    mask_shape = depth.shape if depth is not None else (
        open_png(source_path).shape[:2] if foliage_mask is not None else None
    )
    mask = _validated_foliage_mask(foliage_mask, mask_shape)
    fields = {
        "source": relative_source,
        "normal_strength": normal_strength,
        "convention": convention,
    }
    if depth is not None:
        fields["depth"] = depth
    if mask is not None:
        fields["foliage_mask"] = mask.astype(np.uint8)
    with path.open("wb") as file:
        np.savez_compressed(file, **fields)


def load_project(path: str | Path, *, with_foliage_mask: bool = False):
    """Load a project, optionally including its persisted foliage selection.

    The default four-item tuple is retained for existing callers and projects.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as project:
        source = Path(str(project["source"]))
        if not source.is_absolute():
            source = path.parent / source
        depth = np.asarray(project["depth"], np.float32) if "depth" in project else None
        strength = float(project["normal_strength"])
        convention = str(project["convention"])
        foliage_mask = (
            np.asarray(project["foliage_mask"], dtype=bool)
            if "foliage_mask" in project else None
        )
    rgba = open_png(source)
    if depth is not None and depth.shape != rgba.shape[:2]:
        raise ValueError("Размер проекта не соответствует исходному PNG")
    foliage_mask = _validated_foliage_mask(foliage_mask, rgba.shape[:2])
    if with_foliage_mask:
        return source, depth, strength, convention, foliage_mask
    return source, depth, strength, convention
