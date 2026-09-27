"""Run the existing IntrinsicAnything experiment on one 256px input per sprite."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from smg.albedo import prepare_intrinsic_input, reconstruct_albedo
from smg.model_paths import INTRINSIC_ROOT


DEFAULT_INTRINSIC_ROOT = INTRINSIC_ROOT


def _experiment() -> tuple[Path, Path]:
    configured = os.environ.get("SPRITE_SOUL_INTRINSIC_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_INTRINSIC_ROOT
    python = root / ".venv-intrinsic" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not (root / "inference.py").is_file() or not (root / "weights/albedo/checkpoints/last.ckpt").is_file() or not python.is_file():
        raise RuntimeError(
            f"Не найдена установка IntrinsicAnything: {root}. "
            "Поместите её в models/IntrinsicAnything или укажите SPRITE_SOUL_INTRINSIC_ROOT."
        )
    return root, python


def _sprite_labels(alpha: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Group nearby alpha islands; keep well-separated atlas sprites independent."""
    height, width = alpha.shape
    radius = min(64, max(16, round(min(height, width) * 0.08)))
    grown = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((radius, radius), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats(grown)
    regions = list(range(1, count))
    if not regions:
        return labels, []
    areas = np.bincount(labels[alpha > 0], minlength=count)
    if areas[1:].max() >= 0.9 * areas[1:].sum():
        # Small detached details belong to the one dominant object.
        labels = np.where(alpha > 0, 1, 0).astype(np.int32)
        return labels, [1]
    return labels, regions


def generate_albedo(rgba: np.ndarray, progress=None, strength: float = 1.0,
                    illumination_sigma: float = 2.0, shadow_strength: float = 1.0,
                    debug: bool = False, debug_dir: str | Path | None = None) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Albedo ожидает RGBA uint8")
    labels, regions = _sprite_labels(rgba[..., 3])
    if not regions:
        return rgba.copy()
    root, python = _experiment()
    prepared = []
    for region in regions:
        isolated = rgba.copy()
        isolated[labels != region, 3] = 0
        prepared.append(prepare_intrinsic_input(isolated))
    output = rgba.copy()
    with tempfile.TemporaryDirectory(prefix="sprite-soul-albedo-") as temporary:
        temporary = Path(temporary)
        input_dir = temporary / "input"
        output_dir = temporary / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        for index, (image, _, _) in enumerate(prepared):
            Image.fromarray(image).save(input_dir / f"sprite_{index:04d}.png")
        if progress:
            progress("Генерация Albedo...")
        command = [str(python), str(root / "inference.py"), "--input_dir", str(input_dir),
                   "--output_dir", str(output_dir), "--model_dir", str(root / "weights/albedo"),
                   "--ddim", "100", "--batch_size", "1", "--splits_vertical", "1",
                   "--splits_horizontal", "1"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if result.returncode:
            details = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError("IntrinsicAnything завершился с ошибкой: " + " | ".join(details[-6:]))
        for index, region in enumerate(regions):
            path = output_dir / f"sprite_{index:04d}.png"
            if not path.is_file():
                raise RuntimeError(f"IntrinsicAnything не создал {path.name}")
            with Image.open(path) as image:
                predicted = np.asarray(image.convert("RGB"))
            image_256, bbox, model_rect = prepared[index]
            region_debug = None
            if debug:
                base = Path(debug_dir) if debug_dir is not None else Path.cwd() / "generated" / "albedo_debug"
                region_debug = base / f"sprite_{index:04d}"
            candidate = reconstruct_albedo(rgba, predicted, image_256, bbox, model_rect,
                                           strength, illumination_sigma, shadow_strength, debug, region_debug)
            mask = (labels == region) & (rgba[..., 3] > 0)
            output[mask] = candidate[mask]
            if progress:
                progress(f"Albedo: {index + 1}/{len(regions)}")
    if debug:
        directory = Path(debug_dir) if debug_dir is not None else Path.cwd() / "generated" / "albedo_debug"
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(directory / "albedo_full.png")
    return output
