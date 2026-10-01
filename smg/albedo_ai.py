"""Run the existing IntrinsicAnything experiment on one 256px input per sprite."""

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from smg.albedo import prepare_intrinsic_input, reconstruct_albedo
from smg.model_paths import INTRINSIC_ROOT


DEFAULT_INTRINSIC_ROOT = INTRINSIC_ROOT


def _run_intrinsic(command: list[str], root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    logs = Path.cwd() / "generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"intrinsic-{uuid.uuid4().hex}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    if result.returncode:
        details = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        code = result.returncode & 0xFFFFFFFF
        explanation = ""
        if code == 0xC0000005:
            explanation = " Сбой доступа к памяти Windows (access violation)."
        elif code in (0xC0000017, 0xC000012D) or any(
            marker in line.lower() for line in details
            for marker in ("not enough memory", "paging file is too small", "defaultcpuallocator")
        ):
            explanation = " Недостаточно системной памяти для загрузки модели."
        elif any("cuda out of memory" in line.lower() for line in details):
            explanation = " Недостаточно видеопамяти."
        raise RuntimeError(
            f"IntrinsicAnything завершился с ошибкой (код 0x{code:08X}).{explanation}\n"
            f"Полный журнал: {log_path}\n" + "\n".join(details[-10:])
        )


def _enable_low_vram(root: Path) -> None:
    """Keep the diffusion model in fp16 and autocast its inference operations."""
    replacements = (
        (root / "models" / "matfusion.py",
         "    model.eval().to(device)",
         "    model.eval().half().to(device)"),
        (root / "inference.py",
         "    model.generation(dps_scale=args.guidance, uc_score=1, \n"
         "                     ddim_steps=args.ddim, batch_size=args.batch_size, n_samples=1)",
         "    with torch.autocast(\"cuda\", dtype=torch.float16):\n"
         "        model.generation(dps_scale=args.guidance, uc_score=1, \n"
         "                         ddim_steps=args.ddim, batch_size=args.batch_size, n_samples=1)"),
    )
    updates = []
    for path, original, patched in replacements:
        source = path.read_text(encoding="utf-8")
        if patched in source:
            continue
        if source.count(original) != 1:
            raise RuntimeError(f"Не удалось включить экономию VRAM для IntrinsicAnything: {path}")
        updates.append((path, source.replace(original, patched)))
    for path, source in updates:
        path.write_text(source, encoding="utf-8")


def _experiment() -> tuple[Path, Path]:
    configured = os.environ.get("SPRITE_SOUL_INTRINSIC_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_INTRINSIC_ROOT
    python = root / ".venv-intrinsic" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not (root / "inference.py").is_file() or not (root / "weights/albedo/checkpoints/last.ckpt").is_file() or not python.is_file():
        raise RuntimeError(
            f"Не найдена установка IntrinsicAnything: {root}. "
            "Поместите её в models/IntrinsicAnything или укажите SPRITE_SOUL_INTRINSIC_ROOT."
        )
    _enable_low_vram(root)
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
                    debug: bool = False, debug_dir: str | Path | None = None,
                    session=None) -> np.ndarray:
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
        worker = Path(__file__).with_name("intrinsic_worker.py")
        command = [str(python), "-u", "-X", "faulthandler", str(worker), str(root),
                   "--input_dir", str(input_dir),
                   "--output_dir", str(output_dir), "--model_dir", str(root / "weights/albedo"),
                   "--ddim", "100", "--batch_size", "1", "--splits_vertical", "1",
                   "--splits_horizontal", "1"]
        if session is None:
            _run_intrinsic(command, root)
        else:
            session.run(command[6:], progress)
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
