"""Interactive SAM 2.1 image segmentation for foliage selections."""

import gc
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from smg.model_paths import MODELS_ROOT


# Keep the source and checkpoint paired.  SAM 2.1 checkpoints require recent
# SAM 2 code, so do not silently use an arbitrary globally installed version.
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_SOURCE_ROOT = MODELS_ROOT / f"sam2-{SAM2_COMMIT}"
SAM2_CHECKPOINT = SAM2_SOURCE_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_base_plus.pt"
)


def source_root(progress=None) -> Path:
    """Download the pinned SAM source tree on demand and return its root."""
    if (SAM2_SOURCE_ROOT / "sam2" / "build_sam.py").is_file():
        return SAM2_SOURCE_ROOT
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/facebookresearch/sam2/archive/{SAM2_COMMIT}.zip"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = io.BytesIO()
        size = response.headers.get("Content-Length")
        total = int(size) if size and size.isdigit() else None
        while chunk := response.read(1024 * 1024):
            data.write(chunk)
            if progress:
                progress(data.tell(), total)
    with zipfile.ZipFile(data) as archive:
        archive.extractall(MODELS_ROOT)
    if not (SAM2_SOURCE_ROOT / "sam2" / "build_sam.py").is_file():
        raise RuntimeError("Не удалось загрузить исходный код SAM 2.1")
    return SAM2_SOURCE_ROOT


def checkpoint_path(progress=None) -> Path:
    """Download SAM 2.1 Base+ weights into the pinned source tree."""
    if SAM2_CHECKPOINT.is_file():
        return SAM2_CHECKPOINT
    SAM2_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SAM2_CHECKPOINT.with_suffix(".part")
    try:
        with urllib.request.urlopen(SAM2_CHECKPOINT_URL, timeout=120) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            with temporary.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
                    if progress:
                        progress(file.tell(), total)
        temporary.replace(SAM2_CHECKPOINT)
    finally:
        if temporary.exists():
            temporary.unlink()
    return SAM2_CHECKPOINT


def is_available() -> bool:
    return (
        (SAM2_SOURCE_ROOT / "sam2" / "build_sam.py").is_file()
        and SAM2_CHECKPOINT.is_file()
    )


class SamSession:
    """A CUDA-backed predictor with one encoded RGBA sprite."""

    def __init__(self, rgba: np.ndarray):
        if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
            raise ValueError("SAM ожидает RGBA uint8")
        self.rgba = rgba.copy()
        self._model = None
        self._predictor = None

    def open(self, progress=None) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Для SAM 2.1 нужна NVIDIA CUDA")
        root = source_root()
        checkpoint = checkpoint_path()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM 2.1 установлен не полностью. Повторите подготовку AI.") from exc
        if progress:
            progress("Загрузка SAM 2.1...")
        try:
            self._model = build_sam2(SAM2_CONFIG, str(checkpoint), device="cuda")
            self._predictor = SAM2ImagePredictor(self._model)
            alpha = self.rgba[..., 3:4].astype(np.float32) / 255
            rgb = np.rint(self.rgba[..., :3] * alpha + 127 * (1 - alpha)).astype(np.uint8)
            if progress:
                progress("SAM 2.1 анализирует изображение...")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self._predictor.set_image(np.ascontiguousarray(rgb))
        except torch.OutOfMemoryError as exc:
            self.close()
            raise RuntimeError("Недостаточно VRAM для SAM 2.1. Закройте другие GPU-приложения.") from exc

    def predict(self, points: list[tuple[int, int]], labels: list[int]) -> np.ndarray:
        if len(points) != len(labels) or not points or not any(labels):
            raise ValueError("Для выделения нужен хотя бы один добавляющий клик")
        if self._predictor is None:
            raise RuntimeError("SAM 2.1 ещё не готова")
        import torch
        coordinates = np.asarray(points, np.float32)
        prompt_labels = np.asarray(labels, np.int32)
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, _scores, _logits = self._predictor.predict(
                    point_coords=coordinates, point_labels=prompt_labels,
                    multimask_output=False,
                )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError("Недостаточно VRAM для уточнения маски SAM 2.1.") from exc
        mask = np.asarray(masks[0], dtype=bool)
        if mask.shape != self.rgba.shape[:2]:
            raise RuntimeError("SAM 2.1 вернула маску неверного размера")
        return mask & (self.rgba[..., 3] > 0)

    def close(self) -> None:
        self._predictor = None
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
