"""One replaceable AI boundary: Depth Anything V2 Small."""

import cv2
import numpy as np

from smg.atlas import feather_weights, tile_positions


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthModel:
    def __init__(self) -> None:
        self.model = None
        self.processor = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if not torch.cuda.is_available():
            raise RuntimeError("Для генерации Depth нужна NVIDIA CUDA. Проверьте установку CUDA-сборки PyTorch.")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to("cuda").eval()

    def _infer(self, rgba: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        # Transparent RGB often contains arbitrary colors; use a neutral matte for AI only.
        a = rgba[..., 3:4].astype(np.float32) / 255
        rgb = np.rint(rgba[..., :3] * a + 127 * (1 - a)).astype(np.uint8)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = self.model(**inputs).predicted_depth
        depth = prediction[0].float().cpu().numpy()
        del inputs, prediction
        return cv2.resize(depth, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_CUBIC)

    def generate(self, rgba: np.ndarray, progress=None) -> np.ndarray:
        import torch

        try:
            self._load()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError("Недостаточно VRAM для загрузки модели. Закройте другие GPU-приложения.") from exc
        h, w = rgba.shape[:2]
        if progress:
            progress("Анализ изображения...")
        try:
            if max(h, w) <= 1024:
                return self._infer(rgba)
            return self._tiled(rgba, progress)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if progress:
                progress("Недостаточно VRAM; обработка тайлами...")
            try:
                return self._tiled(rgba, progress, size=768)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError("Недостаточно VRAM даже для тайлов 768 px. Закройте другие GPU-приложения.") from exc

    def _tiled(self, rgba: np.ndarray, progress=None, size: int = 1024) -> np.ndarray:
        h, w = rgba.shape[:2]
        overlap = size // 4
        # A whole-image prediction anchors independent tiles to one depth scale.
        guide_scale = min(1.0, 768 / max(h, w))
        guide_input = cv2.resize(rgba, (max(1, round(w * guide_scale)),
                                         max(1, round(h * guide_scale))), interpolation=cv2.INTER_AREA)
        guide = cv2.resize(self._infer(guide_input), (w, h), interpolation=cv2.INTER_CUBIC)
        ys = tile_positions(h, size, overlap)
        xs = tile_positions(w, size, overlap)
        output = np.zeros((h, w), np.float32)
        weights = np.zeros((h, w), np.float32)
        total = len(xs) * len(ys)
        index = 0
        for y in ys:
            for x in xs:
                index += 1
                if progress:
                    progress(f"Тайл {index}/{total}")
                tile = rgba[y:y + size, x:x + size]
                if not np.any(tile[..., 3]):
                    continue
                pred = self._infer(tile)
                reference = guide[y:y + tile.shape[0], x:x + tile.shape[1]]
                valid = tile[..., 3] > 0
                sample = pred[valid]
                target = reference[valid]
                variance = float(np.var(sample))
                scale = float(np.mean((sample - sample.mean()) * (target - target.mean())) / variance) if variance > 1e-8 else 1.0
                scale = max(0.1, min(scale, 10.0))
                pred = (pred - sample.mean()) * scale + target.mean()
                weight = feather_weights(tile.shape[0], tile.shape[1], overlap,
                                         y > 0, y + size < h, x > 0, x + size < w)
                output[y:y + tile.shape[0], x:x + tile.shape[1]] += pred * weight
                weights[y:y + tile.shape[0], x:x + tile.shape[1]] += weight
        return np.where(weights > 0, output / np.maximum(weights, 1e-6), guide)
