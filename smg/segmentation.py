"""Text-guided crown segmentation with CLIPSeg."""

import gc

import cv2
import numpy as np
from PIL import Image

from smg.model_paths import HUGGINGFACE_HUB_CACHE


MODEL_ID = "CIDAS/clipseg-rd64-refined"
CROWN_PROMPT = "tree canopy"
DEFAULT_CROWN_THRESHOLD = 0.5
TREE_CROWN_FRACTION = 0.4
MODEL_FILES = (
    "config.json", "preprocessor_config.json", "model.safetensors",
    "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json",
)


def crown_mask(scores: np.ndarray, alpha: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold model probabilities in source coordinates and respect transparency."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape != alpha.shape:
        raise ValueError("Размер карты CLIPSeg не соответствует исходному PNG")
    if not 0 <= threshold <= 1:
        raise ValueError("Порог CLIPSeg должен быть от 0 до 1")
    return np.isfinite(scores) & (scores >= threshold) & (alpha > 0)


def _neck_row(alpha: np.ndarray, seed: np.ndarray) -> int:
    """Find the narrowing below a canopy, where its trunk begins."""
    rows = np.count_nonzero(alpha, axis=1).astype(np.float32)
    height = len(rows)
    smooth = cv2.GaussianBlur(
        rows[np.newaxis, :], (0, 0), sigmaX=max(1.5, height * 0.01),
        borderType=cv2.BORDER_REFLECT_101,
    )[0]
    seed_rows = np.flatnonzero(np.any(seed, axis=1))
    alpha_rows = np.flatnonzero(np.any(alpha, axis=1))
    seed_top, seed_bottom = int(seed_rows[0]), int(seed_rows[-1])
    alpha_bottom = int(alpha_rows[-1])
    peak = seed_top + int(np.argmax(smooth[seed_top:seed_bottom + 1]))
    limit = smooth[peak] * 0.45
    span = max(3, round(height * 0.015))
    for y in range(peak + 1, alpha_bottom - span + 2):
        if np.all(smooth[y:y + span] < limit):
            return y
    if alpha_bottom - seed_bottom <= max(5, height * 0.20):
        return alpha_bottom + 1
    return min(alpha_bottom + 1, seed_bottom + max(5, round(height * 0.05)))


def expand_crown_mask(seed_mask: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Grow CLIPSeg's crown to the alpha silhouette without taking the trunk."""
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Расширение кроны ожидает RGBA uint8")
    seed = np.asarray(seed_mask, dtype=bool).copy()
    if seed.shape != rgba.shape[:2]:
        raise ValueError("Размер маски кроны не соответствует исходному PNG")
    opaque = rgba[..., 3] > 0
    seed &= opaque
    result = seed.copy()
    if not np.any(seed):
        return result

    image = np.rint(
        rgba[..., :3].astype(np.float32) * (rgba[..., 3:4] / 255)
        + 127 * (1 - rgba[..., 3:4] / 255)
    ).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        opaque.astype(np.uint8), connectivity=8,
    )
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if area < 32:
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1 = min(opaque.shape[1], x + width + 1)
        y1 = min(opaque.shape[0], y + height + 1)
        region = np.s_[y0:y1, x0:x1]
        alpha_part = components[region] == component_id
        seed_part = seed[region] & alpha_part
        if np.count_nonzero(seed_part) < 16:
            continue

        row = np.arange(alpha_part.shape[0])[:, None]
        cutoff = _neck_row(alpha_part, seed_part)
        alpha_bottom = int(np.flatnonzero(np.any(alpha_part, axis=1))[-1])
        if cutoff > alpha_bottom:
            seed_rows = np.flatnonzero(np.any(seed_part, axis=1))
            tail = alpha_part & (row > seed_rows[-1])
            if np.any(tail):
                upper = seed_part & (row < seed_rows[0] + 0.7 * len(seed_rows))
                lab = cv2.cvtColor(image[region], cv2.COLOR_RGB2LAB)
                separation = np.linalg.norm(
                    np.median(lab[upper], axis=0) - np.median(lab[tail], axis=0)
                )
                if separation > 25:
                    cutoff = min(alpha_bottom, int(seed_rows[-1]) + 3)
            if cutoff > alpha_bottom:
                result[region] |= alpha_part
                continue

        result[region] &= ~(alpha_part & (row >= cutoff))
        labels = np.full(alpha_part.shape, cv2.GC_BGD, np.uint8)
        labels[alpha_part & (row < cutoff)] = cv2.GC_PR_FGD
        labels[alpha_part & (row >= cutoff)] = cv2.GC_PR_BGD
        trunk_margin = max(3, round(height * 0.025))
        labels[alpha_part & (row >= cutoff + trunk_margin)] = cv2.GC_BGD
        core = cv2.erode(
            seed_part.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
        labels[core & (row < cutoff - 2)] = cv2.GC_FGD
        if not np.any(labels == cv2.GC_FGD):
            continue
        try:
            cv2.grabCut(
                image[region], labels, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                3, cv2.GC_INIT_WITH_MASK,
            )
            grown = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
        except cv2.error:
            grown = cv2.dilate(
                seed_part.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            ) > 0
        grown = (grown | seed_part) & alpha_part & (row < cutoff)

        # A small contour band includes antialiased leaf tips without extending
        # through the interior connection between the crown and trunk.
        near_crown = cv2.dilate(
            grown.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        ) > 0
        near_alpha_edge = cv2.distanceTransform(
            np.pad(alpha_part.astype(np.uint8), 1), cv2.DIST_L2, 3,
        )[1:-1, 1:-1] <= 5
        grown |= near_crown & near_alpha_edge & alpha_part & (row < cutoff)
        result[region] |= grown
    return result & opaque


def tree_crown_mask(scores: np.ndarray, rgba: np.ndarray,
                    threshold: float = DEFAULT_CROWN_THRESHOLD) -> np.ndarray | None:
    """Return the expanded crown only when it covers over 40% of opaque pixels."""
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Определение дерева ожидает RGBA uint8")
    opaque_count = np.count_nonzero(rgba[..., 3])
    if opaque_count == 0:
        return None
    seed = crown_mask(scores, rgba[..., 3], threshold)
    mask = expand_crown_mask(seed, rgba)
    return mask if np.count_nonzero(mask) > TREE_CROWN_FRACTION * opaque_count else None


def detect_tree_crown(rgba: np.ndarray, progress=None,
                      threshold: float = DEFAULT_CROWN_THRESHOLD) -> np.ndarray | None:
    """Use CLIPSeg to classify a sprite and return its crown if it is a tree."""
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Определение дерева ожидает RGBA uint8")
    if not np.any(rgba[..., 3]):
        return None
    mask = tree_crown_mask(predict_crown_scores(rgba, progress), rgba, threshold)
    if progress:
        progress("Дерево: крона найдена" if mask is not None else "Крона меньше 40%: обычная Normal")
    return mask


def predict_crown_scores(rgba: np.ndarray, progress=None) -> np.ndarray:
    """Return CLIPSeg probabilities at the original sprite resolution."""
    model = CrownModel()
    try:
        return model.predict(rgba, progress)
    finally:
        model.unload()


class CrownModel:
    """Reusable CLIPSeg session for a CLI batch; unloaded explicitly between stages."""

    def __init__(self):
        self.model = None
        self.processor = None

    def unload(self):
        if self.model is None and self.processor is None:
            return
        import torch

        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, rgba: np.ndarray, progress=None) -> np.ndarray:
        if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
            raise ValueError("CLIPSeg ожидает RGBA uint8")
        import torch
        import torch.nn.functional as F
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Для CLIPSeg нужна NVIDIA CUDA")
        try:
            if self.model is None:
                if progress:
                    progress("Загрузка CLIPSeg...")
                self.processor = CLIPSegProcessor.from_pretrained(
                    MODEL_ID, cache_dir=HUGGINGFACE_HUB_CACHE, local_files_only=True,
                    use_fast=False,
                )
                self.model = CLIPSegForImageSegmentation.from_pretrained(
                    MODEL_ID, cache_dir=HUGGINGFACE_HUB_CACHE, local_files_only=True,
                    use_safetensors=True,
                ).to("cuda").eval()

            alpha = rgba[..., 3:4].astype(np.float32) / 255
            rgb = np.rint(rgba[..., :3] * alpha + 127 * (1 - alpha)).astype(np.uint8)
            inputs = self.processor(
                text=[CROWN_PROMPT], images=[Image.fromarray(rgb)], return_tensors="pt",
            ).to("cuda")
            if progress:
                progress("CLIPSeg определяет крону...")
            with torch.inference_mode():
                logits = self.model(**inputs).logits.unsqueeze(1)
                logits = F.interpolate(
                    logits.float(), size=rgba.shape[:2], mode="bilinear", align_corners=False,
                )
                scores = logits.sigmoid()[0, 0].cpu().numpy().astype(np.float32)
            if scores.shape != rgba.shape[:2]:
                raise RuntimeError("CLIPSeg вернула карту неверного размера")
            return scores
        except torch.OutOfMemoryError as exc:
            raise RuntimeError("Недостаточно VRAM для CLIPSeg. Закройте другие GPU-приложения.") from exc
