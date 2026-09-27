"""Use low-resolution intrinsic albedo to remove lighting from original pixels."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SIZE = 256
EPS = 1e-4
LUMA = np.array((0.2126, 0.7152, 0.0722), np.float32)


def _linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.rint(srgb * 255).astype(np.uint8)


def prepare_intrinsic_input(rgba: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return a white-matted 256px RGBA input and its source/model rectangles.

    Rectangles use (left, top, right, bottom) with exclusive right/bottom.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Albedo ожидает RGBA uint8")
    ys, xs = np.nonzero(rgba[..., 3])
    if len(xs) == 0:
        raise ValueError("У спрайта нет непрозрачных пикселей")
    left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    width, height = right - left, bottom - top
    margin = 8
    scale = (SIZE - 2 * margin) / max(width, height)
    scaled_width = max(1, round(width * scale))
    scaled_height = max(1, round(height * scale))
    pad_x = (SIZE - scaled_width) // 2
    pad_y = (SIZE - scaled_height) // 2
    crop = rgba[top:bottom, left:right]
    alpha = crop[..., 3:4].astype(np.float32) / 255
    # The model sees a white background; the original alpha is kept separately.
    matted = np.rint(crop[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_rgb = cv2.resize(matted, (scaled_width, scaled_height), interpolation=interpolation)
    resized_alpha = cv2.resize(crop[..., 3], (scaled_width, scaled_height), interpolation=interpolation)
    prepared = np.full((SIZE, SIZE, 4), 255, np.uint8)
    prepared[..., 3] = 0
    prepared[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width, :3] = resized_rgb
    prepared[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width, 3] = resized_alpha
    return prepared, (left, top, right, bottom), (pad_x, pad_y, pad_x + scaled_width, pad_y + scaled_height)


def reconstruct_albedo(original_rgba: np.ndarray, intrinsic_albedo: np.ndarray,
                       prepared: np.ndarray, bbox: tuple[int, int, int, int],
                       model_rect: tuple[int, int, int, int], strength: float = 1.0,
                       illumination_sigma: float = 2.0, shadow_strength: float = 1.0,
                       debug: bool = False, debug_dir: str | Path | None = None) -> np.ndarray:
    """Apply only the model's smooth log-luminance change to full-size RGB."""
    if intrinsic_albedo.shape[:2] != (SIZE, SIZE) or intrinsic_albedo.shape[2] < 3:
        raise ValueError("IntrinsicAnything должен вернуть RGB 256×256")
    if original_rgba.ndim != 3 or original_rgba.shape[2] != 4 or original_rgba.dtype != np.uint8:
        raise ValueError("Albedo ожидает RGBA uint8")
    x0, y0, x1, y1 = model_rect
    left, top, right, bottom = bbox
    original_linear = _linear(prepared[..., :3])
    albedo_linear = _linear(intrinsic_albedo[..., :3])
    original_y = original_linear @ LUMA
    albedo_y = albedo_linear @ LUMA
    # Positive means the model darkened the area; subtracting it darkens original.
    raw = np.log(original_y + EPS) - np.log(albedo_y + EPS)
    mask = prepared[..., 3].astype(np.float32) / 255
    sigma = max(0.0, float(illumination_sigma))
    if sigma:
        weight = cv2.GaussianBlur(mask, (0, 0), sigma)
        illumination = cv2.GaussianBlur(raw * mask, (0, 0), sigma) / np.maximum(weight, EPS)
    else:
        illumination = raw
    illumination = np.clip(illumination, -np.log(4), np.log(4))
    high = np.abs(raw - illumination)
    valid = mask > 0.5
    scale = float(np.percentile(high[valid], 90)) if np.any(valid) else 0.0
    confidence = np.ones_like(raw) if scale < EPS else np.clip(1 - high / (scale * 2), 0, 1)
    if sigma:
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigma)
    shadow_mask = np.clip(-illumination / np.log(2), 0, 1)
    if sigma:
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigma)

    size = (right - left, bottom - top)
    def upscale(field: np.ndarray) -> np.ndarray:
        return cv2.resize(field[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)

    correction = upscale(illumination) * np.clip(upscale(confidence), 0, 1) * strength
    shadow_full = np.clip(upscale(shadow_mask), 0, 1)
    correction *= 1 + (shadow_strength - 1) * shadow_full
    output = original_rgba.copy()
    crop = output[top:bottom, left:right]
    linear = _linear(crop[..., :3])
    corrected = np.exp(np.log(linear + EPS) - correction[..., None]) - EPS
    crop[..., :3] = _srgb(corrected)
    output[original_rgba[..., 3] == 0, :3] = 0
    if debug:
        directory = Path(debug_dir) if debug_dir is not None else Path.cwd() / "generated" / "albedo_debug"
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prepared[..., :3]).save(directory / "original_256.png")
        Image.fromarray(intrinsic_albedo[..., :3]).save(directory / "intrinsic_albedo_256.png")
        def save_field(name: str, field: np.ndarray, low: float, high_value: float) -> None:
            image = np.rint(np.clip((field - low) / (high_value - low), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(image).save(directory / name)
        save_field("illumination.png", illumination, -np.log(4), np.log(4))
        save_field("shadow_mask.png", shadow_mask, 0, 1)
        save_field("confidence.png", confidence, 0, 1)
        Image.fromarray(output).save(directory / "albedo_full.png")
    return output
