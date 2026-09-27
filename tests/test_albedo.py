from pathlib import Path

import numpy as np

from smg.albedo import prepare_intrinsic_input, reconstruct_albedo
from smg.albedo_ai import DEFAULT_INTRINSIC_ROOT, _sprite_labels
from smg.model_paths import (
    HUGGINGFACE_HUB_CACHE,
    HUGGINGFACE_XET_CACHE,
    MODELS_ROOT,
    PROJECT_ROOT,
)


def test_guided_albedo_preserves_detail_and_alpha(tmp_path):
    rgba = np.zeros((96, 160, 4), np.uint8)
    yy, xx = np.indices((64, 128))
    # A shadow over alternating material detail; the model removes the shadow.
    detail = np.where(xx % 2 == 0, 80, 110).astype(np.uint8)
    detail[:, :64] = np.rint(detail[:, :64] * 0.6).astype(np.uint8)
    rgba[16:80, 16:144, :3] = detail[..., None]
    rgba[16:80, 16:144, 3] = 255
    rgba[17, 17, 3] = 83
    prepared, bbox, model_rect = prepare_intrinsic_input(rgba)
    intrinsic = prepared[..., :3].copy()
    mid = (model_rect[0] + model_rect[2]) // 2
    intrinsic[:, :mid] = np.rint(np.clip(intrinsic[:, :mid].astype(np.float32) / 0.6, 0, 255)).astype(np.uint8)
    result = reconstruct_albedo(rgba, intrinsic, prepared, bbox, model_rect,
                                debug=True, debug_dir=tmp_path)
    assert result.shape == rgba.shape
    assert np.array_equal(result[..., 3], rgba[..., 3])
    assert result[48, 48, 0] > rgba[48, 48, 0]
    assert abs(int(result[48, 120, 0]) - int(rgba[48, 120, 0])) < 5
    assert result[48, 49, 0] > result[48, 48, 0]  # Original alternating texture remains.
    assert set(path.name for path in tmp_path.iterdir()) == {
        "original_256.png", "intrinsic_albedo_256.png", "illumination.png",
        "shadow_mask.png", "confidence.png", "albedo_full.png",
    }


def test_prepare_crops_alpha_and_keeps_aspect_ratio():
    rgba = np.zeros((100, 200, 4), np.uint8)
    rgba[20:70, 60:160] = (100, 120, 140, 255)
    prepared, bbox, rect = prepare_intrinsic_input(rgba)
    assert bbox == (60, 20, 160, 70)
    assert prepared.shape == (256, 256, 4)
    assert abs((rect[2] - rect[0]) / (rect[3] - rect[1]) - 2) < 0.02


def test_atlas_separates_distant_sprites():
    alpha = np.zeros((128, 320), np.uint8)
    alpha[20:100, 20:100] = 255
    alpha[20:100, 220:300] = 255
    labels, regions = _sprite_labels(alpha)
    assert len(regions) == 2
    assert labels[50, 50] != labels[50, 250]


def test_intrinsic_default_is_inside_project():
    assert DEFAULT_INTRINSIC_ROOT == (
        Path(__file__).resolve().parents[1] / "models" / "IntrinsicAnything"
    )


def test_all_model_storage_is_inside_project():
    assert MODELS_ROOT == PROJECT_ROOT / "models"
    assert HUGGINGFACE_HUB_CACHE.is_relative_to(MODELS_ROOT)
    assert HUGGINGFACE_XET_CACHE.is_relative_to(MODELS_ROOT)
