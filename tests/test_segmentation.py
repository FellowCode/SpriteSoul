import numpy as np
import pytest
import cv2

from smg import segmentation
from smg.segmentation import crown_mask, expand_crown_mask, predict_crown_scores, tree_crown_mask


def test_crown_mask_uses_threshold_and_excludes_transparency():
    scores = np.array([[0.8, 0.7, 0.5], [0.2, np.nan, 0.9]], np.float32)
    alpha = np.array([[255, 0, 255], [255, 255, 128]], np.uint8)

    mask = crown_mask(scores, alpha, 0.5)
    assert mask.dtype == bool
    assert mask.shape == (2, 3)
    assert np.array_equal(mask, [[True, False, True], [False, False, True]])
    assert np.array_equal(crown_mask(scores, alpha, 0.85),
                          [[False, False, False], [False, False, True]])


def test_crown_mask_rejects_wrong_shape_and_threshold():
    with pytest.raises(ValueError, match="Размер"):
        crown_mask(np.zeros((3, 2)), np.zeros((2, 3)), 0.5)
    with pytest.raises(ValueError, match="Порог"):
        crown_mask(np.zeros((2, 3)), np.zeros((2, 3)), 1.1)


def test_clipseg_requires_rgba_uint8():
    with pytest.raises(ValueError, match="RGBA uint8"):
        predict_crown_scores(np.zeros((2, 3, 3), np.uint8))


def test_crown_expands_to_alpha_edges_but_excludes_trunk():
    rgba = np.zeros((100, 100, 4), np.uint8)
    canopy = np.zeros(rgba.shape[:2], np.uint8)
    cv2.ellipse(canopy, (50, 35), (30, 24), 0, 0, 360, 255, -1)
    rgba[canopy > 0] = (35, 110, 35, 255)
    rgba[52:93, 44:57] = (115, 70, 35, 255)
    seed = np.zeros(canopy.shape, np.uint8)
    cv2.ellipse(seed, (50, 33), (18, 14), 0, 0, 360, 1, -1)
    seed[80, 50] = 1  # A stray low-confidence trunk pixel must be removed.
    original = seed.astype(bool)

    expanded = expand_crown_mask(original, rgba)

    assert np.array_equal(original, seed.astype(bool))
    assert expanded.sum() > original.sum()
    assert expanded[35, 20] and expanded[35, 80] and expanded[11, 50]
    assert not expanded[80, 50]
    assert not expanded[55, 50]
    assert not np.any(expanded & (rgba[..., 3] == 0))


def test_crown_without_trunk_fills_its_alpha_component():
    rgba = np.zeros((50, 60, 4), np.uint8)
    rgba[5:45, 8:52] = (40, 120, 50, 255)
    seed = np.zeros(rgba.shape[:2], bool)
    seed[12:38, 16:44] = True
    expanded = expand_crown_mask(seed, rgba)
    assert np.array_equal(expanded, rgba[..., 3] > 0)


def test_crown_expansion_stops_before_wide_trunk_without_a_neck():
    rgba = np.zeros((75, 70, 4), np.uint8)
    rgba[8:45, 12:58] = (40, 120, 50, 255)
    rgba[45:68, 12:58] = (120, 70, 35, 255)
    seed = np.zeros(rgba.shape[:2], bool)
    seed[16:40, 20:50] = True
    expanded = expand_crown_mask(seed, rgba)
    assert expanded[20, 12] and expanded[20, 57]
    assert not expanded[55, 35]


def test_tree_requires_more_than_40_percent_of_opaque_pixels(monkeypatch):
    monkeypatch.setattr(segmentation, "expand_crown_mask", lambda seed, rgba: seed)
    rgba = np.zeros((2, 6, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[0, 5, 3] = 0
    rgba[1, 5, 3] = 0
    scores = np.zeros((2, 6), np.float32)
    scores.flat[:4] = 0.9
    scores[0, 5] = 0.9  # Transparent pixels cannot raise the crown fraction.
    assert tree_crown_mask(scores, rgba) is None  # Exactly 4 / 10.
    scores.flat[4] = 0.9
    mask = tree_crown_mask(scores, rgba)
    assert mask is not None and mask.sum() == 5
    assert tree_crown_mask(scores, rgba, threshold=0.95) is None
    scores.flat[4] = 0
    scores.flat[3] = 0
    assert tree_crown_mask(scores, rgba) is None
    rgba[..., 3] = 0
    assert tree_crown_mask(scores, rgba) is None


def test_auto_detection_uses_expanded_mask(monkeypatch):
    rgba = np.zeros((50, 60, 4), np.uint8)
    rgba[5:45, 8:52] = (40, 120, 50, 255)
    scores = np.zeros(rgba.shape[:2], np.float32)
    scores[12:38, 16:44] = 0.9
    monkeypatch.setattr(segmentation, "predict_crown_scores", lambda image, progress: scores)
    mask = segmentation.detect_tree_crown(rgba)
    assert mask is not None
    assert np.array_equal(mask, rgba[..., 3] > 0)
