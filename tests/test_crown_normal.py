import numpy as np
import pytest

from smg.crown_normal import compose_crown_normals


def _front(shape):
    vectors = np.zeros((*shape, 3), np.float32)
    vectors[..., 2] = 1
    return vectors


def test_cylindrical_crown_faces_left_center_and_right():
    shape = (50, 60)
    mask = np.zeros(shape, bool)
    mask[8:42, 10:50] = True
    alpha = np.full(shape, 255, np.uint8)
    original = _front(shape)

    result = compose_crown_normals(original, mask, alpha)

    assert np.array_equal(result[~mask], original[~mask])
    assert result[25, 15, 0] < -0.5
    assert result[25, 10, 0] > result[25, 15, 0]
    assert np.linalg.norm(result[25, 10] - original[25, 10]) < 0.1
    assert result[25, 44, 0] > 0.5
    assert abs(result[25, 30, 0]) < 0.1
    assert abs(result[25, 30, 1]) < 0.1
    assert result[25, 30, 2] > 0.95
    assert np.allclose(np.linalg.norm(result[mask], axis=-1), 1, atol=1e-5)


def test_tapered_crown_has_upward_slope_and_keeps_ai_detail():
    shape = (60, 70)
    mask = np.zeros(shape, bool)
    for y in range(8, 52):
        radius = 8 + (y - 8) // 3
        mask[y, 35 - radius:36 + radius] = True
    alpha = np.full(shape, 255, np.uint8)
    original = _front(shape)
    original[mask] = (0.3, -0.2, 0.93)  # Broad AI tilt should be replaced.
    baseline = compose_crown_normals(original, mask, alpha)
    assert baseline[30, 35, 1] > 0.15
    assert abs(baseline[30, 35, 0]) < 0.1

    detailed = original.copy()
    detailed[30:33, 32:35] = (-0.25, 0.35, 0.90)
    result = compose_crown_normals(detailed, mask, alpha)
    assert np.linalg.norm(result[31, 33] - baseline[31, 33]) > 0.15
    assert np.allclose(result[15, 35], baseline[15, 35], atol=0.03)


def test_separate_crowns_and_transparent_pixels():
    mask = np.zeros((45, 95), bool)
    mask[6:36, 5:37] = True
    mask[6:36, 55:87] = True
    alpha = np.full(mask.shape, 255, np.uint8)
    alpha[20, 10] = 0
    original = _front(mask.shape)
    result = compose_crown_normals(original, mask, alpha)
    assert result[20, 10, 0] == 0
    assert result[20, 11, 0] < 0
    assert result[20, 61, 0] < 0
    assert result[20, 81, 0] > 0


def test_empty_or_mismatched_crown_mask():
    original = _front((8, 9))
    mask = np.zeros((8, 9), bool)
    alpha = np.full(mask.shape, 255, np.uint8)
    assert np.array_equal(compose_crown_normals(original, mask, alpha), original)
    with pytest.raises(ValueError, match="dimensions"):
        compose_crown_normals(original, mask[:, :-1], alpha)
