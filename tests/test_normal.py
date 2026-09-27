import numpy as np

from smg.normal import (
    ai_normal, dsine_to_opengl, normals_from_depth, orient_ai_vectors,
    postprocess_ai_vectors, smooth_ai_vectors,
)


def test_flat_and_slopes():
    alpha = np.full((16, 16), 255, np.uint8)
    flat = normals_from_depth(np.full((16, 16), 0.5, np.float32), alpha)
    assert np.max(np.abs(flat[..., :3].astype(int) - (128, 128, 255))) <= 1
    x = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (16, 1))
    y = x.T.copy()
    assert normals_from_depth(x, alpha)[8, 8, 0] < 128
    gl = normals_from_depth(y, alpha, convention="OpenGL")
    dx = normals_from_depth(y, alpha, convention="DirectX")
    assert gl[8, 8, 1] > 128
    assert dx[8, 8, 1] < 128
    assert abs(int(gl[8, 8, 1]) + int(dx[8, 8, 1]) - 255) <= 1


def test_alpha_and_silhouette():
    depth = np.zeros((20, 20), np.float32)
    alpha = np.zeros((20, 20), np.uint8)
    alpha[4:16, 4:16] = 179
    alpha[8:11, 8:11] = 0
    depth[alpha > 0] = 0.75
    result = normals_from_depth(depth, alpha, strength=10)
    assert result.shape == (20, 20, 4)
    assert np.array_equal(result[..., 3], alpha)
    assert np.max(np.abs(result[alpha > 0, :3].astype(int) - (128, 128, 255))) <= 1


def test_ai_convention_and_alignment():
    alpha = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7
    raw_dsine = np.zeros((5, 7, 3), np.float32)
    raw_dsine[..., 1:] = (0.6, 0.8)  # DSINE +Y points down.
    vectors = dsine_to_opengl(raw_dsine)
    assert np.allclose(vectors[2, 3], (0, -0.6, 0.8))
    ai_gl = ai_normal(vectors, alpha, "OpenGL")
    ai_dx = ai_normal(vectors, alpha, "DirectX")
    assert ai_gl.shape == (5, 7, 4)
    assert np.array_equal(ai_gl[..., 3], alpha)
    assert ai_gl[2, 3, 1] < 128 < ai_dx[2, 3, 1]


def test_ai_smoothing_reduces_local_artifacts_without_alpha_bleed():
    alpha = np.zeros((17, 17), np.uint8)
    alpha[3:14, 3:14] = 255
    vectors = np.zeros((17, 17, 3), np.float32)
    vectors[..., 2] = 1
    vectors[8, 8] = (1, 0, 0)  # A one-pixel inference artifact.
    smoothed = smooth_ai_vectors(vectors, alpha, sigma=1.5)
    assert 0 < smoothed[8, 8, 0] < 1
    assert smoothed[8, 8, 2] > 0.9
    assert np.allclose(np.linalg.norm(smoothed, axis=-1), 1, atol=1e-5)
    assert np.allclose(smoothed[alpha == 0], (0, 0, 1))


def test_ai_postprocess_restores_normalized_detail_without_touching_silhouette():
    rgba = np.zeros((33, 41, 4), np.uint8)
    rgba[5:28, 7:34, 3] = 255
    # A narrow painted ridge is below the useful detail scale of the AI normal.
    rgba[5:28, 7:34, :3] = 96
    rgba[5:28, 19:22, :3] = 220
    vectors = np.zeros((33, 41, 3), np.float32)
    vectors[..., 2] = 1

    detailed = postprocess_ai_vectors(
        vectors, rgba, smoothing=0, detail_strength=0.4,
    )
    assert detailed.shape == vectors.shape
    assert np.max(np.abs(detailed[10:23, 16:25, 0])) > 0.1
    assert np.allclose(np.linalg.norm(detailed, axis=-1), 1, atol=1e-5)
    assert np.allclose(detailed[rgba[..., 3] == 0], (0, 0, 1))
    flat_rgba = rgba.copy()
    flat_rgba[flat_rgba[..., 3] > 0, :3] = 96
    silhouette = postprocess_ai_vectors(
        vectors, flat_rgba, smoothing=0, detail_strength=0.4,
    )
    assert np.max(np.abs(silhouette[..., :2])) < 1e-5
    # With details disabled, the original broad AI orientation is unchanged.
    plain = postprocess_ai_vectors(vectors, rgba, smoothing=0, detail_strength=0)
    assert np.allclose(plain, vectors)


def test_ai_axis_corrections_are_independent():
    vectors = np.array([[[0.6, -0.3, 0.74]]], np.float32)
    original = vectors.copy()
    corrected = orient_ai_vectors(vectors, invert_x=True)
    assert np.array_equal(vectors, original)
    assert np.allclose(corrected, [[[-0.6, -0.3, 0.74]]])
    assert np.array_equal(corrected[..., 1:], vectors[..., 1:])
    corrected_y = orient_ai_vectors(vectors, invert_y=True)
    assert np.allclose(corrected_y, [[[0.6, 0.3, 0.74]]])
    assert np.array_equal(corrected_y[..., (0, 2)], vectors[..., (0, 2)])
    corrected_both = orient_ai_vectors(vectors, invert_x=True, invert_y=True)
    assert np.allclose(corrected_both, [[[-0.6, 0.3, 0.74]]])
    alpha = np.array([[113]], np.uint8)
    gl = ai_normal(corrected_both, alpha, "OpenGL")
    dx = ai_normal(corrected_both, alpha, "DirectX")
    assert gl[0, 0, 1] > 128 > dx[0, 0, 1]
    assert np.array_equal(gl[..., 3], alpha)
