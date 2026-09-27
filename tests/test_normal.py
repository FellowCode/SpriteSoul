import numpy as np

from smg.normal import ai_normal, dsine_to_opengl, hybrid_normal, normals_from_depth, orient_ai_vectors


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


def test_ai_hybrid_endpoints_convention_and_alignment():
    alpha = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7
    depth = np.tile(np.linspace(0, 1, 7, dtype=np.float32), (5, 1))
    base = normals_from_depth(depth, alpha)
    raw_dsine = np.zeros((5, 7, 3), np.float32)
    raw_dsine[..., 1:] = (0.6, 0.8)  # DSINE +Y points down.
    vectors = dsine_to_opengl(raw_dsine)
    assert np.allclose(vectors[2, 3], (0, -0.6, 0.8))
    ai_gl = ai_normal(vectors, alpha, "OpenGL")
    ai_dx = ai_normal(vectors, alpha, "DirectX")
    assert ai_gl.shape == (5, 7, 4)
    assert np.array_equal(ai_gl[..., 3], alpha)
    assert ai_gl[2, 3, 1] < 128 < ai_dx[2, 3, 1]
    assert np.array_equal(hybrid_normal(base, vectors, alpha, 0), base)
    assert np.array_equal(hybrid_normal(base, vectors, alpha, 1), ai_gl)
    mixed = hybrid_normal(base, vectors, alpha, 0.35)
    lengths = np.linalg.norm(mixed[..., :3].astype(np.float32) / 127.5 - 1, axis=-1)
    assert np.allclose(lengths, 1, atol=0.01)


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
