import numpy as np
import pytest

from smg.ui.lighting_preview import render_lighting


def test_roughness_controls_highlight_peak_and_width():
    source = np.full((1, 3, 4), (80, 100, 120, 128), np.uint8)
    # Three copies of the same material with different roughness.
    roughness = np.array([[0, 0.5, 1]], np.float32)
    normal = np.full_like(source, (128, 128, 255, 255))
    head_on = render_lighting(source, normal, 0, 0, roughness=roughness)
    assert head_on[0, 0, 0] > head_on[0, 1, 0] > head_on[0, 2, 0]

    # Tilting the surface moves it outside the smooth material's narrow lobe,
    # while a rough surface still reflects a weak, broad highlight.
    normal[..., :3] = (201, 128, 232)
    diffuse = render_lighting(source, normal, 0, 0)
    tilted = render_lighting(source, normal, 0, 0, roughness=roughness)
    assert tilted[0, 0, 0] == diffuse[0, 0, 0]
    assert tilted[0, 2, 0] > diffuse[0, 2, 0]
    assert np.array_equal(tilted[..., 3], source[..., 3])


def test_roughness_highlight_follows_light_position():
    source = np.full((1, 1, 4), (80, 100, 120, 128), np.uint8)
    normal = np.full_like(source, (128, 128, 255, 255))
    roughness = np.zeros((1, 1), np.float32)
    centered = render_lighting(source, normal, 0, 0, roughness=roughness)
    moved = render_lighting(source, normal, 1, 0, roughness=roughness)
    moved_diffuse = render_lighting(source, normal, 1, 0)
    assert centered[0, 0, 0] > moved[0, 0, 0]
    assert np.array_equal(moved, moved_diffuse)


def test_ao_darkens_highlights_and_preserves_alpha():
    source = np.full((1, 3, 4), (50, 70, 90, 255), np.uint8)
    source[..., 3] = (0, 128, 255)
    normal = np.full_like(source, (128, 128, 255, 255))
    roughness = np.zeros((1, 3), np.float32)
    ao = np.array([[1, 0.5, 0]], np.float32)
    lit = render_lighting(source, normal, 0, 0, ao=ao, roughness=roughness)
    assert np.array_equal(lit[0, 1, :3], lit[0, 0, :3] // 2)
    assert np.all(lit[0, 2, :3] == 0)
    assert np.array_equal(lit[..., 3], source[..., 3])


def test_unlit_normals_do_not_receive_highlights():
    source = np.full((1, 2, 4), (80, 100, 120, 128), np.uint8)
    normal = np.array([[[255, 128, 128, 255], [128, 128, 0, 255]]], np.uint8)
    diffuse = render_lighting(source, normal, -1, 0)
    lit = render_lighting(source, normal, -1, 0, roughness=np.zeros((1, 2), np.float32))
    assert np.array_equal(lit, diffuse)


@pytest.mark.parametrize("use_ao", [False, True])
def test_no_roughness_preserves_diffuse_preview(use_ao):
    rng = np.random.default_rng(42)
    source = rng.integers(0, 256, (5, 7, 4), dtype=np.uint8)
    normal = rng.integers(0, 256, source.shape, dtype=np.uint8)
    ao = rng.random(source.shape[:2], dtype=np.float32) if use_ao else None
    direction = np.array((0.4, -0.6, 0.8), np.float32)
    direction /= np.linalg.norm(direction)
    vectors = normal[..., :3].astype(np.float32) / 127.5 - 1
    intensity = 0.28 + 0.92 * np.maximum(0, np.sum(vectors * direction, axis=-1))
    if ao is not None:
        intensity *= ao
    expected = source.copy()
    expected[..., :3] = np.clip(source[..., :3] * intensity[..., None], 0, 255).astype(np.uint8)
    assert np.array_equal(render_lighting(source, normal, 0.4, -0.6, ao), expected)
