import numpy as np
import pytest
from PIL import Image

from smg.ao import ao_from_depth
from smg.export import export_ao
from smg.ui.lighting_preview import render_lighting
from smg.normal import normalize_vectors


def groove_normals(size=41, convex=False):
    y, x = np.mgrid[:size, :size].astype(np.float32) - size // 2
    relief = (-1 if not convex else 1) * 3 * np.exp(-(x*x + y*y) / 18)
    gy, gx = np.gradient(relief)
    return normalize_vectors(np.dstack((-gx, gy, np.ones_like(gx))))


@pytest.mark.parametrize("with_normals", [False, True])
def test_inclined_plane_does_not_shadow_itself(with_normals):
    y, x = np.mgrid[:29, :37].astype(np.float32)
    depth = 0.1 + x * 0.012 + y * 0.007
    alpha = np.full(depth.shape, 255, np.uint8)
    normals = np.broadcast_to(normalize_vectors(np.array([[[-0.12, 0.07, 1]]])), (*depth.shape, 3))
    ao = ao_from_depth(depth, alpha, radius=16, height_scale=10,
                       normals=normals if with_normals else None)
    assert np.allclose(ao, 1, atol=1e-6)


def test_normals_reveal_a_groove_missing_from_flat_depth_but_not_a_convex_peak():
    depth = np.full((41, 41), 0.5, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    plain = ao_from_depth(depth, alpha, radius=16)
    concave = ao_from_depth(depth, alpha, radius=16, normals=groove_normals())
    convex = ao_from_depth(depth, alpha, radius=16, normals=groove_normals(convex=True))
    assert plain[20, 20] == 1
    assert concave[20, 20] < 0.97
    assert convex[20, 20] > 0.999
    assert concave[2, 2] > 0.999


def test_constant_normal_tilt_does_not_create_relief_or_shadow():
    depth = np.full((23, 29), 0.5, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    normals = np.broadcast_to(np.array((0.7, -0.3, 0.5), np.float32), (*depth.shape, 3))
    assert np.allclose(ao_from_depth(depth, alpha, normals=normals), 1)


def test_normal_hemisphere_changes_which_wall_occludes():
    depth = np.full((33, 33), 0.3, np.float32)
    depth[:, 22:] = 0.7
    alpha = np.full(depth.shape, 255, np.uint8)
    toward = np.broadcast_to(np.array((0.6, 0, 0.8), np.float32), (*depth.shape, 3))
    away = toward.copy()
    away[..., 0] *= -1
    ao_toward = ao_from_depth(depth, alpha, radius=16, normals=toward)
    ao_away = ao_from_depth(depth, alpha, radius=16, normals=away)
    assert ao_toward[16, 18] < ao_away[16, 18]


def test_guided_ao_is_independent_of_padding_background_and_other_sprites():
    depth = np.full((41, 41), 0.5, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    normals = groove_normals()
    expected = ao_from_depth(depth, alpha, radius=64, normals=normals)
    atlas_depth = np.full((57, 98), np.nan, np.float32)
    atlas_alpha = np.zeros(atlas_depth.shape, np.uint8)
    atlas_normals = np.full((*atlas_depth.shape, 3), np.nan, np.float32)
    region = np.s_[8:49, 7:48]
    atlas_depth[region], atlas_alpha[region], atlas_normals[region] = depth, alpha, normals
    atlas_alpha[10:40, 53:92] = 255
    atlas_depth[10:40, 53:92] = 1
    atlas_normals[10:40, 53:92] = (0.5, 0.5, 0.7)
    actual = ao_from_depth(atlas_depth, atlas_alpha, radius=64, normals=atlas_normals)
    assert np.allclose(actual[region], expected, atol=1e-6)
    assert np.all(actual[atlas_alpha == 0] == 1)


def test_guided_ao_rotation_preserves_opengl_y_orientation():
    depth = np.full((41, 41), 0.5, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    normals = groove_normals()
    # Make the groove asymmetric before rotating its image and vector basis.
    normals[..., 0] *= 0.4
    rotated = np.rot90(normals).copy()
    rotated[..., 0], rotated[..., 1] = -np.rot90(normals[..., 1]), np.rot90(normals[..., 0])
    expected = np.rot90(ao_from_depth(depth, alpha, normals=normals, radius=16))
    actual = ao_from_depth(depth, alpha, normals=rotated, radius=16)
    assert np.allclose(actual, expected, atol=2e-5)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
def test_tiny_sprites_remain_finite(shape):
    depth = np.full(shape, 0.5, np.float32)
    normals = np.broadcast_to(np.array((0, 0, 1), np.float32), (*shape, 3))
    ao = ao_from_depth(depth, np.full(shape, 255, np.uint8), normals=normals, radius=1)
    assert ao.dtype == np.float32 and np.all(ao == 1)


def test_guided_ao_strength_is_monotone_and_zero_disables_it():
    depth = np.full((41, 41), 0.5, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    normals = groove_normals()
    weak = ao_from_depth(depth, alpha, strength=1, normals=normals)
    strong = ao_from_depth(depth, alpha, strength=3, normals=normals)
    assert np.all(strong <= weak + 1e-7)
    assert np.all(ao_from_depth(depth, alpha, strength=0, normals=normals) == 1)


@pytest.mark.parametrize("cup_radius", [5, 10])
def test_cup_ao_matches_independent_cosine_weighted_ray_tracing(cup_radius):
    y, x = np.mgrid[:81, :81] - 40
    height = np.where(x*x + y*y >= cup_radius**2, 3, 0).astype(np.float32)
    rng = np.random.default_rng(42)
    count = 20000
    radial = np.sqrt(rng.random(count))
    angle = rng.uniform(0, 2*np.pi, count)
    rays = np.stack((radial*np.cos(angle), radial*np.sin(angle), np.sqrt(1-radial*radial)), axis=-1)
    first_hit = np.full(count, np.inf)
    # Sample continuous hemisphere rays independently of the horizon algorithm.
    for distance in np.arange(0.25, 24.01, 0.25):
        points = rays * distance
        px = np.rint(points[:, 0]).astype(int) + 40
        py = np.rint(points[:, 1]).astype(int) + 40
        hit = (points[:, 2] + 0.025 <= height[py, px]) & ~np.isfinite(first_hit)
        first_hit[hit] = distance
    reference = (1 - np.exp(-2 * (first_hit/24)**2).mean()) ** 2
    ao = ao_from_depth(height/10, np.full(height.shape, 255, np.uint8), height_scale=10)
    assert abs(float(ao[40, 40]) - reference) < 0.03


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"radius": 1.5}, {"strength": float("nan")},
                                   {"height_scale": -1}, {"normals": np.zeros((2, 2, 3))}])
def test_invalid_ao_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ao_from_depth(np.zeros((3, 3)), np.full((3, 3), 255), **kwargs)


def test_ui_normal_settings_invalidate_ao_but_output_convention_does_not():
    from PySide6.QtWidgets import QApplication
    from smg.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.source = np.full((41, 41, 4), 255, np.uint8)
    window._generated(np.full((41, 41), 0.5, np.float32))
    window._ai_generated(groove_normals())
    window.mode.setCurrentText("AO")
    before = window._selected_ao().copy()
    window.convention.setCurrentText("DirectX")
    assert np.allclose(window._selected_ao(), before)
    window.ai_smoothing.setValue(6)
    assert not np.allclose(window._selected_ao(), before)
    window.close()
    app.processEvents()


def test_ui_discards_background_ao_from_outdated_geometry():
    from PySide6.QtWidgets import QApplication
    from smg.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.source = np.full((5, 7, 4), 255, np.uint8)
    window._generated(np.full((5, 7), 0.5, np.float32))
    old_revision = window._ao_revision
    window._invalidate_ao()
    outdated = np.zeros((5, 7), np.float32)
    window._ao_calculated(old_revision, outdated)
    assert window._preview_ao is None
    current = np.full((5, 7), 0.8, np.float32)
    window._ao_calculated(window._ao_revision, current)
    assert window._preview_ao is current
    window.close()
    app.processEvents()


@pytest.mark.parametrize("needs_depth,needs_normal", [(True, True), (False, True), (True, False)])
def test_ao_worker_only_generates_missing_geometry(monkeypatch, needs_depth, needs_normal):
    from smg.ui import main_window

    calls = []
    monkeypatch.setattr(main_window, "prepare_environment", lambda models, *a, **kw: calls.append(tuple(models)))
    monkeypatch.setattr(main_window, "DepthModel", lambda: object())
    monkeypatch.setattr(main_window, "generate_depth", lambda rgba, *a: np.full(rgba.shape[:2], 0.5, np.float32))
    monkeypatch.setattr(main_window, "detect_tree_crown", lambda *a: calls.append("crown"))

    class Model:
        def __init__(self, fov):
            pass

        def generate(self, rgba, *a):
            calls.append("normal")
            return np.broadcast_to(np.array((0, 0, 1), np.float32), (*rgba.shape[:2], 3)).copy()

    monkeypatch.setattr(main_window, "DSINENormalModel", Model)
    worker = main_window.GenerateAOThread(np.full((5, 7, 4), 255, np.uint8),
                                        needs_depth, needs_normal, 60, None)
    outputs = []
    worker.depth_generated.connect(lambda data: outputs.append("depth"))
    worker.normal_generated.connect(lambda data: outputs.append("normal"))
    worker.completed.connect(lambda: outputs.append("done"))
    worker.run()
    assert outputs == (["depth"] if needs_depth else []) + (["normal"] if needs_normal else []) + ["done"]
    assert calls[0] == (("depth",) if needs_depth else ()) + (("ai", "clipseg") if needs_normal else ())


def test_flat_depth_is_white_and_a_cavity_is_darker():
    depth = np.full((33, 33), 0.8, np.float32)
    alpha = np.full(depth.shape, 255, np.uint8)
    assert np.allclose(ao_from_depth(depth, alpha), 1)

    depth[12:21, 12:21] = 0.2
    ao = ao_from_depth(depth, alpha, radius=10)
    assert ao[16, 16] < ao[2, 2]
    assert 0 <= ao.min() <= ao.max() <= 1


def test_separate_alpha_islands_do_not_occlude_each_other():
    depth = np.zeros((11, 15), np.float32)
    alpha = np.zeros(depth.shape, np.uint8)
    alpha[3:8, 2:6] = 255
    alpha[3:8, 8:12] = 255
    depth[3:8, 8:12] = 1
    ao = ao_from_depth(depth, alpha, radius=8)
    assert np.allclose(ao[3:8, 2:6], 1)


def test_export_ao_preserves_alpha(tmp_path):
    alpha = np.array([[0, 128], [255, 32]], np.uint8)
    path = export_ao("sprite.png", np.array([[1, 0.5], [0, 0.25]]), alpha, tmp_path)
    rgba = np.array(Image.open(path).convert("RGBA"))
    assert path.name == "sprite_ao.png"
    assert np.array_equal(rgba[..., 3], alpha)
    assert rgba[0, 0, 0] == 255 and rgba[1, 0, 0] == 0


def test_ui_ao_uses_processed_depth_without_running_model():
    from PySide6.QtWidgets import QApplication
    from smg.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.source = np.full((33, 33, 4), 255, np.uint8)
    depth = np.full((33, 33), 0.8, np.float32)
    depth[12:21, 12:21] = 0.2
    window._generated(depth)
    window.ai_vectors = np.zeros((33, 33, 3), np.float32)
    window.ai_vectors[..., 2] = 1
    window.generate_ao()
    assert window.mode.currentText() == "AO"
    before = window._selected_ao().copy()
    window.smooth.setValue(30)
    after = window._selected_ao()
    assert not np.array_equal(after, before)
    window.close()
    app.processEvents()


def test_lighting_preview_darkens_with_ao_and_preserves_alpha():
    source = np.array([[[120, 100, 80, 255], [120, 100, 80, 128]]], np.uint8)
    normal = np.full_like(source, (128, 128, 255, 255))
    ao = np.array([[1, 0.5]], np.float32)

    lit = render_lighting(source, normal, 0, 0, ao)
    assert np.array_equal(lit[0, 1, :3], lit[0, 0, :3] // 2)
    assert np.array_equal(lit[..., 3], source[..., 3])


def test_lighting_preview_recalculates_ao_when_strength_changes(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from smg.ui import main_window

    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.full((33, 33, 4), 255, np.uint8)
    depth = np.full((33, 33), 0.8, np.float32)
    depth[12:21, 12:21] = 0.2
    window._generated(depth)
    window._preview_normal = np.full_like(window.source, (128, 128, 255, 255))
    window.ai_vectors = np.zeros((33, 33, 3), np.float32)
    seen = []

    def capture(base, normal, light_x, light_y, ao, roughness=None):
        seen.append(ao.copy())
        return base

    monkeypatch.setattr(main_window, "render_lighting", capture)
    window.mode.setCurrentText("Lighting Preview")
    assert seen[-1][16, 16] < seen[-1][2, 2]

    window.ao_strength.setValue(0)
    assert len(seen) >= 2
    assert np.allclose(seen[-1], 1)
    window.close()
    app.processEvents()
