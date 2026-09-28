import numpy as np
from PIL import Image

from smg.ao import ao_from_depth
from smg.export import export_ao
from smg.ui.lighting_preview import render_lighting


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


def test_ui_ao_uses_edited_depth_without_running_model():
    from PySide6.QtWidgets import QApplication
    from smg.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.source = np.full((33, 33, 4), 255, np.uint8)
    depth = np.full((33, 33), 0.8, np.float32)
    depth[12:21, 12:21] = 0.2
    window._generated(depth)
    window.generate_ao()
    assert window.mode.currentText() == "AO"
    before = window._selected_ao().copy()
    window.tool.setCurrentText("Raise")
    window.radius.setValue(3)
    window._brush_event("begin", 16, 16)
    window._brush_event("end", 16, 16)
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

    def capture(base, normal, light_x, light_y, ao):
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
