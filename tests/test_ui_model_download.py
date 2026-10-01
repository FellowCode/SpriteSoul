import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QApplication

from smg.ui import main_window


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _StatusBar:
    def __init__(self):
        self.message = None

    def showMessage(self, message):
        self.message = message


class _Window:
    def __init__(self):
        self._status = _StatusBar()

    def statusBar(self):
        return self._status


def test_missing_model_is_offered_for_download(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_window, "missing_models", lambda models: ("depth",))
    def answer_yes(*args):
        captured["text"] = args[2]
        return QMessageBox.Yes

    monkeypatch.setattr(main_window.QMessageBox, "question", answer_yes)
    window = _Window()
    assert main_window.MainWindow._confirm_model_download(window, ("depth",))
    assert "Depth Anything V2" in captured["text"]
    assert "models" in captured["text"]


def test_declining_model_download_cancels_operation(monkeypatch):
    monkeypatch.setattr(main_window, "missing_models", lambda models: ("ai",))
    monkeypatch.setattr(main_window.QMessageBox, "question", lambda *args: QMessageBox.No)
    window = _Window()
    assert not main_window.MainWindow._confirm_model_download(window, ("ai",))
    assert window._status.message == "Загрузка модели отменена"


def test_generate_menu_groups_all_map_actions():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    assert window.generate_menu.title() == "Генерировать"
    assert [action.text() for action in window.generate_menu.actions()] == [
        "Карта глубины", "Карта AO", "Карта нормалей", "Albedo", "Карта шероховатости", "Все карты",
    ]
    assert not hasattr(window, "normal_source")
    assert window.ai_smoothing.value() == 1.5
    assert window.ai_details.value() == 0.35
    assert window.select_foliage_action.text() == "Определить крону"
    assert window.mode.itemText(window.mode.count() - 1) == "Маска кроны"
    assert "SAM" not in " ".join(action.text() for action in window.findChildren(main_window.QAction))
    window.close()
    app.processEvents()


def test_custom_controls_follow_system_color_scheme_without_styling_dialog_labels():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()

    window._apply_system_theme(Qt.ColorScheme.Dark)
    dark_styles = window.styleSheet()
    assert "background: #20252e" in dark_styles
    assert "color: #e7edf5" in dark_styles
    assert "\n            QLabel," not in dark_styles

    window._apply_system_theme(Qt.ColorScheme.Light)
    assert "background: #f5f6f8" in window.styleSheet()
    window.close()
    app.processEvents()


def test_lighting_preview_prefers_albedo_and_falls_back_to_source(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.full((4, 5, 4), (20, 30, 40, 255), np.uint8)
    window.ai_vectors = np.zeros((4, 5, 3), np.float32)
    window.ai_vectors[..., 2] = 1
    window._preview_normal = window._selected_normal("OpenGL")
    window.mode.blockSignals(True)
    window.mode.setCurrentText("Lighting Preview")
    window.mode.blockSignals(False)
    bases = []

    def fake_render(base, normal, light_x, light_y, ao=None, roughness=None):
        bases.append(base)
        return base

    monkeypatch.setattr(main_window, "render_lighting", fake_render)
    window.albedo = np.full_like(window.source, (100, 110, 120, 255))
    window._refresh()
    assert bases[-1] is window.albedo
    window.albedo = None
    window._refresh()
    assert bases[-1] is window.source
    window.close()
    app.processEvents()


def test_crown_detection_threshold_confirm_cancel_and_redetect():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.full((4, 5, 4), 255, np.uint8)
    saved = np.zeros((4, 5), bool)
    saved[0, 0] = True
    window.foliage_mask = saved.copy()
    window._selection_active = True
    scores = np.zeros((4, 5), np.float32)
    scores[2, 3] = 0.7
    scores[1, 4] = 0.9
    window.source[1, 4, 3] = 0
    window._selection_scores_generated(scores)
    assert not window.open_action.isEnabled()
    assert not window.generate_menu.menuAction().isEnabled()
    assert window._selection_mask[2, 3]
    assert not window._selection_mask[1, 4]
    window.selection_threshold.setValue(80)
    assert not window._selection_mask.any()
    assert not window.selection_done.isEnabled()
    window.selection_threshold.setValue(50)
    window.finish_foliage_selection()
    candidate = window.foliage_mask.copy()
    assert candidate[2, 3] and candidate.sum() == 1
    assert window.open_action.isEnabled()

    window._selection_active = True
    window._selection_scores_generated(np.full((4, 5), 0.95, np.float32))
    window.cancel_foliage_selection()
    assert np.array_equal(window.foliage_mask, candidate)
    window.close()
    app.processEvents()


def test_crown_normal_updates_before_and_after_ai_generation():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.full((56, 64, 4), 120, np.uint8)
    window.source[..., 3] = 255
    front = np.zeros((56, 64, 3), np.float32)
    front[..., 2] = 1
    mask = np.zeros((56, 64), bool)
    mask[8:48, 10:54] = True
    window.source[..., 3] = np.where(mask, 255, 0).astype(np.uint8)

    window.foliage_mask = mask
    window._ai_generated(front)
    assert window.mode.currentText() == "Normal"
    early = window._selected_normal("OpenGL")
    assert early[28, 17, 0] < 100 < early[28, 47, 0]

    window.foliage_mask = None
    window._refresh()
    flat = window._selected_normal("OpenGL")
    window._selection_active = True
    window._selection_scores_generated(mask.astype(np.float32))
    window.finish_foliage_selection()
    late = window._selected_normal("OpenGL")
    assert window.mode.currentText() == "Normal"
    assert np.array_equal(late, early)
    assert not np.array_equal(late, flat)
    window.close()
    app.processEvents()


def test_detected_crown_preview_and_saved_mask_reach_alpha_edges():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.zeros((50, 60, 4), np.uint8)
    window.source[5:45, 8:52] = (40, 120, 50, 255)
    scores = np.zeros((50, 60), np.float32)
    scores[12:38, 16:44] = 0.9
    window._selection_active = True
    window._selection_scores_generated(scores)
    assert np.array_equal(window._selection_mask, window.source[..., 3] > 0)
    window.finish_foliage_selection()
    assert np.array_equal(window.foliage_mask, window.source[..., 3] > 0)
    window.close()
    app.processEvents()


def test_normal_workers_detect_crown_only_without_saved_mask(monkeypatch):
    prepared = []
    detected = []
    monkeypatch.setattr(main_window, "prepare_environment",
                        lambda models, progress, events: prepared.append(tuple(models)))

    def fake_detect(rgba, progress):
        detected.append(True)
        return np.ones(rgba.shape[:2], bool)

    monkeypatch.setattr(main_window, "detect_tree_crown", fake_detect)

    class FlatAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(main_window, "DSINENormalModel", FlatAI)
    rgba = np.full((4, 5, 4), 255, np.uint8)
    worker = main_window.GenerateNormalThread(rgba, 60)
    results = []
    worker.generated.connect(results.append)
    worker.run()
    assert prepared == [("ai", "clipseg")]
    assert len(detected) == 1
    assert results[0][1].all()

    prepared.clear()
    saved = np.zeros(rgba.shape[:2], bool)
    saved[1, 2] = True
    worker = main_window.GenerateNormalThread(rgba, 60, saved)
    worker.generated.connect(results.append)
    worker.run()
    assert prepared == [("ai",)]
    assert len(detected) == 1
    assert np.array_equal(results[-1][1], saved)


def test_auto_crown_is_applied_when_normal_arrives():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.source = np.full((56, 64, 4), 120, np.uint8)
    window.source[..., 3] = 255
    front = np.zeros((56, 64, 3), np.float32)
    front[..., 2] = 1
    mask = np.zeros((56, 64), bool)
    mask[8:48, 10:54] = True
    window._ai_generated((front, mask))
    assert np.array_equal(window.foliage_mask, mask)
    curved = window._selected_normal("OpenGL")
    assert curved[28, 17, 0] < 100 < curved[28, 47, 0]
    window.foliage_mask = None
    window._ai_generated((front, None))
    assert window.foliage_mask is None
    assert window._selected_normal("OpenGL")[28, 17, 0] == 128
    window.close()
    app.processEvents()


def test_generate_all_worker_detects_crown(monkeypatch):
    prepared = []
    monkeypatch.setattr(main_window, "prepare_environment",
                        lambda models, progress, events: prepared.append(tuple(models)))
    monkeypatch.setattr(main_window, "DepthModel", lambda: object())
    monkeypatch.setattr(main_window, "generate_depth",
                        lambda rgba, model, progress: np.zeros(rgba.shape[:2], np.float32))
    monkeypatch.setattr(main_window, "generate_albedo", lambda rgba, progress: rgba)
    monkeypatch.setattr(main_window, "generate_roughness",
                        lambda rgba, progress: np.full(rgba.shape[:2], 0.5, np.float32))
    monkeypatch.setattr(main_window, "detect_tree_crown",
                        lambda rgba, progress: np.ones(rgba.shape[:2], bool))

    class FlatAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress):
            return np.zeros((*rgba.shape[:2], 3), np.float32)

    monkeypatch.setattr(main_window, "DSINENormalModel", FlatAI)
    worker = main_window.GenerateAllThread(np.full((4, 5, 4), 255, np.uint8), 60)
    results = []
    failures = []
    worker.normal_generated.connect(results.append)
    roughness_results = []
    worker.roughness_generated.connect(roughness_results.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert failures == []
    assert prepared == [("depth", "ai", "albedo", "roughness", "clipseg")]
    assert results[0][1].all()
    assert roughness_results[0].shape == (4, 5)
