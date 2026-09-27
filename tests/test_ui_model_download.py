import os

import numpy as np
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
        "Карта глубины", "Карта нормалей", "Albedo", "Все карты",
    ]
    assert not hasattr(window, "normal_source")
    assert window.ai_smoothing.value() == 1.5
    assert window.ai_details.value() == 0.35
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

    def fake_render(base, normal, light_x, light_y):
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
