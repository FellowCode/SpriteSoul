import os
import threading
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from smg.ui import main_window


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _wait_for(app, condition):
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert condition(), "Setup worker did not deliver its progress in time"


@pytest.fixture
def setup_window():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    window.show()
    yield app, window
    if window.worker is not None:
        assert window.worker.wait(5000)
        _wait_for(app, lambda: window.worker is None)
    if window.setup_dialog is not None:
        window.setup_dialog.close()
    window.close()
    app.processEvents()


def test_prepare_ai_tracks_threaded_stages_and_keeps_success_visible(monkeypatch, setup_window):
    app, window = setup_window
    continue_depth = threading.Event()
    continue_ai = threading.Event()
    prepared = []

    def prepare(models, progress, events):
        prepared.append(tuple(models))
        events({"phase": "cuda", "status": "start", "message": "Проверка CUDA"})
        events({"phase": "cuda", "status": "ready", "message": "CUDA готова"})
        events({"phase": "model_download", "model": "depth", "status": "cached",
                "message": "config.json", "unit": "files", "current": 1, "total": 3})
        events({"phase": "model_download", "model": "depth", "status": "start",
                "message": "preprocessor_config.json", "unit": "files", "current": 1, "total": 3})
        assert continue_depth.wait(5)
        events({"phase": "model_download", "model": "depth", "status": "done",
                "message": "Depth готова", "unit": "files", "current": 3, "total": 3})
        events({"phase": "pip_install", "component": "geffnet", "status": "cached",
                "message": "geffnet уже установлен"})
        events({"phase": "model_download", "model": "ai", "status": "start",
                "message": "DSINE: исходный код", "unit": "files", "current": 0, "total": 2})
        events({"phase": "model_download", "model": "ai", "status": "update",
                "message": "Загрузка кода DSINE", "unit": "bytes", "current": 100, "total": 100})
        assert continue_ai.wait(5)
        events({"phase": "model_download", "model": "ai", "status": "done",
                "message": "DSINE готова", "unit": "files", "current": 2, "total": 2})
        events({"phase": "model_download", "model": "clipseg", "status": "cached",
                "message": "CLIPSeg готова", "unit": "files", "current": 7, "total": 7})

    monkeypatch.setattr(main_window, "prepare_environment", prepare)
    window.setup_action.trigger()
    dialog = window.setup_dialog
    try:
        assert dialog.isVisible()
        assert dialog.windowModality() == Qt.WindowModal
        assert not dialog.close_button.isEnabled()
        _wait_for(app, lambda: dialog.steps["depth"].progress.value() == 33)
        assert prepared == [("depth", "ai", "clipseg", "roughness")]
        assert dialog.steps["cuda"].state == "done"
        assert dialog.steps["depth"].state == "running"
        assert dialog.steps["geffnet"].state == "waiting"
        assert dialog.overall.value() == 1
        assert not window.setup_action.isEnabled()
        dialog.reject()
        dialog.close()
        assert dialog.isVisible()

        continue_depth.set()
        _wait_for(app, lambda: dialog.steps["ai"].progress.value() == 50)
        assert dialog.steps["ai"].state == "running"
        assert dialog.overall.value() == 3
        continue_ai.set()
        _wait_for(app, lambda: window.worker is None)
        assert dialog.isVisible()
        assert all(step.state == "done" and step.progress.value() == 100
                   for step in dialog.steps.values())
        assert dialog.overall.value() == 6
        assert "Подготовка завершена" in dialog.summary.text()
        assert dialog.close_button.isEnabled()
        assert window.setup_action.isEnabled()
        dialog.close_button.click()
        assert window.setup_dialog is None
        assert not dialog.isVisible()
    finally:
        continue_depth.set()
        continue_ai.set()


def test_prepare_ai_shows_failure_and_preserves_completed_stages(monkeypatch, setup_window):
    app, window = setup_window

    def prepare(models, progress, events):
        events({"phase": "cuda", "status": "ready", "message": "CUDA готова"})
        events({"phase": "model_download", "model": "depth", "status": "start",
                "message": "Загрузка Depth", "unit": "files", "current": 2, "total": 3})
        raise RuntimeError("Не удалось скачать модель: соединение прервано")

    monkeypatch.setattr(main_window, "prepare_environment", prepare)
    monkeypatch.setattr(main_window.QMessageBox, "critical",
                        lambda *args: pytest.fail("Failure should be shown in the setup dialog"))
    window.prepare_ai()
    dialog = window.setup_dialog
    _wait_for(app, lambda: window.worker is None)
    assert dialog.isVisible()
    assert dialog.steps["cuda"].state == "done"
    assert dialog.steps["depth"].state == "failed"
    assert dialog.steps["depth"].progress.value() == 66
    assert all(dialog.steps[key].state == "skipped" for key in ("geffnet", "ai", "clipseg", "roughness"))
    assert dialog.overall.value() == 1
    assert dialog.error.isVisible()
    assert "соединение прервано" in dialog.error.text()
    assert window.statusBar().currentMessage() == "Подготовка AI не удалась"
    assert dialog.close_button.isEnabled()
    dialog.close_button.click()
    assert window.setup_dialog is None

    window.prepare_ai()
    assert window.setup_dialog is not dialog
    assert all(step.state == "waiting" for step in window.setup_dialog.steps.values())
    _wait_for(app, lambda: window.worker is None)
