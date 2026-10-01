import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from PySide6.QtWidgets import QApplication

from smg.ui import main_window
from smg.ui.setup_dialog import SetupProgressDialog


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    instance = main_window.MainWindow()
    instance.source = np.full((3, 5, 4), 255, np.uint8)
    instance.source[..., 3] = np.arange(15, dtype=np.uint8).reshape(3, 5)
    instance.source_path = Path("sprite.png")
    instance._update_actions()
    yield instance
    instance.close()
    app.processEvents()


def test_roughness_preview_and_standalone_export(window, monkeypatch, tmp_path):
    rendered = []
    monkeypatch.setattr(window.view, "set_pixels", lambda pixels, fit=False: rendered.append(pixels.copy()))
    window._roughness_generated(np.full((3, 5), 0.5, np.float32))
    assert window.mode.currentText() == "Roughness"
    assert np.all(rendered[-1][..., :3] == 128)
    assert np.array_equal(rendered[-1][..., 3], window.source[..., 3])
    assert window.export_action.isEnabled()
    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window.export()
    assert {path.name for path in tmp_path.iterdir()} == {"sprite_roughness.png"}
    exported = np.asarray(Image.open(tmp_path / "sprite_roughness.png"))
    assert np.array_equal(exported, rendered[-1])


def test_roughness_reset_when_opening_new_source(window, monkeypatch, tmp_path):
    window.roughness = np.ones((3, 5), np.float32)
    source = tmp_path / "new.png"
    Image.fromarray(window.source).save(source)
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    window.open_file()
    assert window.roughness is None
    assert window.mode.currentText() == "Source"
    assert not window.export_action.isEnabled()


def test_lighting_preview_uses_roughness_and_refreshes_with_light(window, monkeypatch):
    window.source[..., :3] = 80
    window.ai_vectors = np.zeros((3, 5, 3), np.float32)
    window.ai_vectors[..., 2] = 1
    window._preview_normal = np.full_like(window.source, (128, 128, 255, 255))
    rendered = []
    monkeypatch.setattr(window.view, "set_pixels", lambda pixels, fit=False: rendered.append(pixels.copy()))
    window._roughness_generated(np.tile(np.linspace(0, 1, 5, dtype=np.float32), (3, 1)))
    window.light = (0, 0)
    window.mode.setCurrentText("Lighting Preview")
    centered = rendered[-1]
    assert centered[0, 0, 0] > centered[0, -1, 0]
    assert np.array_equal(centered[..., 3], window.source[..., 3])
    window._light_changed(1, 0)
    assert rendered[-1][0, 0, 0] < centered[0, 0, 0]
    window.roughness = None
    window._refresh()
    assert np.all(rendered[-1][..., 0] == rendered[-1][0, 0, 0])


def test_roughness_worker_prepares_only_supermat(monkeypatch):
    prepared, generated, failures = [], [], []
    source = np.full((3, 5, 4), 255, np.uint8)
    monkeypatch.setattr(main_window, "prepare_environment", lambda models, progress, events:
                        prepared.append(tuple(models)))
    monkeypatch.setattr(main_window, "generate_roughness", lambda rgba, progress:
                        np.full(rgba.shape[:2], 0.4, np.float32))
    worker = main_window.GenerateRoughnessThread(source)
    worker.generated.connect(generated.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert prepared == [("roughness",)]
    assert not failures and np.all(generated[0] == np.float32(0.4))


def test_large_download_progress_does_not_overflow_qt(window):
    window._show_progress_event({"phase": "model_download", "status": "update",
                                 "unit": "bytes", "current": 2_000_000_000,
                                 "total": 4_000_000_000})
    assert window.download_progress.value() == 500
    assert window.download_progress.maximum() == 1000


def test_setup_dialog_routes_runtime_and_supermat_downloads(window):
    dialog = SetupProgressDialog(window)
    dialog.update_progress({"phase": "model_download", "model": "roughness", "status": "start",
                            "unit": "files", "current": 1, "total": 14})
    dialog.update_progress({"phase": "pip_install", "component": "supermat-runtime",
                            "status": "update", "message": "Библиотеки SuperMat",
                            "unit": "bytes", "current": 20, "total": 100})
    assert dialog.steps["roughness"].state == "running"
    assert dialog.steps["roughness"].detail.text() == "Библиотеки SuperMat"
    dialog.update_progress({"phase": "model_download", "model": "roughness", "status": "cached",
                            "unit": "files", "current": 14, "total": 14})
    assert dialog.steps["roughness"].state == "done"
    assert dialog.overall.value() == 1
    dialog.finish()
    dialog.close()
