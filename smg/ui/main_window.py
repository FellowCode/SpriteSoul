from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSlider,
    QSpinBox, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from smg.depth.inference import DepthModel
from smg.depth.processing import DepthSettings
from smg.editor import DepthEditor
from smg.export import export_maps, load_project, save_project
from smg.normal import ai_normal, hybrid_normal, orient_ai_vectors
from smg.normal_ai import DSINENormalModel
from smg.pipeline import generate_depth, generate_normal, open_png
from smg.ui.image_view import ImageView
from smg.ui.lighting_preview import render_lighting


class GenerateThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, rgba: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()

    def run(self) -> None:
        try:
            depth = generate_depth(self.rgba, DepthModel(), self.progress.emit)
            self.generated.emit(depth)
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateNormalThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, rgba: np.ndarray, fov: float, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()
        self.fov = fov

    def run(self) -> None:
        try:
            normal = DSINENormalModel(self.fov).generate(self.rgba, self.progress.emit)
            self.generated.emit(normal)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sprite Soul")
        self.resize(1180, 760)
        self.source_path: Path | None = None
        self.source: np.ndarray | None = None
        self.editor: DepthEditor | None = None
        self._preview_normal: np.ndarray | None = None
        self.ai_vectors: np.ndarray | None = None  # OpenGL float XYZ, converted at inference time
        self.worker: GenerateThread | None = None
        self.light = (0.35, 0.45)
        self._building = True
        self._build_ui()
        self._building = False
        self._update_actions()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Откройте PNG для начала работы")

    def _build_ui(self) -> None:
        toolbar = QToolBar("Действия", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.open_action = self._action("Открыть", "Ctrl+O", self.open_file)
        self.generate_action = self._action("Генерировать", "Ctrl+G", self.generate)
        self.generate_ai_action = self._action("Генерировать AI Normal", "Ctrl+Shift+G", self.generate_ai_normal)
        self.undo_action = self._action("Отменить", "Ctrl+Z", self.undo)
        self.redo_action = self._action("Повторить", "Ctrl+Y", self.redo)
        self.export_action = self._action("Экспорт", "Ctrl+E", self.export)
        for action in (self.open_action, self.generate_action, self.generate_ai_action, self.undo_action,
                       self.redo_action, self.export_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        fit_action = self._action("Вписать", "Ctrl+0", self.fit_image)
        toolbar.addAction(fit_action)

        central = QWidget()
        self.setCentralWidget(central)
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.view = ImageView()
        self.view.brush_event.connect(self._brush_event)
        self.view.light_changed.connect(self._light_changed)
        row.addWidget(self.view, 1)

        panel = QFrame()
        panel.setObjectName("controls")
        panel.setFixedWidth(244)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        row.addWidget(panel)

        layout.addWidget(QLabel("Просмотр"))
        self.mode = QComboBox()
        self.mode.addItems(("Source", "Depth", "Normal", "Lighting Preview"))
        self.mode.currentTextChanged.connect(self._mode_changed)
        layout.addWidget(self.mode)

        layout.addWidget(QLabel("Depth"))
        self.invert = QCheckBox("Invert Depth")
        self.invert.toggled.connect(self._settings_changed)
        layout.addWidget(self.invert)
        form = QFormLayout()
        layout.addLayout(form)
        self.depth_strength = self._slider(1, 300, 100, self._settings_changed)
        self.contrast = self._slider(25, 300, 100, self._settings_changed)
        self.smooth = self._slider(0, 80, 0, self._settings_changed)
        form.addRow("Range", self.depth_strength)
        form.addRow("Contrast", self.contrast)
        form.addRow("Smooth", self.smooth)

        layout.addWidget(QLabel("Normal"))
        normal_form = QFormLayout()
        layout.addLayout(normal_form)
        self.normal_strength = self._slider(0, 1000, 200, self._normal_changed)
        normal_form.addRow("Strength", self.normal_strength)
        self.convention = QComboBox()
        self.convention.addItems(("OpenGL", "DirectX"))
        self.convention.currentTextChanged.connect(self._normal_changed)
        normal_form.addRow("Format", self.convention)
        self.normal_source = QComboBox()
        self.normal_source.addItems(("Depth", "AI", "Hybrid"))
        self.normal_source.currentTextChanged.connect(self._normal_source_changed)
        normal_form.addRow("Normal Source", self.normal_source)
        self.ai_influence = QDoubleSpinBox()
        self.ai_influence.setRange(0, 1)
        self.ai_influence.setSingleStep(0.01)
        self.ai_influence.setDecimals(2)
        self.ai_influence.setValue(0.35)
        self.ai_influence.valueChanged.connect(self._normal_changed)
        normal_form.addRow("AI Influence", self.ai_influence)
        self.ai_fov = QSpinBox()
        self.ai_fov.setRange(20, 120)
        self.ai_fov.setValue(60)
        self.ai_fov.setSuffix("°")
        self.ai_fov.valueChanged.connect(self._fov_changed)
        normal_form.addRow("DSINE FOV", self.ai_fov)
        self.invert_ai_x = QCheckBox("Invert AI X")
        self.invert_ai_x.setChecked(True)
        self.invert_ai_x.toggled.connect(self._normal_changed)
        normal_form.addRow(self.invert_ai_x)
        self.invert_ai_y = QCheckBox("Invert AI Y")
        self.invert_ai_y.setChecked(True)
        self.invert_ai_y.toggled.connect(self._normal_changed)
        normal_form.addRow(self.invert_ai_y)

        layout.addWidget(QLabel("Кисть"))
        self.tool = QComboBox()
        self.tool.addItems(("Pan", "Raise", "Lower", "Smooth"))
        self.tool.currentTextChanged.connect(self._tool_changed)
        layout.addWidget(self.tool)
        brush_form = QFormLayout()
        layout.addLayout(brush_form)
        self.radius = QSpinBox()
        self.radius.setRange(1, 256)
        self.radius.setValue(24)
        self.brush_strength = self._slider(1, 100, 12, lambda _value: None)
        brush_form.addRow("Radius", self.radius)
        brush_form.addRow("Strength", self.brush_strength)
        layout.addStretch()
        self.hint = QLabel("В режиме Lighting Preview тяните мышью по изображению, чтобы переместить свет.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #6b7280")
        layout.addWidget(self.hint)
        arrow_icon = (Path(__file__).resolve().parent / "icons" / "chevron-down.svg").as_posix()
        styles = """
            QFrame#controls { background: #f5f6f8; border-left: 1px solid #d7dbe1; }
            QLabel, QFrame#controls QCheckBox { color: #252b34; }
            QSlider { min-height: 22px; }
            QFrame#controls QComboBox, QFrame#controls QSpinBox, QFrame#controls QDoubleSpinBox {
                color: #252b34;
                background: #ffffff;
                border: 1px solid #aeb7c2;
                border-radius: 3px;
                min-height: 26px;
                padding: 0 7px;
            }
            QFrame#controls QComboBox { padding-right: 29px; }
            QFrame#controls QComboBox:hover, QFrame#controls QSpinBox:hover, QFrame#controls QDoubleSpinBox:hover {
                border-color: #66788b;
            }
            QFrame#controls QComboBox:focus, QFrame#controls QSpinBox:focus, QFrame#controls QDoubleSpinBox:focus {
                border-color: #3074a8;
            }
            QFrame#controls QComboBox::drop-down {
                background: #354454;
                border-left: 1px solid #354454;
                width: 25px;
            }
            QFrame#controls QComboBox::down-arrow {
                image: url(CHEVRON_ICON);
                width: 14px;
                height: 14px;
            }
            QComboBox QAbstractItemView {
                color: #252b34;
                background: #ffffff;
                selection-color: #172a3b;
                selection-background-color: #d5e7f7;
                border: 1px solid #aeb7c2;
                outline: none;
            }
        """
        self.setStyleSheet(styles.replace("CHEVRON_ICON", arrow_icon))

    def _action(self, text: str, shortcut: str, callback) -> QAction:
        action = QAction(text, self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        return action

    def _slider(self, low: int, high: int, value: int, callback) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        return slider

    def _update_actions(self) -> None:
        ready = self.source is not None
        busy = self.worker is not None and self.worker.isRunning()
        self.open_action.setEnabled(not busy)
        self.generate_action.setEnabled(ready and not busy)
        self.generate_ai_action.setEnabled(ready and not busy)
        self.ai_fov.setEnabled(not busy)
        self.export_action.setEnabled(self.editor is not None and not busy)
        self.undo_action.setEnabled(self.editor is not None and bool(self.editor.strokes) and not busy)
        self.redo_action.setEnabled(self.editor is not None and bool(self.editor.redo_strokes) and not busy)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть PNG или проект", "", "Sprite Soul (*.png *.ssoul);;PNG (*.png);;Project (*.ssoul)")
        if not path:
            return
        try:
            if path.lower().endswith(".ssoul"):
                source_path, depth, strength, convention = load_project(path)
                source = open_png(source_path)
            else:
                source_path = Path(path)
                source = open_png(path)
                depth = None
            self.source_path = source_path
            self.source = source
            self.editor = DepthEditor(depth, source[..., 3]) if depth is not None else None
            self._preview_normal = None
            self.ai_vectors = None
            self.normal_source.setCurrentText("Depth")
            self.ai_influence.setValue(0.35)
            self.ai_fov.setValue(60)
            for widget, value in ((self.invert, False), (self.depth_strength, 100),
                                  (self.contrast, 100), (self.smooth, 0)):
                widget.blockSignals(True)
                widget.setChecked(value) if isinstance(widget, QCheckBox) else widget.setValue(value)
                widget.blockSignals(False)
            self.mode.setCurrentText("Source")
            if depth is not None:
                self.normal_strength.setValue(round(strength * 10))
                self.convention.setCurrentText(convention)
            else:
                self.normal_strength.setValue(200)
                self.convention.setCurrentText("OpenGL")
            self._refresh(fit=True)
            self._update_actions()
            self.statusBar().showMessage(f"{source_path.name} — {source.shape[1]} × {source.shape[0]}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка открытия", str(exc))

    def generate(self) -> None:
        if self.source is None:
            return
        self.worker = GenerateThread(self.source, self)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.generated.connect(self._generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def generate_ai_normal(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        self.worker = GenerateNormalThread(self.source, self.ai_fov.value(), self)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.generated.connect(self._ai_generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def _ai_generated(self, vectors: np.ndarray) -> None:
        if vectors.shape != (*self.source.shape[:2], 3):
            self._generation_failed("DSINE вернула неверный размер Normal")
            return
        self.ai_vectors = vectors
        self._preview_normal = None
        self._refresh()
        self.statusBar().showMessage("AI Normal готова")

    def _generated(self, depth: np.ndarray) -> None:
        self.editor = DepthEditor(depth, self.source[..., 3])
        self._preview_normal = None
        self.mode.setCurrentText("Depth")
        self._settings_changed()
        self.statusBar().showMessage("Depth готова. При необходимости поправьте кистью.")

    def _generation_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка генерации", message)
        self.statusBar().showMessage("Генерация не удалась")

    def _generation_finished(self) -> None:
        self.worker.deleteLater()
        self.worker = None
        self._update_actions()

    def _settings_changed(self) -> None:
        if self._building or self.editor is None:
            return
        settings = DepthSettings(self.invert.isChecked(), self.depth_strength.value() / 100,
                                 self.contrast.value() / 100, self.smooth.value() / 10)
        self.editor.set_settings(settings)
        self._preview_normal = None
        self._refresh()

    def _normal_changed(self) -> None:
        if not self._building:
            self._preview_normal = None
            self._refresh()

    def _normal_source_changed(self, source: str) -> None:
        self._normal_changed()
        if source in ("AI", "Hybrid") and self.ai_vectors is None and self.source is not None:
            self.generate_ai_normal()

    def _fov_changed(self) -> None:
        if self._building:
            return
        self.ai_vectors = None
        self._preview_normal = None
        if self.normal_source.currentText() in ("AI", "Hybrid"):
            self.generate_ai_normal()

    def _selected_normal(self, convention: str) -> np.ndarray:
        alpha = self.source[..., 3]
        source = self.normal_source.currentText()
        ai_vectors = self.ai_vectors
        if ai_vectors is not None:
            ai_vectors = orient_ai_vectors(ai_vectors, self.invert_ai_x.isChecked(),
                                           self.invert_ai_y.isChecked())
        if source == "AI":
            if ai_vectors is None:
                raise RuntimeError("Сначала сгенерируйте AI Normal")
            return ai_normal(ai_vectors, alpha, convention)
        if self.editor is None:
            raise RuntimeError("Сначала сгенерируйте Depth")
        depth_normal = generate_normal(self.editor.depth, self.source,
                                       self.normal_strength.value() / 10, convention)
        if source == "Depth":
            return depth_normal
        if ai_vectors is None:
            raise RuntimeError("Сначала сгенерируйте AI Normal")
        return hybrid_normal(depth_normal, ai_vectors, alpha,
                             self.ai_influence.value(), convention)

    def _mode_changed(self, mode: str) -> None:
        self.view.mode = mode
        self._refresh()

    def _tool_changed(self, tool: str) -> None:
        self.view.tool = tool
        self.view.setDragMode(ImageView.ScrollHandDrag if tool == "Pan" else ImageView.NoDrag)

    def _refresh(self, fit: bool = False) -> None:
        if self.source is None:
            self.view.clear_image()
            return
        mode = self.mode.currentText()
        if mode == "Source" or (self.editor is None and self.normal_source.currentText() != "AI"):
            pixels = self.source
        elif mode == "Depth":
            if self.editor is None:
                pixels = self.source
                self.view.set_pixels(pixels, fit)
                self._update_actions()
                return
            grey = np.rint(self.editor.depth * 255).astype(np.uint8)
            pixels = np.empty_like(self.source)
            pixels[..., :3] = grey[..., None]
            pixels[..., 3] = self.source[..., 3]
        else:
            if self.normal_source.currentText() in ("AI", "Hybrid") and self.ai_vectors is None:
                pixels = self.source
            elif mode == "Normal":
                pixels = self._selected_normal(self.convention.currentText())
            else:
                if self._preview_normal is None:
                    self._preview_normal = self._selected_normal("OpenGL")
                pixels = render_lighting(self.source, self._preview_normal, *self.light)
        self.view.set_pixels(pixels, fit)
        self._update_actions()

    def _brush_event(self, phase: str, x: int, y: int) -> None:
        if self.editor is None or (self.worker is not None and self.worker.isRunning()):
            return
        if phase == "begin":
            if not (0 <= x < self.source.shape[1] and 0 <= y < self.source.shape[0]) or self.source[y, x, 3] == 0:
                return
            self.editor.begin(self.tool.currentText(), self.radius.value(),
                              self.brush_strength.value() / 100, x, y)
        elif phase == "move":
            self.editor.move(x, y)
        else:
            self.editor.end()
        self._preview_normal = None
        self._refresh()

    def _light_changed(self, x: float, y: float) -> None:
        self.light = (max(-1, min(1, x)), max(-1, min(1, y)))
        if self.mode.currentText() == "Lighting Preview":
            self._refresh()

    def undo(self) -> None:
        if self.editor and self.editor.undo():
            self._preview_normal = None
            self._refresh()

    def redo(self) -> None:
        if self.editor and self.editor.redo():
            self._preview_normal = None
            self._refresh()

    def fit_image(self) -> None:
        if self.source is not None:
            self.view.fitInView(self.view.item, Qt.KeepAspectRatio)

    def export(self) -> None:
        if self.editor is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Папка экспорта", str(self.source_path.parent))
        if not directory:
            return
        try:
            normal = self._selected_normal(self.convention.currentText())
            depth_path, normal_path = export_maps(self.source_path, self.editor.depth, normal,
                                                  self.source[..., 3], directory)
            project_path = Path(directory) / f"{self.source_path.stem}.ssoul"
            save_project(project_path, self.source_path, self.editor.depth,
                         self.normal_strength.value() / 10, self.convention.currentText())
            self.statusBar().showMessage(f"Сохранено: {depth_path.name}, {normal_path.name}, {project_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            self.statusBar().showMessage("Дождитесь завершения генерации перед закрытием")
            return
        super().closeEvent(event)
