from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QThread, Signal, Qt
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QProgressBar, QSlider,
    QScrollArea, QSizePolicy, QSpinBox, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from smg.depth.inference import DepthModel
from smg.ao import ao_from_depth
from smg.albedo_ai import generate_albedo
from smg.roughness_ai import generate_roughness
from smg.crown_normal import compose_crown_normals
from smg.depth.processing import DepthSettings, process_depth
from smg.export import export_albedo, export_ao, export_depth, export_normal, export_roughness, load_project, save_project
from smg.normal import ai_normal, orient_ai_vectors, postprocess_ai_vectors
from smg.normal_ai import DSINENormalModel
from smg.pipeline import generate_depth, open_png
from smg.segmentation import crown_mask, detect_tree_crown, expand_crown_mask, predict_crown_scores
from smg.setup import MODEL_LABELS, missing_models, prepare_environment
from smg.ui.control_section import ControlSection, FocusWheelGuard
from smg.ui.image_view import ImageView
from smg.ui.lighting_preview import render_lighting
from smg.ui.setup_dialog import SetupProgressDialog


class GenerateThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()

    def run(self) -> None:
        try:
            prepare_environment(("depth",), self.progress.emit, events=self.progress_event.emit)
            depth = generate_depth(self.rgba, DepthModel(), self.progress.emit)
            self.generated.emit(depth)
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateNormalThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, fov: float,
                 foliage_mask: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()
        self.fov = fov
        self.foliage_mask = None if foliage_mask is None else foliage_mask.copy()

    def run(self) -> None:
        try:
            models = ("ai", "clipseg") if self.foliage_mask is None else ("ai",)
            prepare_environment(models, self.progress.emit, events=self.progress_event.emit)
            crown = self.foliage_mask
            if crown is None:
                crown = detect_tree_crown(self.rgba, self.progress.emit)
            normal = DSINENormalModel(self.fov).generate(self.rgba, self.progress.emit)
            self.generated.emit((normal, crown))
        except Exception as exc:
            self.failed.emit(str(exc))


class SetupThread(QThread):
    prepared = Signal()
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def run(self) -> None:
        try:
            prepare_environment(
                ("depth", "ai", "clipseg", "roughness"), self.progress.emit, events=self.progress_event.emit
            )
            self.prepared.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class CrownDetectionThread(QThread):
    """Run one CLIPSeg inference without blocking the interface."""

    scores_generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()

    def run(self) -> None:
        try:
            prepare_environment(("clipseg",), self.progress.emit, events=self.progress_event.emit)
            self.scores_generated.emit(predict_crown_scores(self.rgba, self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateAlbedoThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()

    def run(self) -> None:
        try:
            prepare_environment(
                ("albedo",), self.progress.emit, install_cuda=False,
                events=self.progress_event.emit,
            )
            self.generated.emit(generate_albedo(self.rgba, self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateRoughnessThread(QThread):
    generated = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()

    def run(self) -> None:
        try:
            prepare_environment(("roughness",), self.progress.emit, events=self.progress_event.emit)
            self.generated.emit(generate_roughness(self.rgba, self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class GenerateAllThread(QThread):
    depth_generated = Signal(object)
    normal_generated = Signal(object)
    albedo_generated = Signal(object)
    roughness_generated = Signal(object)
    completed = Signal()
    failed = Signal(str)
    progress = Signal(str)
    progress_event = Signal(object)

    def __init__(self, rgba: np.ndarray, fov: float,
                 foliage_mask: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.rgba = rgba.copy()
        self.fov = fov
        self.foliage_mask = None if foliage_mask is None else foliage_mask.copy()

    def run(self) -> None:
        try:
            models = ("depth", "ai", "albedo", "roughness")
            if self.foliage_mask is None:
                models += ("clipseg",)
            prepare_environment(
                models, self.progress.emit,
                events=self.progress_event.emit,
            )
            self.progress.emit("Генерация карты глубины...")
            self.depth_generated.emit(
                generate_depth(self.rgba, DepthModel(), self.progress.emit)
            )
            self.progress.emit("Генерация карты нормалей...")
            crown = self.foliage_mask
            if crown is None:
                crown = detect_tree_crown(self.rgba, self.progress.emit)
            self.normal_generated.emit(
                (DSINENormalModel(self.fov).generate(self.rgba, self.progress.emit), crown)
            )
            self.progress.emit("Генерация Albedo...")
            self.albedo_generated.emit(generate_albedo(self.rgba, self.progress.emit))
            self.progress.emit("Генерация карты шероховатости...")
            self.roughness_generated.emit(generate_roughness(self.rgba, self.progress.emit))
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    _settings = QSettings("SpriteSoul", "SpriteSoul")
    _last_open_directory_key = "files/last_open_directory"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sprite Soul")
        icon_path = Path(__file__).resolve().parent / "icons" / "sprite-soul.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1180, 760)
        self.source_path: Path | None = None
        self.source: np.ndarray | None = None
        self.base_depth: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self._preview_ao: np.ndarray | None = None
        self._show_ao_after_generation = False
        self._preview_normal: np.ndarray | None = None
        self.ai_vectors: np.ndarray | None = None  # OpenGL float XYZ, converted at inference time
        self.albedo: np.ndarray | None = None
        self.roughness: np.ndarray | None = None
        self.worker: QThread | None = None
        self.setup_dialog: SetupProgressDialog | None = None
        self.foliage_mask: np.ndarray | None = None
        self._selection_scores: np.ndarray | None = None
        self._selection_mask: np.ndarray | None = None
        self._selection_active = False
        self._selection_pending = False
        self._mode_before_selection = "Source"
        self.light = (0.35, 0.45)
        self._building = True
        self._build_ui()
        self._building = False
        self._theme_hints = QGuiApplication.styleHints()
        self._theme_hints.colorSchemeChanged.connect(self._apply_system_theme)
        self._apply_system_theme(self._theme_hints.colorScheme())
        self._update_actions()
        self.setStatusBar(QStatusBar())
        self.download_progress = QProgressBar(self)
        self.download_progress.setFixedWidth(220)
        self.download_progress.setTextVisible(True)
        self.download_progress.hide()
        self.statusBar().addPermanentWidget(self.download_progress)
        self.statusBar().showMessage("Откройте PNG для начала работы")

    def _build_ui(self) -> None:
        toolbar = QToolBar("Действия", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.open_action = self._action("Открыть", "Ctrl+O", self.open_file)
        self.setup_action = self._action("Подготовить AI", "Ctrl+Shift+S", self.prepare_ai)
        self.generate_depth_action = self._action("Карта глубины", "Ctrl+G", self.generate)
        self.generate_ao_action = self._action("Карта AO", "Ctrl+Shift+O", self.generate_ao)
        self.generate_normal_action = self._action(
            "Карта нормалей", "Ctrl+Shift+G", self.generate_ai_normal
        )
        self.generate_albedo_action = self._action(
            "Albedo", "Ctrl+Shift+A", self.generate_albedo
        )
        self.generate_roughness_action = self._action(
            "Карта шероховатости", "Ctrl+Shift+R", self.generate_roughness
        )
        self.generate_all_action = self._action(
            "Все карты", "Ctrl+Alt+G", self.generate_all
        )
        self.select_foliage_action = self._action(
            "Определить крону", "Ctrl+Shift+L", self.start_foliage_selection
        )
        # Compatibility aliases for code that used the previous action names.
        self.generate_action = self.generate_depth_action
        self.generate_ai_action = self.generate_normal_action
        self.generate_menu = self.menuBar().addMenu("Генерировать")
        for action in (
            self.generate_depth_action, self.generate_ao_action, self.generate_normal_action,
            self.generate_albedo_action, self.generate_roughness_action, self.generate_all_action,
        ):
            self.generate_menu.addAction(action)
        self.export_action = self._action("Экспорт", "Ctrl+E", self.export)
        for action in (
            self.open_action, self.setup_action, self.generate_menu.menuAction(),
            self.export_action,
        ):
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
        self.view.light_changed.connect(self._light_changed)
        row.addWidget(self.view, 1)

        panel = QFrame()
        panel.setObjectName("controls")
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row.addWidget(panel)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(16, 16, 16, 14)
        preview_layout.setSpacing(8)
        title = QLabel("Настройки")
        title.setObjectName("panelTitle")
        preview_layout.addWidget(title)
        preview_layout.addWidget(QLabel("Просмотр карты"))
        self.mode = QComboBox()
        self.mode.addItems(("Source", "Albedo", "Depth", "AO", "Normal", "Roughness", "Lighting Preview", "Маска кроны"))
        self.mode.setAccessibleName("Просмотр карты")
        self.mode.currentTextChanged.connect(self._mode_changed)
        preview_layout.addWidget(self.mode)
        layout.addWidget(preview)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("controlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        groups = QWidget()
        groups.setObjectName("controlGroups")
        groups_layout = QVBoxLayout(groups)
        groups_layout.setContentsMargins(12, 0, 12, 12)
        groups_layout.setSpacing(8)
        self.controls_scroll.setWidget(groups)
        layout.addWidget(self.controls_scroll, 1)

        self.depth_section = ControlSection("Глубина · Depth", "Общий рельеф и мягкость карты глубины.")
        groups_layout.addWidget(self.depth_section)
        self.invert = QCheckBox("Инвертировать глубину")
        self.invert.toggled.connect(self._settings_changed)
        self.depth_section.content_layout.addWidget(self.invert)
        form = self._control_form(self.depth_section.content_layout)
        self.depth_strength = self._slider(1, 300, 100, self._settings_changed)
        self.contrast = self._slider(25, 300, 100, self._settings_changed)
        self.smooth = self._slider(0, 80, 0, self._settings_changed)
        self.depth_strength.setAccessibleName("Диапазон глубины")
        self.contrast.setAccessibleName("Контраст глубины")
        self.smooth.setAccessibleName("Сглаживание глубины")
        form.addRow("Диапазон", self._slider_field(self.depth_strength, lambda v: f"{v / 100:.2f}×"))
        form.addRow("Контраст", self._slider_field(self.contrast, lambda v: f"{v / 100:.2f}×"))
        form.addRow("Сглаживание", self._slider_field(self.smooth, lambda v: f"{v / 10:.1f}"))

        self.ao_section = ControlSection("Затенение · AO", "Затемнение углублений на основе Depth.")
        groups_layout.addWidget(self.ao_section)
        ao_form = self._control_form(self.ao_section.content_layout)
        self.ao_radius = QSpinBox()
        self.ao_radius.setRange(1, 128)
        self.ao_radius.setValue(24)
        self.ao_radius.setSuffix(" px")
        self.ao_radius.valueChanged.connect(self._ao_changed)
        ao_form.addRow("Радиус", self.ao_radius)
        self.ao_strength = QDoubleSpinBox()
        self.ao_strength.setRange(0, 4)
        self.ao_strength.setSingleStep(0.1)
        self.ao_strength.setValue(2.0)
        self.ao_strength.valueChanged.connect(self._ao_changed)
        ao_form.addRow("Сила", self.ao_strength)

        self.normal_section = ControlSection("Нормали · Normal", "Сглаживание и мелкий рельеф карты AI (DSINE).")
        groups_layout.addWidget(self.normal_section)
        normal_form = self._control_form(self.normal_section.content_layout)
        self.convention = QComboBox()
        self.convention.addItems(("OpenGL", "DirectX"))
        self.convention.currentTextChanged.connect(self._normal_changed)
        normal_form.addRow("Формат", self.convention)
        self.ai_smoothing = QDoubleSpinBox()
        self.ai_smoothing.setRange(0, 8)
        self.ai_smoothing.setSingleStep(0.25)
        self.ai_smoothing.setDecimals(2)
        self.ai_smoothing.setValue(1.5)
        self.ai_smoothing.valueChanged.connect(self._normal_changed)
        normal_form.addRow("Сглаживание", self.ai_smoothing)
        self.ai_details = QDoubleSpinBox()
        self.ai_details.setRange(0, 1.5)
        self.ai_details.setSingleStep(0.05)
        self.ai_details.setDecimals(2)
        self.ai_details.setValue(0.35)
        self.ai_details.valueChanged.connect(self._normal_changed)
        normal_form.addRow("Детали", self.ai_details)

        self.normal_advanced = ControlSection("Дополнительно")
        self.normal_advanced.setObjectName("advancedSection")
        self.normal_section.content_layout.addWidget(self.normal_advanced)
        advanced_form = self._control_form(self.normal_advanced.content_layout)
        self.ai_fov = QSpinBox()
        self.ai_fov.setRange(20, 120)
        self.ai_fov.setValue(60)
        self.ai_fov.setSuffix("°")
        self.ai_fov.setToolTip("Угол обзора DSINE. Изменение запускает повторную генерацию нормалей.")
        self.ai_fov.valueChanged.connect(self._fov_changed)
        advanced_form.addRow("Угол обзора", self.ai_fov)
        self.invert_ai_x = QCheckBox("Инвертировать X")
        self.invert_ai_x.setChecked(True)
        self.invert_ai_x.toggled.connect(self._normal_changed)
        advanced_form.addRow(self.invert_ai_x)
        self.invert_ai_y = QCheckBox("Инвертировать Y")
        self.invert_ai_y.setChecked(True)
        self.invert_ai_y.toggled.connect(self._normal_changed)
        advanced_form.addRow(self.invert_ai_y)

        self.crown_section = ControlSection("Маска кроны")
        groups_layout.addWidget(self.crown_section)
        self.crown_status = QLabel()
        self.crown_status.setObjectName("controlHelp")
        self.crown_status.setWordWrap(True)
        self.crown_section.content_layout.addWidget(self.crown_status)
        self.selection_start = QPushButton("Определить крону")
        self.selection_start.clicked.connect(self.select_foliage_action.trigger)
        self.crown_section.content_layout.addWidget(self.selection_start)
        self.selection_controls = QWidget()
        selection_layout = QVBoxLayout(self.selection_controls)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(10)
        selection_form = self._control_form(selection_layout)
        self.selection_threshold = QSlider(Qt.Horizontal)
        self.selection_threshold.setRange(0, 100)
        self.selection_threshold.setValue(50)
        self.selection_threshold.setAccessibleName("Порог выделения кроны")
        self.selection_threshold.valueChanged.connect(self._selection_threshold_changed)
        self.selection_threshold_label = QLabel("0.50")
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.selection_threshold)
        threshold_row.addWidget(self.selection_threshold_label)
        selection_form.addRow("Порог", threshold_row)
        self.selection_done = QPushButton("Готово")
        self.selection_done.setObjectName("primaryButton")
        self.selection_done.clicked.connect(self.finish_foliage_selection)
        self.selection_cancel = QPushButton("Отмена")
        self.selection_cancel.clicked.connect(self.cancel_foliage_selection)
        selection_actions = QHBoxLayout()
        selection_actions.addWidget(self.selection_cancel)
        selection_actions.addWidget(self.selection_done)
        selection_layout.addLayout(selection_actions)
        self.crown_section.content_layout.addWidget(self.selection_controls)
        self.selection_controls.hide()
        self._controls_selecting = False

        groups_layout.addStretch()
        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(16, 12, 16, 14)
        layout.addWidget(self.hint)
        self._wheel_guard = FocusWheelGuard(self.controls_scroll)
        for field in groups.findChildren(QWidget):
            if isinstance(field, (QComboBox, QSpinBox, QDoubleSpinBox, QSlider)):
                field.setFocusPolicy(Qt.StrongFocus)
                field.installEventFilter(self._wheel_guard)

    @staticmethod
    def _control_form(layout: QVBoxLayout) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        layout.addLayout(form)
        return form

    @staticmethod
    def _slider_field(slider: QSlider, format_value) -> QWidget:
        field = QWidget()
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        value = QLabel(format_value(slider.value()))
        value.setObjectName("controlValue")
        value.setMinimumWidth(42)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v: value.setText(format_value(v)))
        layout.addWidget(slider, 1)
        layout.addWidget(value)
        return field

    def _apply_system_theme(self, scheme=None) -> None:
        """Keep custom controls in step with Qt's system colour-scheme support."""
        if scheme is None:
            scheme = QGuiApplication.styleHints().colorScheme()
        dark = scheme == Qt.ColorScheme.Dark
        if dark:
            colors = {
                "panel": "#20252e", "panel_border": "#3a4352", "text": "#e7edf5",
                "muted": "#aeb9c8", "field": "#2b313c", "field_border": "#59687b",
                "hover": "#8ea2b8", "focus": "#5aa9e6", "drop_down": "#3b4656",
                "section": "#282e38", "header_hover": "#333c49", "disabled": "#788392",
            }
        else:
            colors = {
                "panel": "#f5f6f8", "panel_border": "#d7dbe1", "text": "#252b34",
                "muted": "#6b7280", "field": "#ffffff", "field_border": "#aeb7c2",
                "hover": "#66788b", "focus": "#3074a8", "drop_down": "#354454",
                "section": "#ffffff", "header_hover": "#edf1f5", "disabled": "#929aa6",
            }
        arrow_icon = (Path(__file__).resolve().parent / "icons" / "chevron-down.svg").as_posix()
        up_arrow_icon = (Path(__file__).resolve().parent / "icons" / "chevron-up.svg").as_posix()
        styles = """
            QFrame#controls { background: %(panel)s; border-left: 1px solid %(panel_border)s; }
            QFrame#controls QLabel, QFrame#controls QCheckBox { color: %(text)s; }
            QFrame#controls QLabel#panelTitle { font-size: 16px; font-weight: 600; }
            QFrame#controls QLabel#hint {
                color: %(muted)s;
                border-top: 1px solid %(panel_border)s;
            }
            QFrame#controls QLabel#controlHelp { color: %(muted)s; }
            QFrame#controls QLabel#controlValue { color: %(muted)s; }
            QFrame#controls QLabel:disabled, QFrame#controls QCheckBox:disabled { color: %(disabled)s; }
            QScrollArea#controlsScroll, QWidget#controlGroups { background: %(panel)s; border: none; }
            QFrame#controls QScrollBar:vertical { background: %(panel)s; width: 10px; margin: 4px 2px; }
            QFrame#controls QScrollBar::handle:vertical {
                background: %(field_border)s;
                border-radius: 3px;
                min-height: 24px;
            }
            QFrame#controls QScrollBar::handle:vertical:hover { background: %(hover)s; }
            QFrame#controls QScrollBar::add-line:vertical, QFrame#controls QScrollBar::sub-line:vertical { height: 0; }
            QFrame#controls QScrollBar::add-page:vertical, QFrame#controls QScrollBar::sub-page:vertical { background: transparent; }
            QFrame#controlSection {
                background: %(section)s;
                border: 1px solid %(panel_border)s;
                border-radius: 7px;
            }
            QFrame#advancedSection {
                background: %(panel)s;
                border: 1px solid %(panel_border)s;
                border-radius: 5px;
            }
            QToolButton#sectionHeader {
                color: %(text)s;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                text-align: left;
                font-weight: 600;
                min-height: 22px;
                padding: 8px 10px;
            }
            QToolButton#sectionHeader:hover { background: %(header_hover)s; }
            QToolButton#sectionHeader:focus { border-color: %(focus)s; }
            QToolButton#sectionHeader:disabled { color: %(disabled)s; }
            QFrame#advancedSection QToolButton#sectionHeader { padding: 5px 8px; font-weight: 400; }
            QFrame#controls QPushButton {
                color: %(text)s;
                background: %(field)s;
                border: 1px solid %(field_border)s;
                border-radius: 4px;
                min-height: 28px;
                padding: 2px 10px;
            }
            QFrame#controls QPushButton:hover { background: %(header_hover)s; border-color: %(hover)s; }
            QFrame#controls QPushButton:focus { border-color: %(focus)s; }
            QFrame#controls QPushButton#primaryButton { background: %(focus)s; color: #ffffff; border-color: %(focus)s; }
            QFrame#controls QPushButton:disabled { color: %(disabled)s; background: %(panel)s; border-color: %(panel_border)s; }
            QFrame#controls QPushButton#primaryButton:disabled { color: %(disabled)s; background: %(panel)s; border-color: %(panel_border)s; }
            QSlider { min-height: 22px; }
            QFrame#controls QSlider::groove:horizontal {
                height: 4px;
                background: %(panel_border)s;
                border-radius: 2px;
            }
            QFrame#controls QSlider::sub-page:horizontal { background: %(focus)s; border-radius: 2px; }
            QFrame#controls QSlider::handle:horizontal {
                background: %(focus)s;
                border: 2px solid %(section)s;
                width: 12px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QFrame#controls QSlider::handle:horizontal:hover { background: %(hover)s; }
            QFrame#controls QSlider::handle:horizontal:disabled,
            QFrame#controls QSlider::sub-page:horizontal:disabled { background: %(disabled)s; }
            QFrame#controls QComboBox, QFrame#controls QSpinBox, QFrame#controls QDoubleSpinBox {
                color: %(text)s;
                background: %(field)s;
                border: 1px solid %(field_border)s;
                border-radius: 3px;
                min-height: 26px;
                padding: 0 7px;
            }
            QFrame#controls QComboBox { padding-right: 29px; }
            QFrame#controls QAbstractSpinBox { padding-right: 24px; }
            QFrame#controls QAbstractSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                background: %(drop_down)s;
                width: 20px;
                height: 14px;
                border-top-right-radius: 3px;
            }
            QFrame#controls QAbstractSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                background: %(drop_down)s;
                width: 20px;
                height: 14px;
                border-bottom-right-radius: 3px;
            }
            QFrame#controls QAbstractSpinBox::up-button:hover,
            QFrame#controls QAbstractSpinBox::down-button:hover { background: %(hover)s; }
            QFrame#controls QAbstractSpinBox::up-button:disabled,
            QFrame#controls QAbstractSpinBox::down-button:disabled { background: %(field_border)s; }
            QFrame#controls QAbstractSpinBox::up-arrow { image: url(CHEVRON_UP_ICON); width: 10px; height: 10px; }
            QFrame#controls QAbstractSpinBox::down-arrow { image: url(CHEVRON_ICON); width: 10px; height: 10px; }
            QFrame#controls QComboBox:hover, QFrame#controls QSpinBox:hover, QFrame#controls QDoubleSpinBox:hover {
                border-color: %(hover)s;
            }
            QFrame#controls QComboBox:focus, QFrame#controls QSpinBox:focus, QFrame#controls QDoubleSpinBox:focus {
                border-color: %(focus)s;
            }
            QFrame#controls QComboBox:disabled, QFrame#controls QSpinBox:disabled, QFrame#controls QDoubleSpinBox:disabled {
                color: %(disabled)s;
                background: %(panel)s;
                border-color: %(panel_border)s;
            }
            QFrame#controls QComboBox::drop-down {
                background: %(drop_down)s;
                border-left: 1px solid %(drop_down)s;
                width: 25px;
            }
            QFrame#controls QComboBox::down-arrow {
                image: url(CHEVRON_ICON);
                width: 14px;
                height: 14px;
            }
        """ % colors
        self.setStyleSheet(styles.replace("CHEVRON_ICON", arrow_icon).replace("CHEVRON_UP_ICON", up_arrow_icon))
        self.view.set_dark_theme(dark)

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
        selecting = self._selection_active
        self.open_action.setEnabled(not busy and not selecting)
        self.setup_action.setEnabled(not busy and not selecting)
        self.generate_depth_action.setEnabled(ready and not busy and not selecting)
        self.generate_ao_action.setEnabled(ready and not busy and not selecting)
        self.generate_normal_action.setEnabled(ready and not busy and not selecting)
        self.generate_albedo_action.setEnabled(ready and not busy and not selecting)
        self.generate_roughness_action.setEnabled(ready and not busy and not selecting)
        self.generate_all_action.setEnabled(ready and not busy and not selecting)
        self.select_foliage_action.setEnabled(ready and not busy and not selecting)
        self.generate_menu.menuAction().setEnabled(ready and not busy and not selecting)
        self.mode.setEnabled(not selecting)
        self.ai_fov.setEnabled(not busy and not selecting)
        self.ai_smoothing.setEnabled(not busy and not selecting)
        self.ai_details.setEnabled(not busy and not selecting)
        self.export_action.setEnabled(
            (self.depth is not None or self.ai_vectors is not None or self.albedo is not None
             or self.roughness is not None or self.foliage_mask is not None) and not busy and not selecting
        )
        controls_enabled = selecting and self._selection_scores is not None and not self._selection_pending
        self.selection_threshold.setEnabled(controls_enabled)
        self.selection_done.setEnabled(controls_enabled and self._selection_mask is not None
                                       and bool(np.any(self._selection_mask)))
        self.selection_cancel.setEnabled(selecting)
        self.selection_start.setEnabled(self.select_foliage_action.isEnabled())
        self.selection_start.setVisible(not selecting)
        self.selection_controls.setVisible(selecting)
        self.depth_section.content.setEnabled(self.depth is not None and not busy and not selecting)
        self.ao_section.content.setEnabled(self.depth is not None and not busy and not selecting)
        self.normal_section.content.setEnabled(ready and not busy and not selecting)
        if selecting != self._controls_selecting:
            self._controls_selecting = selecting
            self._focus_control_sections("Маска кроны" if selecting else self.mode.currentText())
        if selecting:
            self.crown_status.setText(
                "CLIPSeg определяет крону…" if self._selection_pending else
                "Голубым показана крона. Настройте порог и подтвердите выделение."
            )
        else:
            self.crown_status.setText(
                "Маска готова и учитывается в нормалях. Можно определить крону заново."
                if self.foliage_mask is not None else
                "Выделите листву, чтобы придать кроне объём в карте нормалей."
            )
        self._update_hint()

    def _focus_control_sections(self, mode: str) -> None:
        """Bring the settings for the selected preview into view."""
        self.depth_section.set_expanded(mode == "Depth")
        self.ao_section.set_expanded(mode in ("AO", "Lighting Preview"))
        self.normal_section.set_expanded(mode in ("Normal", "Lighting Preview"))
        self.crown_section.set_expanded(mode == "Маска кроны")
        self.controls_scroll.verticalScrollBar().setValue(0)

    def _update_hint(self) -> None:
        if self.source is None:
            text = "Откройте PNG или проект. Выберите карту в меню «Генерировать»."
        elif self._selection_active:
            text = "«Готово» применит маску, «Отмена» сохранит прежнее выделение."
        else:
            mode = self.mode.currentText()
            if mode in ("Depth", "AO") and self.depth is None:
                text = "Сначала создайте карту глубины в меню «Генерировать»."
            elif mode in ("Normal", "Lighting Preview") and self.ai_vectors is None:
                text = "Сначала создайте карту нормалей в меню «Генерировать»."
            elif mode == "Albedo" and self.albedo is None:
                text = "Создайте Albedo в меню «Генерировать». Сейчас показан исходник."
            elif mode == "Roughness":
                text = ("Создайте карту шероховатости в меню «Генерировать». Сейчас показан исходник."
                        if self.roughness is None else
                        "Чёрный — гладкая поверхность, белый — шероховатая. Экспорт сохранит Roughness PNG.")
            elif mode == "Lighting Preview":
                text = "Перетаскивайте мышь по изображению, чтобы переместить свет. Колесо меняет масштаб."
            elif mode == "Маска кроны":
                text = "Определите крону и нажмите «Готово». Экспорт сохранит маску в проекте."
            else:
                text = "Колесо меняет масштаб. Перетаскивание перемещает изображение."
        self.hint.setText(text)

    def open_file(self) -> None:
        last_directory = self._settings.value(self._last_open_directory_key, "", type=str)
        if last_directory and not Path(last_directory).is_dir():
            last_directory = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть PNG или проект", last_directory,
            "Sprite Soul (*.png *.ssoul);;PNG (*.png);;Project (*.ssoul)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".ssoul"):
                source_path, depth, _strength, convention, foliage_mask = load_project(
                    path, with_foliage_mask=True
                )
                source = open_png(source_path)
            else:
                source_path = Path(path)
                source = open_png(path)
                depth = None
                foliage_mask = None
            self.source_path = source_path
            self.source = source
            self.base_depth = depth
            self.depth = process_depth(depth, source[..., 3], DepthSettings()) if depth is not None else None
            self._preview_ao = None
            self._preview_normal = None
            self.ai_vectors = None
            self.albedo = None
            self.roughness = None
            self.foliage_mask = foliage_mask
            self._selection_scores = None
            self._selection_mask = None
            self.ai_smoothing.setValue(1.5)
            self.ai_details.setValue(0.35)
            self.ai_fov.setValue(60)
            for widget, value in ((self.invert, False), (self.depth_strength, 100),
                                  (self.contrast, 100), (self.smooth, 0)):
                widget.blockSignals(True)
                widget.setChecked(value) if isinstance(widget, QCheckBox) else widget.setValue(value)
                widget.blockSignals(False)
            self.mode.setCurrentText("Source")
            if depth is not None:
                self.convention.setCurrentText(convention)
            else:
                self.convention.setCurrentText("OpenGL")
            self._refresh(fit=True)
            self._update_actions()
            self._settings.setValue(self._last_open_directory_key, str(Path(path).parent))
            self.statusBar().showMessage(f"{source_path.name} — {source.shape[1]} × {source.shape[0]}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка открытия", str(exc))

    def start_foliage_selection(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        if not self._confirm_model_download(("clipseg",)):
            return
        self._selection_active = True
        self._selection_pending = True
        self._selection_scores = None
        self._selection_mask = None
        self.selection_threshold.setValue(50)
        self._mode_before_selection = self.mode.currentText()
        self.mode.setCurrentText("Source")
        self.worker = CrownDetectionThread(self.source, self)
        self._connect_worker_progress(self.worker)
        self.worker.scores_generated.connect(self._selection_scores_generated)
        self.worker.failed.connect(self._selection_setup_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self.statusBar().showMessage("CLIPSeg определяет крону...")
        self._refresh()
        self._update_actions()

    def _selection_scores_generated(self, scores: np.ndarray) -> None:
        if not self._selection_active:
            return
        if self.source is None or scores.shape != self.source.shape[:2]:
            self._selection_setup_failed("CLIPSeg вернула карту неверного размера")
            return
        self._selection_scores = np.asarray(scores, dtype=np.float32)
        self._selection_pending = False
        self._selection_threshold_changed(self.selection_threshold.value())
        self.statusBar().showMessage("Настройте порог и нажмите «Готово».")

    def _selection_threshold_changed(self, value: int) -> None:
        self.selection_threshold_label.setText(f"{value / 100:.2f}")
        if self._selection_scores is None or self.source is None or not self._selection_active:
            return
        seed = crown_mask(
            self._selection_scores, self.source[..., 3], value / 100,
        )
        self._selection_mask = expand_crown_mask(seed, self.source)
        self._refresh()
        self._update_actions()

    def _selection_setup_failed(self, message: str) -> None:
        if not self._selection_active:
            return
        self._stop_foliage_selection()
        QMessageBox.critical(self, "Ошибка CLIPSeg", message)
        self.statusBar().showMessage("Определить крону не удалось")

    def finish_foliage_selection(self) -> None:
        if (not self._selection_active or self._selection_mask is None or self._selection_pending
                or not np.any(self._selection_mask)):
            return
        self.foliage_mask = self._selection_mask.copy()
        self._preview_normal = None
        self._stop_foliage_selection()
        if self.ai_vectors is not None:
            self.mode.setCurrentText("Normal")
            self.statusBar().showMessage("Нормаль кроны обновлена. Нажмите «Экспорт», чтобы сохранить карты.")
        else:
            self.statusBar().showMessage("Маска кроны готова. Нажмите «Экспорт», чтобы сохранить проект.")

    def cancel_foliage_selection(self) -> None:
        if not self._selection_active:
            return
        self._stop_foliage_selection()
        self.mode.setCurrentText(self._mode_before_selection)
        self.statusBar().showMessage("Определение кроны отменено")

    def _stop_foliage_selection(self) -> None:
        self._selection_active = False
        self._selection_pending = False
        self._selection_scores = None
        self._selection_mask = None
        self._refresh()
        self._update_actions()

    def generate(self) -> None:
        self._start_depth_generation(False)

    def _start_depth_generation(self, show_ao: bool) -> None:
        if self.source is None:
            return
        if not self._confirm_model_download(("depth",)):
            return
        self._show_ao_after_generation = show_ao
        self.worker = GenerateThread(self.source, self)
        self._connect_worker_progress(self.worker)
        self.worker.generated.connect(self._generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def generate_ao(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        if self.depth is not None:
            self.mode.setCurrentText("AO")
            self._refresh()
            return
        self._start_depth_generation(True)

    def prepare_ai(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.setup_dialog is not None:
            self.setup_dialog.raise_()
            self.setup_dialog.activateWindow()
            return
        self.setup_dialog = SetupProgressDialog(self)
        self.setup_dialog.finished.connect(self._setup_dialog_closed)
        self.worker = SetupThread(self)
        self._connect_worker_progress(self.worker)
        self.worker.progress_event.connect(self.setup_dialog.update_progress)
        self.worker.prepared.connect(self.setup_dialog.show_success)
        self.worker.prepared.connect(lambda: self.statusBar().showMessage("CUDA и модели готовы"))
        self.worker.failed.connect(self._setup_failed)
        self.worker.finished.connect(self.setup_dialog.finish)
        self.worker.finished.connect(self._generation_finished)
        self.setup_dialog.open()
        self.worker.start()
        self._update_actions()

    def _setup_dialog_closed(self) -> None:
        self.setup_dialog.deleteLater()
        self.setup_dialog = None

    def _setup_failed(self, message: str) -> None:
        self.setup_dialog.show_failure(message)
        self.statusBar().showMessage("Подготовка AI не удалась")

    def generate_ai_normal(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        models = ("ai", "clipseg") if self.foliage_mask is None else ("ai",)
        if not self._confirm_model_download(models):
            return
        self.worker = GenerateNormalThread(
            self.source, self.ai_fov.value(), self.foliage_mask, self,
        )
        self._connect_worker_progress(self.worker)
        self.worker.generated.connect(self._ai_generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def generate_albedo(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        if not self._confirm_model_download(("albedo",)):
            return
        self.worker = GenerateAlbedoThread(self.source, self)
        self._connect_worker_progress(self.worker)
        self.worker.generated.connect(self._albedo_generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def generate_roughness(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        if not self._confirm_model_download(("roughness",)):
            return
        self.worker = GenerateRoughnessThread(self.source, self)
        self._connect_worker_progress(self.worker)
        self.worker.generated.connect(self._roughness_generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def generate_all(self) -> None:
        if self.source is None or (self.worker is not None and self.worker.isRunning()):
            return
        models = ("depth", "ai", "albedo", "roughness")
        if self.foliage_mask is None:
            models += ("clipseg",)
        if not self._confirm_model_download(models):
            return
        self.worker = GenerateAllThread(
            self.source, self.ai_fov.value(), self.foliage_mask, self,
        )
        self._connect_worker_progress(self.worker)
        self.worker.depth_generated.connect(self._generated)
        self.worker.normal_generated.connect(self._ai_generated)
        self.worker.albedo_generated.connect(self._albedo_generated)
        self.worker.roughness_generated.connect(self._roughness_generated)
        self.worker.completed.connect(self._all_generated)
        self.worker.failed.connect(self._generation_failed)
        self.worker.finished.connect(self._generation_finished)
        self.worker.start()
        self._update_actions()

    def _all_generated(self) -> None:
        self.statusBar().showMessage("Все карты готовы")

    def _roughness_generated(self, roughness: np.ndarray) -> None:
        if roughness.shape != self.source.shape[:2] or not np.isfinite(roughness).all():
            self._generation_failed("SuperMat вернула неверную карту Roughness")
            return
        self.roughness = np.clip(roughness, 0, 1).astype(np.float32)
        self.mode.setCurrentText("Roughness")
        self._refresh()
        self.statusBar().showMessage("Карта шероховатости готова")

    def _albedo_generated(self, albedo: np.ndarray) -> None:
        self.albedo = albedo
        self.mode.setCurrentText("Albedo")
        self._refresh()
        self.statusBar().showMessage("Full-resolution Albedo готова")

    def _ai_generated(self, result: np.ndarray | tuple[np.ndarray, np.ndarray | None]) -> None:
        vectors, crown = result if isinstance(result, tuple) else (result, None)
        if vectors.shape != (*self.source.shape[:2], 3):
            self._generation_failed("DSINE вернула неверный размер Normal")
            return
        if crown is not None:
            self.foliage_mask = crown.copy()
        self.ai_vectors = vectors
        self._preview_normal = None
        self.mode.setCurrentText("Normal")
        self._refresh()
        self.statusBar().showMessage("AI Normal готова")

    def _generated(self, depth: np.ndarray) -> None:
        self.base_depth = np.asarray(depth, np.float32)
        self.depth = process_depth(self.base_depth, self.source[..., 3], DepthSettings())
        self._preview_ao = None
        self._preview_normal = None
        self.mode.setCurrentText("AO" if self._show_ao_after_generation else "Depth")
        self._show_ao_after_generation = False
        self._settings_changed()
        self.statusBar().showMessage("Depth готова")

    def _generation_failed(self, message: str) -> None:
        self._show_ao_after_generation = False
        QMessageBox.critical(self, "Ошибка генерации", message)
        self.statusBar().showMessage("Генерация не удалась")

    def _confirm_model_download(self, models: tuple[str, ...]) -> bool:
        missing = missing_models(models)
        if not missing:
            return True
        names = "\n".join(f"• {MODEL_LABELS[model]}" for model in missing)
        note = ""
        if "albedo" in missing:
            note = "\n\nВес IntrinsicAnything составляет около 14,4 ГиБ."
        if "roughness" in missing:
            note += "\n\nSuperMat и необходимые компоненты занимают около 4,1 ГиБ."
        answer = QMessageBox.question(
            self,
            "Требуется загрузка модели",
            f"Для этой операции отсутствует модель:\n\n{names}"
            f"{note}\n\nСкачать её в папку models проекта?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self.statusBar().showMessage("Загрузка модели отменена")
            return False
        return True

    def _connect_worker_progress(self, worker: QThread) -> None:
        worker.progress.connect(self.statusBar().showMessage)
        worker.progress_event.connect(self._show_progress_event)

    def _show_progress_event(self, event: dict) -> None:
        if event.get("phase") not in {"model_download", "pip_install"}:
            return
        status = event.get("status")
        current = event.get("current")
        total = event.get("total")
        self.download_progress.show()
        if status == "start":
            self.download_progress.setRange(0, 0)
            self.download_progress.setFormat("Загрузка…")
        elif isinstance(current, int) and isinstance(total, int) and total > 0:
            maximum = 1000 if event.get("unit") == "bytes" else total
            self.download_progress.setRange(0, maximum)
            self.download_progress.setValue(min(maximum, int(current / total * maximum)))
            self.download_progress.setFormat("%p%")
        else:
            self.download_progress.setRange(0, 0)
            self.download_progress.setFormat("Загрузка…")
        if status in {"done", "cached", "ready"} and current == total and total is not None:
            self.download_progress.setValue(self.download_progress.maximum())

    def _generation_finished(self) -> None:
        self.download_progress.hide()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.worker.deleteLater()
        self.worker = None
        self._update_actions()

    def _settings_changed(self) -> None:
        if self._building or self.base_depth is None:
            return
        settings = DepthSettings(self.invert.isChecked(), self.depth_strength.value() / 100,
                                 self.contrast.value() / 100, self.smooth.value() / 10)
        self.depth = process_depth(self.base_depth, self.source[..., 3], settings)
        self._preview_ao = None
        self._preview_normal = None
        self._refresh()

    def _normal_changed(self) -> None:
        if not self._building:
            self._preview_normal = None
            self._refresh()

    def _ao_changed(self) -> None:
        if not self._building:
            self._preview_ao = None
            if self.mode.currentText() in ("AO", "Lighting Preview"):
                self._refresh()

    def _selected_ao(self) -> np.ndarray:
        if self._preview_ao is None:
            self._preview_ao = ao_from_depth(
                self.depth, self.source[..., 3],
                self.ao_radius.value(), self.ao_strength.value(),
            )
        return self._preview_ao

    def _fov_changed(self) -> None:
        if self._building:
            return
        self.ai_vectors = None
        self._preview_normal = None
        if self.source is not None:
            self.generate_ai_normal()

    def _selected_normal(self, convention: str) -> np.ndarray:
        alpha = self.source[..., 3]
        if self.ai_vectors is None:
            raise RuntimeError("Сначала сгенерируйте AI Normal")
        ai_vectors = orient_ai_vectors(
            self.ai_vectors, self.invert_ai_x.isChecked(), self.invert_ai_y.isChecked()
        )
        ai_vectors = postprocess_ai_vectors(
            ai_vectors, self.source, self.ai_smoothing.value(),
            self.ai_details.value(),
        )
        if self.foliage_mask is not None:
            ai_vectors = compose_crown_normals(
                ai_vectors, self.foliage_mask, alpha,
            )
        return ai_normal(ai_vectors, alpha, convention)

    def _mode_changed(self, mode: str) -> None:
        self.view.mode = mode
        self._focus_control_sections(mode)
        self._update_hint()
        self._refresh()

    def _refresh(self, fit: bool = False) -> None:
        if self.source is None:
            self.view.clear_image()
            return
        mode = self.mode.currentText()
        if mode == "Albedo":
            pixels = self.albedo if self.albedo is not None else self.source
        elif mode == "Roughness":
            pixels = self.source
            if self.roughness is not None:
                grey = np.rint(self.roughness * 255).astype(np.uint8)
                pixels = self.source.copy()
                pixels[..., :3] = grey[..., None]
        elif mode == "Source":
            pixels = self._foliage_overlay(self.source, self._selection_mask) if (
                self._selection_active and self._selection_mask is not None
            ) else self.source
        elif mode == "Маска кроны":
            pixels = self._foliage_preview()
        elif mode in ("Depth", "AO"):
            if self.depth is None:
                pixels = self.source
                self.view.set_pixels(pixels, fit)
                self._update_actions()
                return
            grey = np.rint((self.depth if mode == "Depth" else self._selected_ao()) * 255).astype(np.uint8)
            pixels = np.empty_like(self.source)
            pixels[..., :3] = grey[..., None]
            pixels[..., 3] = self.source[..., 3]
        else:
            if self.ai_vectors is None:
                pixels = self.source
            elif mode == "Normal":
                pixels = self._selected_normal(self.convention.currentText())
            else:
                if self._preview_normal is None:
                    self._preview_normal = self._selected_normal("OpenGL")
                lighting_base = self.albedo if self.albedo is not None else self.source
                ao = self._selected_ao() if self.depth is not None else None
                pixels = render_lighting(
                    lighting_base, self._preview_normal, *self.light,
                    ao=ao, roughness=self.roughness,
                )
        self.view.set_pixels(pixels, fit)
        self._update_actions()

    def _foliage_overlay(self, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        preview = source.copy()
        # Cyan has clear contrast with the common green foliage palette.
        rgb = preview[..., :3]
        rgb[mask] = np.rint(
            rgb[mask].astype(np.float32) * 0.55
            + np.array((0, 220, 255), np.float32) * 0.45
        ).astype(np.uint8)
        return preview

    def _foliage_preview(self) -> np.ndarray:
        if self.foliage_mask is None:
            return self.source
        pixels = np.zeros_like(self.source)
        pixels[..., :3] = np.where(self.foliage_mask[..., None], 255, 0).astype(np.uint8)
        pixels[..., 3] = self.source[..., 3]
        return pixels

    def _light_changed(self, x: float, y: float) -> None:
        self.light = (max(-1, min(1, x)), max(-1, min(1, y)))
        if self.mode.currentText() == "Lighting Preview":
            self._refresh()

    def fit_image(self) -> None:
        if self.source is not None:
            self.view.fitInView(self.view.item, Qt.KeepAspectRatio)

    def export(self) -> None:
        if (self.depth is None and self.ai_vectors is None and self.albedo is None
                and self.roughness is None and self.foliage_mask is None):
            return
        default_directory = Path.cwd() / "generated"
        default_directory.mkdir(parents=True, exist_ok=True)
        directory = QFileDialog.getExistingDirectory(self, "Папка экспорта", str(default_directory))
        if not directory:
            return
        try:
            saved = []
            if self.depth is not None:
                depth_path = export_depth(
                    self.source_path, self.depth, self.source[..., 3], directory
                )
                ao_path = export_ao(
                    self.source_path, self._selected_ao(), self.source[..., 3], directory
                )
                project_path = Path(directory) / f"{self.source_path.stem}.ssoul"
                save_project(project_path, self.source_path, self.depth,
                             20.0, self.convention.currentText(), self.foliage_mask)
                saved.extend((depth_path.name, ao_path.name, project_path.name))
            elif self.foliage_mask is not None:
                project_path = Path(directory) / f"{self.source_path.stem}.ssoul"
                save_project(project_path, self.source_path, None,
                             20.0, self.convention.currentText(), self.foliage_mask)
                saved.append(project_path.name)
            if self.ai_vectors is not None:
                normal = self._selected_normal(self.convention.currentText())
                normal_path = export_normal(self.source_path, normal, directory)
                saved.append(normal_path.name)
            if self.albedo is not None:
                albedo_path = export_albedo(self.source_path, self.albedo, directory)
                saved.append(albedo_path.name)
            if self.roughness is not None:
                roughness_path = export_roughness(
                    self.source_path, self.roughness, self.source[..., 3], directory,
                )
                saved.append(roughness_path.name)
            self.statusBar().showMessage("Сохранено: " + ", ".join(saved))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            self.statusBar().showMessage("Дождитесь завершения генерации перед закрытием")
            return
        super().closeEvent(event)
