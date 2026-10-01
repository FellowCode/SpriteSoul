"""Show each AI preparation stage while the setup worker runs."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QStyle,
    QVBoxLayout,
)


class SetupStep(QFrame):
    def __init__(self, number: int, title: str, parent=None):
        super().__init__(parent)
        self.number = number
        self.title = title
        self.state = "waiting"
        self.completed_files = 0
        self.total_files = 0
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.marker = QLabel(str(number))
        self.marker.setFixedWidth(22)
        self.marker.setAlignment(Qt.AlignCenter)
        header.addWidget(self.marker)
        title_label = QLabel(title)
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        header.addWidget(title_label, 1)
        self.status = QLabel("Ожидание")
        header.addWidget(self.status)
        layout.addLayout(header)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName(title)
        layout.addWidget(self.progress)
        self.detail = QLabel("Ожидает своей очереди")
        self.detail.setTextFormat(Qt.PlainText)
        layout.addWidget(self.detail)

    def set_detail(self, message: str) -> None:
        self.detail.setToolTip(message)
        self.detail.setText(self.detail.fontMetrics().elidedText(message, Qt.ElideMiddle, 560))

    def set_state(self, state: str) -> None:
        self.state = state
        marker, status = {
            "waiting": (str(self.number), "Ожидание"),
            "running": ("…", "Выполняется"),
            "done": ("✓", "Готово"),
            "failed": ("!", "Ошибка"),
            "skipped": ("—", "Не выполнено"),
        }[state]
        self.marker.setText(marker)
        if state in {"done", "failed"}:
            icon = QStyle.SP_DialogApplyButton if state == "done" else QStyle.SP_MessageBoxCritical
            self.marker.setPixmap(self.style().standardIcon(icon).pixmap(16, 16))
        self.status.setText(status)
        if state in {"done", "failed", "skipped"}:
            value = self.progress.value() if self.progress.maximum() else 0
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if state == "done" else max(0, value))
            self.progress.setFormat("%p%" if state == "done" else status)

    def update_progress(self, event: dict) -> None:
        self.set_state("running")
        self.set_detail(event.get("message", "Подготовка…"))
        current, total = event.get("current"), event.get("total")
        if event.get("unit") == "files" and isinstance(total, int) and total > 0:
            self.completed_files = current or 0
            self.total_files = total
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            fraction = min(current / total, 1)
            if event.get("unit") == "bytes" and self.total_files:
                fraction = (self.completed_files + fraction) / self.total_files
            self.progress.setRange(0, 100)
            self.progress.setValue(min(99, int(fraction * 100)))
            self.progress.setFormat("%p%")
            if event.get("unit") == "files":
                self.progress.setFormat(f"%p% · {self.completed_files} из {self.total_files} файлов")
        elif not self.total_files:
            self.progress.setRange(0, 0)


class SetupProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Подготовка AI")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(640)
        self._running = True
        self._result_shown = False
        self._active_step: SetupStep | None = None
        layout = QVBoxLayout(self)
        self.summary = QLabel("Проверяем окружение и подготавливаем AI-модели…")
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.overall = QProgressBar()
        self.steps = {}
        for number, (key, title) in enumerate((
            ("cuda", "CUDA и PyTorch"),
            ("depth", "Depth Anything V2"),
            ("geffnet", "Библиотека DSINE (geffnet)"),
            ("ai", "DSINE: исходный код и веса"),
            ("clipseg", "CLIPSeg: определение кроны"),
            ("roughness", "SuperMat: шероховатость"),
        ), 1):
            step = SetupStep(number, title, self)
            self.steps[key] = step
            layout.addWidget(step)
        self.overall.setRange(0, len(self.steps))
        self.overall.setValue(0)
        self.overall.setFormat("Завершено этапов: %v из %m")
        layout.addWidget(self.overall)
        self.error = QLabel()
        self.error.setTextFormat(Qt.PlainText)
        self.error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.close_button = QPushButton("Закрыть")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def update_progress(self, event: dict) -> None:
        if self._result_shown:
            return
        phase = event.get("phase")
        if phase == "cuda":
            key = "cuda"
        elif phase == "pip_install":
            key = {"torch": "cuda", "geffnet": "geffnet",
                   "supermat-runtime": "roughness"}.get(event.get("component"))
        elif phase == "model_download":
            key = event.get("model")
        else:
            return
        step = self.steps.get(key)
        if step is None:
            return
        self._active_step = step
        step.update_progress(event)
        status = event.get("status")
        completed = (
            (phase == "cuda" and status == "ready")
            or (key == "geffnet" and status in {"done", "cached"})
            or (phase == "model_download" and status in {"done", "cached"}
                and event.get("unit") == "files" and step.total_files > 0
                and step.completed_files >= step.total_files)
        )
        if completed:
            step.set_state("done")
        self.overall.setValue(sum(item.state == "done" for item in self.steps.values()))
        self.summary.setText(f"Этап {step.number} из {len(self.steps)}: {step.title}")

    def show_success(self) -> None:
        self._result_shown = True
        for step in self.steps.values():
            step.set_state("done")
        self.overall.setValue(len(self.steps))
        self.summary.setText("Подготовка завершена. CUDA и все модели готовы к работе.")

    def show_failure(self, message: str) -> None:
        self._result_shown = True
        step = self._active_step
        if step is None or step.state == "done":
            step = next((item for item in self.steps.values() if item.state == "waiting"), None)
        if step is not None:
            step.set_state("failed")
            step.set_detail(message)
        for item in self.steps.values():
            if item.state == "waiting":
                item.set_state("skipped")
                item.set_detail("Подготовка остановлена из-за ошибки")
        self.summary.setText("Подготовка AI не удалась.")
        self.error.setText(message)
        self.error.show()

    def finish(self) -> None:
        if not self._result_shown:
            self.show_failure("Подготовка прервана до завершения всех этапов.")
        self._running = False
        self.close_button.setEnabled(True)
        self.close_button.setFocus()

    def done(self, result: int) -> None:
        if not self._running:
            super().done(result)

    def closeEvent(self, event) -> None:
        if self._running:
            event.ignore()
        else:
            super().closeEvent(event)
