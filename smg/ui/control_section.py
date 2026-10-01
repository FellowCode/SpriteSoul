from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class FocusWheelGuard(QObject):
    """Let an unfocused field pass wheel scrolling to the sidebar."""

    def __init__(self, scroll: QScrollArea):
        super().__init__(scroll)
        self.scroll = scroll

    def eventFilter(self, field, event) -> bool:
        if event.type() == QEvent.Wheel and not field.hasFocus():
            QCoreApplication.sendEvent(self.scroll.viewport(), event)
            return True
        return False


class ControlSection(QFrame):
    """A keyboard-accessible, collapsible group of sidebar controls."""

    def __init__(self, title: str, description: str = "", *, expanded: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("controlSection")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QToolButton(self)
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setAccessibleName(title)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setCheckable(True)
        layout.addWidget(self.header)

        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 0, 12, 12)
        self.content_layout.setSpacing(10)
        if description:
            label = QLabel(description)
            label.setObjectName("controlHelp")
            label.setWordWrap(True)
            self.content_layout.addWidget(label)
        layout.addWidget(self.content)

        self.header.toggled.connect(self._toggle)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)
        self._toggle(expanded)

    def _toggle(self, expanded: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setToolTip("Свернуть раздел" if expanded else "Развернуть раздел")
        self.content.setVisible(expanded)
