import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def image_from_array(pixels: np.ndarray) -> QImage:
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    if pixels.ndim == 2:
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_Grayscale8)
    else:
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGBA8888)
    return image.copy()


class ImageView(QGraphicsView):
    brush_event = Signal(str, int, int)
    light_changed = Signal(float, float)
    selection_event = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.mode = "Source"
        self.tool = "Pan"
        self.selection_enabled = False
        self._dragging = False
        self._has_image = False
        self.set_dark_theme(False)

    def set_dark_theme(self, dark: bool) -> None:
        """Use a neutral transparency grid that does not glare in dark mode."""
        checker = QPixmap(24, 24)
        checker.fill(QColor("#2a303a" if dark else "#d9dce0"))
        painter = QPainter(checker)
        painter.fillRect(0, 0, 12, 12, QColor("#363e4a" if dark else "#f0f1f3"))
        painter.fillRect(12, 12, 12, 12, QColor("#363e4a" if dark else "#f0f1f3"))
        painter.end()
        self.setBackgroundBrush(QBrush(checker))

    def set_pixels(self, pixels: np.ndarray, fit: bool = False) -> None:
        self.item.setPixmap(QPixmap.fromImage(image_from_array(pixels)))
        self.scene().setSceneRect(self.item.boundingRect())
        if fit or not self._has_image:
            self.fitInView(self.item, Qt.KeepAspectRatio)
        self._has_image = True

    def clear_image(self) -> None:
        self.item.setPixmap(QPixmap())
        self._has_image = False

    def wheelEvent(self, event) -> None:
        if not self._has_image:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def _position(self, event) -> tuple[int, int]:
        point = self.mapToScene(event.position().toPoint())
        return round(point.x()), round(point.y())

    def _emit_light(self, event) -> None:
        x, y = self._position(event)
        rect = self.item.boundingRect()
        if rect.width() and rect.height():
            self.light_changed.emit(2 * x / rect.width() - 1, 1 - 2 * y / rect.height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._has_image:
            if self.selection_enabled:
                self.selection_event.emit(*self._position(event))
                return
            if self.mode == "Depth" and self.tool != "Pan":
                self._dragging = True
                self.brush_event.emit("begin", *self._position(event))
                return
            if self.mode == "Lighting Preview":
                self._dragging = True
                self._emit_light(event)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            if self.mode == "Depth":
                self.brush_event.emit("move", *self._position(event))
            else:
                self._emit_light(event)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            if self.mode == "Depth":
                self.brush_event.emit("end", *self._position(event))
            return
        super().mouseReleaseEvent(event)
