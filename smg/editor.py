from dataclasses import dataclass

import cv2
import numpy as np

from smg.depth.processing import DepthSettings, process_depth


@dataclass
class Stroke:
    tool: str
    radius: int
    strength: float
    points: list[tuple[int, int]]


class DepthEditor:
    def __init__(self, base: np.ndarray, alpha: np.ndarray):
        self.base = np.asarray(base, np.float32)
        self.alpha = alpha
        self.settings = DepthSettings()
        self.strokes: list[Stroke] = []
        self.redo_strokes: list[Stroke] = []
        self.current: Stroke | None = None
        self.depth = process_depth(self.base, self.alpha, self.settings)

    def set_settings(self, settings: DepthSettings) -> None:
        self.settings = settings
        self.rebuild()

    def rebuild(self) -> None:
        self.depth = process_depth(self.base, self.alpha, self.settings)
        for stroke in self.strokes:
            for x, y in stroke.points:
                self._stamp(stroke, x, y)

    def begin(self, tool: str, radius: int, strength: float, x: int, y: int) -> None:
        self.current = Stroke(tool, radius, strength, [])
        self.redo_strokes.clear()
        self.move(x, y)

    def move(self, x: int, y: int) -> None:
        if self.current is None:
            return
        points = self.current.points
        if points:
            px, py = points[-1]
            steps = max(1, int(np.hypot(x - px, y - py) / max(1, self.current.radius / 4)))
            path = [(round(px + (x - px) * i / steps), round(py + (y - py) * i / steps)) for i in range(1, steps + 1)]
        else:
            path = [(x, y)]
        for point in path:
            if points and point == points[-1]:
                continue
            points.append(point)
            self._stamp(self.current, *point)

    def end(self) -> None:
        if self.current and self.current.points:
            self.strokes.append(self.current)
        self.current = None

    def undo(self) -> bool:
        if not self.strokes:
            return False
        self.redo_strokes.append(self.strokes.pop())
        self.rebuild()
        return True

    def redo(self) -> bool:
        if not self.redo_strokes:
            return False
        self.strokes.append(self.redo_strokes.pop())
        self.rebuild()
        return True

    def _stamp(self, stroke: Stroke, x: int, y: int) -> None:
        h, w = self.depth.shape
        r = stroke.radius
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(r, 1)
        weight = np.maximum(0, 1 - distance ** 2) ** 2
        weight *= (self.alpha[y0:y1, x0:x1] > 0)
        region = self.depth[y0:y1, x0:x1]
        amount = stroke.strength * weight
        if stroke.tool == "Raise":
            region[:] = np.minimum(1, region + amount)
        elif stroke.tool == "Lower":
            region[:] = np.maximum(0, region - amount)
        elif stroke.tool == "Smooth":
            sigma = max(1, r / 3)
            pad = int(np.ceil(3 * sigma))
            bx0, bx1 = max(0, x0 - pad), min(w, x1 + pad)
            by0, by1 = max(0, y0 - pad), min(h, y1 + pad)
            mask = (self.alpha[by0:by1, bx0:bx1] > 0).astype(np.float32)
            area = self.depth[by0:by1, bx0:bx1]
            blurred = cv2.GaussianBlur(area * mask, (0, 0), sigma)
            divisor = cv2.GaussianBlur(mask, (0, 0), sigma)
            target = (blurred / np.maximum(divisor, 1e-6))[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
            region[:] = region * (1 - amount) + target * amount
