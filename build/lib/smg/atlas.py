import numpy as np


def tile_positions(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    step = size - overlap
    positions = list(range(0, length - size + 1, step))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def feather_weights(height: int, width: int, overlap: int,
                    top: bool, bottom: bool, left: bool, right: bool) -> np.ndarray:
    y = np.ones(height, np.float32)
    x = np.ones(width, np.float32)
    margin_y = min(overlap, height // 2)
    margin_x = min(overlap, width // 2)
    if top:
        y[:margin_y] = np.linspace(0.01, 1, margin_y)
    if bottom:
        y[-margin_y:] = np.linspace(1, 0.01, margin_y)
    if left:
        x[:margin_x] = np.linspace(0.01, 1, margin_x)
    if right:
        x[-margin_x:] = np.linspace(1, 0.01, margin_x)
    return y[:, None] * x[None, :]
