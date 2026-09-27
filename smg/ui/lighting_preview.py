import numpy as np


def render_lighting(source: np.ndarray, normal: np.ndarray,
                    light_x: float, light_y: float) -> np.ndarray:
    direction = np.array((light_x, light_y, 0.8), np.float32)
    direction /= np.linalg.norm(direction)
    vector = normal[..., :3].astype(np.float32) / 127.5 - 1
    # Normals are encoded OpenGL +Y; preview light coordinates use the same convention.
    diffuse = np.maximum(0, np.sum(vector * direction, axis=-1))
    intensity = 0.28 + 0.92 * diffuse
    result = np.empty_like(source)
    result[..., :3] = np.clip(source[..., :3].astype(np.float32) * intensity[..., None], 0, 255).astype(np.uint8)
    result[..., 3] = source[..., 3]
    return result
