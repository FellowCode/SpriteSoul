import numpy as np


def render_lighting(source: np.ndarray, normal: np.ndarray,
                    light_x: float, light_y: float,
                    ao: np.ndarray | None = None,
                    roughness: np.ndarray | None = None) -> np.ndarray:
    """Render diffuse light and optional roughness-controlled Blinn-Phong highlights."""
    direction = np.array((light_x, light_y, 0.8), np.float32)
    direction /= np.linalg.norm(direction)
    vector = normal[..., :3].astype(np.float32) / 127.5 - 1
    # Normals are encoded OpenGL +Y; preview light coordinates use the same convention.
    diffuse = np.maximum(0, np.sum(vector * direction, axis=-1))
    intensity = 0.28 + 0.92 * diffuse
    if ao is not None:
        intensity *= ao
    rgb = source[..., :3].astype(np.float32) * intensity[..., None]
    if roughness is not None:
        roughness = np.clip(roughness, 0, 1).astype(np.float32)
        # The viewer faces the sprite along +Z. Normalize decoded normals to
        # prevent 8-bit quantization from distorting narrow highlights.
        unit_normal = vector / np.maximum(
            np.linalg.norm(vector, axis=-1, keepdims=True), 1e-6,
        )
        halfway = direction + np.array((0, 0, 1), np.float32)
        halfway /= np.linalg.norm(halfway)
        ndoth = np.clip(np.sum(unit_normal * halfway, axis=-1), 0, 1)
        ndotl = np.clip(np.sum(unit_normal * direction, axis=-1), 0, 1)
        # Clamp the minimum roughness so perfectly smooth pixels stay finite.
        shininess = np.maximum(2, 2 / np.maximum(roughness, 0.04) ** 2 - 2)
        strength = 0.04 + 0.46 * (1 - roughness) ** 2
        specular = strength * ndoth ** shininess * ndotl
        specular *= (unit_normal[..., 2] > 0)
        if ao is not None:
            specular *= ao
        rgb += 255 * specular[..., None]
    result = np.empty_like(source)
    result[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    result[..., 3] = source[..., 3]
    return result
