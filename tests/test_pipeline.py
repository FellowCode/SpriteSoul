import cv2
import numpy as np
from PIL import Image

from smg.depth.processing import normalize_depth
from smg.export import export_maps, load_project, save_project
from smg.pipeline import generate_normal, open_png


def test_export_and_project(tmp_path):
    source = np.zeros((12, 17, 4), np.uint8)
    source[..., :3] = 100
    source[2:9, 3:15, 3] = 211
    path = tmp_path / "sprite.png"
    Image.fromarray(source).save(path)
    depth = np.full((12, 17), 0.6, np.float32)
    normal = generate_normal(depth, source)
    depth_path, normal_path = export_maps(path, depth, normal, source[..., 3], tmp_path)
    saved_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    saved_normal = open_png(normal_path)
    assert saved_depth.shape == source.shape
    assert saved_depth.dtype == np.uint16
    assert np.array_equal(saved_depth[..., 3] // 257, source[..., 3].astype(np.uint16))
    assert np.array_equal(saved_normal[..., 3], source[..., 3])
    project = tmp_path / "sprite.ssoul"
    save_project(project, path, depth, 3.5, "DirectX")
    loaded_path, loaded_depth, strength, convention = load_project(project)
    assert loaded_path == path
    assert np.array_equal(loaded_depth, depth)
    assert strength == 3.5 and convention == "DirectX"


def test_normalized_depth_is_float32():
    raw = np.arange(25, dtype=np.float32).reshape(5, 5)
    alpha = np.full((5, 5), 255, np.uint8)
    assert normalize_depth(raw, alpha).dtype == np.float32


def test_project_persists_optional_foliage_mask_and_supports_mask_only(tmp_path):
    source = np.zeros((9, 13, 4), np.uint8)
    source[..., 3] = 255
    path = tmp_path / "tree.png"
    Image.fromarray(source).save(path)
    mask = np.zeros((9, 13), bool)
    mask[2:7, 4:11] = True
    project = tmp_path / "tree.ssoul"
    save_project(project, path, None, 20.0, "OpenGL", foliage_mask=mask)

    # Existing callers retain their original four-item API.
    loaded_path, depth, strength, convention = load_project(project)
    assert loaded_path == path and depth is None
    assert strength == 20.0 and convention == "OpenGL"
    loaded_path, depth, strength, convention, restored = load_project(
        project, with_foliage_mask=True
    )
    assert loaded_path == path and depth is None
    assert strength == 20.0 and convention == "OpenGL"
    assert restored.dtype == bool and np.array_equal(restored, mask)
