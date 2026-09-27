import cv2
import numpy as np
from PIL import Image

from smg.editor import DepthEditor
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


def test_brush_undo_redo():
    alpha = np.full((40, 40), 255, np.uint8)
    editor = DepthEditor(np.full((40, 40), 0.5, np.float32), alpha)
    editor.begin("Raise", 5, 0.1, 20, 20)
    editor.move(23, 20)
    editor.end()
    assert editor.depth[20, 20] > 0.5
    assert editor.undo() and np.allclose(editor.depth, 0.5)
    assert editor.redo() and editor.depth[20, 20] > 0.5


def test_normalized_depth_is_float32():
    raw = np.arange(25, dtype=np.float32).reshape(5, 5)
    alpha = np.full((5, 5), 255, np.uint8)
    assert normalize_depth(raw, alpha).dtype == np.float32
