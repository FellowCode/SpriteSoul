import cv2
import json
import numpy as np
import pytest
from PIL import Image

from smg.cli import main
from smg.export import load_project
from smg.pipeline import open_png


def _source(path):
    rgba = np.zeros((8, 11, 4), np.uint8)
    rgba[..., :3] = 100
    rgba[1:7, 2:10, 3] = 173
    Image.fromarray(rgba).save(path)
    return rgba


def test_cli_from_depth_map_preserves_size_alpha_and_project(tmp_path, monkeypatch):
    from smg import normal_ai, setup

    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: None)

    class FakeAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(normal_ai, "DSINENormalModel", FakeAI)
    source = tmp_path / "sprite.png"
    rgba = _source(source)
    values = np.tile(np.linspace(0, 65535, 11, dtype=np.uint16), (8, 1))
    depth_map = tmp_path / "input_depth.png"
    assert cv2.imwrite(str(depth_map), values)
    output = tmp_path / "out"

    assert main([str(source), "-o", str(output), "--depth-map", str(depth_map),
                 "--convention", "directx", "--save-project"]) == 0
    depth = cv2.imread(str(output / "sprite_depth.png"), cv2.IMREAD_UNCHANGED)
    normal = open_png(output / "sprite_normal.png")
    assert depth.shape == rgba.shape and depth.dtype == np.uint16
    assert np.array_equal(depth[..., 0], values)
    assert np.array_equal(depth[..., 3] // 257, rgba[..., 3].astype(np.uint16))
    assert normal.shape == rgba.shape
    assert np.array_equal(normal[..., 3], rgba[..., 3])
    _, saved, strength, convention = load_project(output / "sprite.ssoul")
    assert np.allclose(saved, values.astype(np.float32) / 65535)
    assert strength == 20 and convention == "DirectX"

    assert main([str(output / "sprite.ssoul"), "-o", str(output),
                 "--maps", "normal", "--overwrite"]) == 0
    assert (output / "sprite_normal.png").exists()


def test_cli_uses_generated_directory_by_default(tmp_path, monkeypatch):
    source = tmp_path / "sprite.png"
    _source(source)
    depth_map = tmp_path / "input_depth.png"
    assert cv2.imwrite(str(depth_map), np.full((8, 11), 128, np.uint8))
    monkeypatch.chdir(tmp_path)

    assert main([str(source), "--maps", "depth", "--depth-map", str(depth_map)]) == 0
    assert (tmp_path / "generated" / "sprite_depth.png").exists()


def test_cli_batch_and_existing_output(tmp_path, monkeypatch):
    from smg.depth import inference
    from smg import setup

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: prepared.append(tuple(models)))

    class FakeDepth:
        def generate(self, rgba, progress=None):
            return np.tile(np.arange(rgba.shape[1], dtype=np.float32),
                           (rgba.shape[0], 1))

    monkeypatch.setattr(inference, "DepthModel", FakeDepth)
    one, two = tmp_path / "one.png", tmp_path / "two.png"
    _source(one)
    _source(two)
    output = tmp_path / "out"
    command = [str(one), str(two), "-o", str(output), "--maps", "depth"]
    assert main(command) == 0
    assert prepared == [("depth",), ("depth",)]
    assert (output / "one_depth.png").exists()
    assert (output / "two_depth.png").exists()
    assert not (output / "one_normal.png").exists()
    assert main(command) == 1
    assert main(command + ["--overwrite"]) == 0


def test_cli_ai_only_skips_depth_model(tmp_path, monkeypatch):
    from smg import normal_ai
    from smg import setup
    from smg.depth import inference

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: prepared.append(tuple(models)))

    class NoDepth:
        def __init__(self):
            raise AssertionError("Depth model must not be loaded")

    class FakeAI:
        def __init__(self, fov):
            assert fov == 60

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., :] = (0.6, -0.3, 0.74)
            return vectors

    monkeypatch.setattr(inference, "DepthModel", NoDepth)
    monkeypatch.setattr(normal_ai, "DSINENormalModel", FakeAI)
    source = tmp_path / "sprite.png"
    rgba = _source(source)
    output = tmp_path / "out"
    assert main([str(source), "-o", str(output), "--maps", "normal",
                 "--normal-source", "ai", "--no-invert-ai-x"]) == 0
    assert prepared == [("ai",)]
    normal = open_png(output / "sprite_normal.png")
    assert normal[3, 3, 0] > 128 and normal[3, 3, 1] > 128
    assert np.array_equal(normal[..., 3], rgba[..., 3])
    assert not (output / "sprite_depth.png").exists()


def test_cli_rejects_removed_hybrid_source(tmp_path):
    source = tmp_path / "sprite.png"
    _source(source)
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--maps", "normal", "--normal-source", "hybrid"])
    assert exc.value.code == 2


def test_cli_json_result_and_error_stream(tmp_path, capsys):
    source = tmp_path / "sprite.png"
    _source(source)
    depth_map = tmp_path / "depth.png"
    assert cv2.imwrite(str(depth_map), np.full((8, 11), 128, np.uint8))
    output = tmp_path / "out"
    command = ["generate", str(source), "-o", str(output), "--depth-map", str(depth_map),
               "--maps", "depth", "--progress", "json"]
    assert main(command) == 0
    result = capsys.readouterr()
    events = [json.loads(line) for line in result.out.splitlines()]
    assert result.err == ""
    assert [event["event"] for event in events] == ["result", "complete"]
    assert events[-1]["status"] == "success"
    assert all(event["schema_version"] == 1 for event in events)
    assert main(command) == 1
    errors = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in errors] == ["error", "complete"]
    assert errors[0]["input"] == str(source)
    assert errors[-1]["failed"] == 1


def test_cli_help_lists_commands_and_progress(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "generate" in help_text and "setup" in help_text
    with pytest.raises(SystemExit) as exc:
        main(["setup", "--help"])
    assert exc.value.code == 0
    assert "--progress" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])
    assert exc.value.code == 0
    generate_help = capsys.readouterr().out
    assert "--normal-source" in generate_help
    assert "--ai-smoothing" in generate_help
    assert "--ai-details" in generate_help
    assert "hybrid" not in generate_help.lower()
