import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from smg.cli import main
from smg.export import load_project
from smg.pipeline import open_png


@pytest.fixture
def batch_models(monkeypatch):
    from smg import albedo_ai, model_session, normal_ai, roughness_ai, segmentation, setup
    from smg.depth import inference

    events, settings = [], []
    active = set()

    def load(kind):
        assert not active, f"Models overlap: {active} and {kind}"
        active.add(kind)
        events.append((kind, "load"))

    def unload(kind):
        active.remove(kind)
        events.append((kind, "unload"))

    def infer(kind, rgba):
        assert active == {kind}
        events.append((kind, int(rgba[0, 0, 0])))

    class Depth:
        def __init__(self, keep_loaded):
            assert keep_loaded
            load("depth")

        def generate(self, rgba, progress=None):
            infer("depth", rgba)
            return np.tile(np.arange(rgba.shape[1], dtype=np.float32), (rgba.shape[0], 1))

        def unload(self):
            unload("depth")

    class Crown:
        def __init__(self):
            load("clipseg")

        def predict(self, rgba, progress=None):
            infer("clipseg", rgba)
            return np.full(rgba.shape[:2], 0.9, np.float32)

        def unload(self):
            unload("clipseg")

    class Normal:
        def __init__(self, fov, keep_loaded):
            assert fov == 60 and keep_loaded
            load("normal")

        def generate(self, rgba, progress=None):
            infer("normal", rgba)
            result = np.zeros((*rgba.shape[:2], 3), np.float32)
            result[..., 2] = 1
            return result

        def unload(self):
            unload("normal")

    class Session:
        def __init__(self, python, worker, root):
            self.kind = "albedo" if worker.name == "intrinsic_worker.py" else "roughness"
            load(self.kind)

        def run(self, rgba, progress):
            infer(self.kind, rgba)

        def close(self):
            unload(self.kind)

    def albedo(rgba, progress, session, **kwargs):
        settings.append(kwargs)
        session.run(rgba, progress)
        return rgba.copy()

    def roughness(rgba, progress, session):
        session.run(rgba, progress)
        return np.full(rgba.shape[:2], 0.37, np.float32)

    monkeypatch.setattr(inference, "DepthModel", Depth)
    monkeypatch.setattr(segmentation, "CrownModel", Crown)
    monkeypatch.setattr(normal_ai, "DSINENormalModel", Normal)
    monkeypatch.setattr(albedo_ai, "_experiment", lambda: (Path("intrinsic"), Path("python")))
    monkeypatch.setattr(albedo_ai, "generate_albedo", albedo)
    monkeypatch.setattr(roughness_ai, "generate_roughness", roughness)
    monkeypatch.setattr(model_session, "ModelSession", Session)
    monkeypatch.setattr(setup, "prepare_environment",
                        lambda models, progress, events=None: settings.append(tuple(models)))
    return events, settings, active


def sources(tmp_path):
    directory = tmp_path / "sprites"
    directory.mkdir()
    result = []
    for name, value, shape in (("one", 100, (8, 11)), ("two", 120, (9, 12))):
        rgba = np.full((*shape, 4), value, np.uint8)
        rgba[..., 3] = 0
        rgba[1:-1, 1:-1, 3] = 173
        path = directory / f"{name}.png"
        Image.fromarray(rgba).save(path)
        result.append(path)
    return result


@pytest.mark.parametrize("input_kind", ["files", "directory", "glob"])
def test_batch_all_maps_reuses_models_in_order(tmp_path, batch_models, capsys, input_kind):
    paths = sources(tmp_path)
    inputs = paths if input_kind == "files" else [paths[0].parent if input_kind == "directory"
                                                else paths[0].parent / "*.png"]
    output = tmp_path / "out"
    debug = tmp_path / "debug"
    assert main(["generate", *map(str, inputs), "--maps", "all", "--save-project",
                 "--albedo-debug", str(debug), "--progress", "json", "-o", str(output)]) == 0
    events, settings, active = batch_models
    assert not active
    assert events == [(kind, item) for kind in ("depth", "clipseg", "normal", "albedo", "roughness")
                      for item in ("load", 100, 120, "unload")]
    assert settings[0] == ("depth", "ai", "clipseg", "albedo", "roughness")
    assert [item["debug_dir"] for item in settings[1:]] == [debug / "one", debug / "two"]
    for path in paths:
        original = open_png(path)
        for kind in ("depth", "normal", "albedo", "ao", "roughness"):
            generated = open_png(output / f"{path.stem}_{kind}.png")
            assert generated.shape == original.shape
            assert np.array_equal(generated[..., 3], original[..., 3])
        _, depth, _, _, foliage = load_project(output / f"{path.stem}.ssoul", with_foliage_mask=True)
        assert depth.shape == original.shape[:2]
        assert np.array_equal(foliage, original[..., 3] > 0)
    captured = capsys.readouterr()
    assert captured.err == ""
    messages = [json.loads(line) for line in captured.out.splitlines()]
    results = [item for item in messages if item["event"] == "result"]
    assert len(results) == 12
    assert {item["input"] for item in results} == set(map(str, paths))
    assert messages[-1] == {"schema_version": 1, "event": "complete", "status": "success",
                            "processed": 2, "failed": 0}


def test_batch_failure_skips_later_stages_for_one_sprite(tmp_path, batch_models, monkeypatch, capsys):
    from smg import normal_ai

    paths = sources(tmp_path)
    normal = normal_ai.DSINENormalModel.generate

    def fail_first(self, rgba, progress=None):
        if rgba[0, 0, 0] == 100:
            raise RuntimeError("broken sprite")
        return normal(self, rgba, progress)

    monkeypatch.setattr(normal_ai.DSINENormalModel, "generate", fail_first)
    output = tmp_path / "out"
    assert main([*map(str, paths), "--maps", "all", "-o", str(output), "--progress", "json"]) == 1
    events, _, active = batch_models
    assert not active
    assert ("albedo", 100) not in events and ("roughness", 100) not in events
    assert (output / "one_depth.png").exists()
    assert not (output / "one_normal.png").exists()
    assert (output / "two_roughness.png").exists()
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    error, = [item for item in messages if item["event"] == "error"]
    assert error["input"] == str(paths[0]) and error["message"] == "broken sprite"
    assert messages[-1]["processed"] == messages[-1]["failed"] == 1


def test_batch_interrupt_unloads_active_model(tmp_path, batch_models, monkeypatch, capsys):
    from smg import normal_ai

    paths = sources(tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(normal_ai.DSINENormalModel, "generate", interrupt)
    assert main([*map(str, paths), "--maps", "both", "-o", str(tmp_path / "out"),
                 "--progress", "json"]) == 130
    events, _, active = batch_models
    assert not active and events[-1] == ("normal", "unload")
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]["status"] == "failed"


def test_batch_preflight_rejects_duplicate_even_with_overwrite(tmp_path, batch_models, capsys):
    paths = sources(tmp_path)
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / "one.png"
    Image.open(paths[0]).save(duplicate)
    assert main([str(paths[0]), str(duplicate), str(paths[1]), "--maps", "depth", "--overwrite",
                 "-o", str(tmp_path / "out")]) == 1
    assert batch_models[0] == [("depth", item) for item in ("load", 100, 120, "unload")]
    assert "Повторяющееся имя" in capsys.readouterr().err


def test_batch_invalid_inputs_do_not_stop_valid_sprites(tmp_path, batch_models, capsys):
    paths = sources(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty), str(tmp_path / "missing*.png"), *map(str, paths), "--maps", "depth",
                 "-o", str(tmp_path / "out"), "--progress", "json"]) == 1
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]["processed"] == messages[-1]["failed"] == 2


def test_batch_projects_reuse_depth_and_crown(tmp_path, batch_models):
    from smg.export import save_project

    paths = sources(tmp_path)
    projects = []
    for path in paths:
        rgba = open_png(path)
        project = path.with_suffix(".ssoul")
        save_project(project, path, np.full(rgba.shape[:2], 0.7, np.float32), 20, "OpenGL",
                     foliage_mask=rgba[..., 3] > 0)
        projects.append(project)
    assert main([*map(str, projects), "--maps", "both", "--save-project", "--invert-depth",
                 "-o", str(tmp_path / "out")]) == 0
    assert batch_models[0] == [("normal", item) for item in ("load", 100, 120, "unload")]
    assert batch_models[1] == [("ai",)]
    for project in projects:
        _, depth, _, _ = load_project(tmp_path / "out" / project.name)
        assert np.allclose(depth, 0.3)


def test_batch_existing_output_does_not_run_models_for_that_input(tmp_path, batch_models, capsys):
    paths = sources(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "one_depth.png"
    existing.write_bytes(b"preserve existing output")
    assert main([*map(str, paths), "--maps", "depth", "-o", str(output)]) == 1
    assert existing.read_bytes() == b"preserve existing output"
    assert (output / "two_depth.png").is_file()
    assert batch_models[0] == [("depth", item) for item in ("load", 120, "unload")]
    assert "--overwrite" in capsys.readouterr().err


def test_batch_setup_failure_reports_each_input_and_skips_inference(tmp_path, batch_models, monkeypatch, capsys):
    from smg import setup

    paths = sources(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(setup, "prepare_environment", fail)
    output = tmp_path / "out"
    assert main([*map(str, paths), "--maps", "all", "-o", str(output), "--progress", "json"]) == 1
    assert not output.exists() and not batch_models[0]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["input"] for item in events if item["event"] == "error"] == list(map(str, paths))
    assert events[-1]["failed"] == 2 and events[-1]["processed"] == 0


def test_directory_expands_before_depth_map_validation(tmp_path):
    paths = sources(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([str(paths[0].parent), "--maps", "depth", "--depth-map", str(tmp_path / "depth.png")])
    assert exc.value.code == 2


def test_existing_input_with_glob_characters_is_literal(tmp_path):
    import cv2

    source = tmp_path / "sprite[1].png"
    Image.fromarray(np.full((3, 5, 4), 173, np.uint8)).save(source)
    depth = tmp_path / "depth.png"
    assert cv2.imwrite(str(depth), np.full((3, 5), 128, np.uint8))
    output = tmp_path / "out"
    assert main([str(source), "--maps", "depth", "--depth-map", str(depth), "-o", str(output)]) == 0
    assert (output / "sprite[1]_depth.png").exists()
