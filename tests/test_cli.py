import cv2
import json
import numpy as np
import pytest
from PIL import Image

from smg.cli import main
from smg.export import load_project, save_project
from smg.pipeline import open_png


def _source(path):
    rgba = np.zeros((8, 11, 4), np.uint8)
    rgba[..., :3] = 100
    rgba[1:7, 2:10, 3] = 173
    Image.fromarray(rgba).save(path)
    return rgba


def test_cli_from_depth_map_preserves_size_alpha_and_project(tmp_path, monkeypatch):
    from smg import normal_ai, segmentation, setup

    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: None)

    class FakeAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(normal_ai, "DSINENormalModel", FakeAI)
    monkeypatch.setattr(segmentation, "detect_tree_crown", lambda rgba, progress: None)
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


def test_cli_ao_from_existing_depth_map_skips_model(tmp_path, monkeypatch):
    from smg import setup
    from smg.depth import inference

    monkeypatch.setattr(setup, "prepare_environment", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Model setup is unnecessary")))
    monkeypatch.setattr(inference, "DepthModel", lambda: (_ for _ in ()).throw(AssertionError("Model is unnecessary")))
    source = tmp_path / "sprite.png"
    rgba = _source(source)
    values = np.full(rgba.shape[:2], 220, np.uint8)
    values[3:5, 4:7] = 40
    depth_map = tmp_path / "depth.png"
    assert cv2.imwrite(str(depth_map), values)

    output = tmp_path / "out"
    assert main([str(source), "-o", str(output), "--maps", "ao",
                 "--depth-map", str(depth_map), "--ao-depth-only"]) == 0
    ao = open_png(output / "sprite_ao.png")
    assert ao.shape == rgba.shape
    assert np.array_equal(ao[..., 3], rgba[..., 3])
    assert ao[3, 5, 0] < ao[2, 3, 0]
    assert not (output / "sprite_depth.png").exists()


def test_cli_batch_ao_uses_matching_depth_maps(tmp_path, monkeypatch):
    from smg import setup
    from smg.depth import inference

    monkeypatch.setattr(setup, "prepare_environment", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Depth AI is unnecessary")))
    monkeypatch.setattr(inference, "DepthModel", lambda:
                        (_ for _ in ()).throw(AssertionError("Depth AI is unnecessary")))
    depth_dir = tmp_path / "depth_maps"
    depth_dir.mkdir()
    sources = []
    for name, shape in (("one", (8, 11)), ("two", (9, 12))):
        source = tmp_path / f"{name}.png"
        rgba = np.full((*shape, 4), 100, np.uint8)
        rgba[..., 3] = 255
        Image.fromarray(rgba).save(source)
        depth = np.full(shape, 200, np.uint8)
        depth[3:5, 4:7] = 40
        assert cv2.imwrite(str(depth_dir / f"{name}_depth.png"), depth)
        sources.append(source)

    output = tmp_path / "out"
    assert main([*(str(source) for source in sources), "-o", str(output),
                 "--maps", "ao", "--depth-map", str(depth_dir), "--ao-depth-only"]) == 0
    for name, shape in (("one", (8, 11)), ("two", (9, 12))):
        ao = open_png(output / f"{name}_ao.png")
        assert ao.shape[:2] == shape
        assert ao[3, 5, 0] < ao[2, 3, 0]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("convention", ["opengl", "directx"])
def test_cli_ao_uses_imported_depth_and_normals_without_models(tmp_path, monkeypatch, batch, convention):
    from smg import setup
    from smg.ao import ao_from_depth
    from smg.normal import ai_normal, decode_normals, normalize_vectors

    monkeypatch.setattr(setup, "prepare_environment", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("Imported maps must not load models")))
    directory = tmp_path / "maps"
    directory.mkdir()
    inputs = []
    expected = {}
    for name in (["one", "two"] if batch else ["one"]):
        rgba = np.full((33, 33, 4), (120, 100, 80, 255), np.uint8)
        rgba[0, :, 3] = 0
        rgba[1, :, 3] = 128
        source = tmp_path / f"{name}.png"
        Image.fromarray(rgba).save(source)
        inputs.append(source)
        y, x = np.mgrid[:33, :33].astype(np.float32) - 16
        relief = -3 * np.exp(-(x*x + y*y) / 18)
        gy, gx = np.gradient(relief)
        vectors = normalize_vectors(np.dstack((-gx, gy, np.ones_like(gx))))
        normal = ai_normal(vectors, rgba[..., 3], "OpenGL" if convention == "opengl" else "DirectX")
        Image.fromarray(normal).save(directory / f"{name}_normal.png")
        assert cv2.imwrite(str(directory / f"{name}_depth.png"), np.full((33, 33), 128, np.uint8))
        decoded = decode_normals(normal, "OpenGL" if convention == "opengl" else "DirectX")
        expected[name] = np.rint(ao_from_depth(np.full((33, 33), 128/255, np.float32),
                                             rgba[..., 3], normals=decoded) * 255).astype(np.uint8)
    output = tmp_path / "out"
    assert main([*map(str, inputs), "--maps", "ao", "--depth-map", str(directory),
                 "--normal-map", str(directory), "--convention", convention, "-o", str(output)]) == 0
    assert len(list(output.iterdir())) == len(inputs)
    for source in inputs:
        ao = open_png(output / f"{source.stem}_ao.png")
        assert np.array_equal(ao[..., 3], open_png(source)[..., 3])
        assert np.array_equal(ao[..., 0], expected[source.stem])
        assert ao[16, 16, 0] < 250


def test_cli_rejects_wrong_normal_size_before_loading_models(tmp_path, monkeypatch, capsys):
    from smg import setup

    monkeypatch.setattr(setup, "prepare_environment", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("Invalid map reached model setup")))
    source = tmp_path / "sprite.png"
    _source(source)
    normal = tmp_path / "normal.png"
    Image.fromarray(np.full((4, 5, 4), 255, np.uint8)).save(normal)
    assert main([str(source), "--maps", "ao", "--normal-map", str(normal)]) == 1
    assert "Normal 5x4" in capsys.readouterr().err


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cli_routes_ao_device_in_single_and_batch_exports(tmp_path, monkeypatch, batch, device):
    from smg import ao, setup

    monkeypatch.setattr(setup, "prepare_environment", lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError("Imported maps must skip model setup")))
    captured = []

    def calculate(depth, alpha, *args, **kwargs):
        captured.append(kwargs["device"])
        assert kwargs["normals"] is not None
        return np.full(depth.shape, 0.7, np.float32)

    monkeypatch.setattr(ao, "ao_from_depth", calculate)
    directory = tmp_path / "maps"
    directory.mkdir()
    sources = []
    for name in (["one", "two"] if batch else ["one"]):
        source = tmp_path / f"{name}.png"
        rgba = _source(source)
        sources.append(source)
        assert cv2.imwrite(str(directory / f"{name}_depth.png"), np.full(rgba.shape[:2], 128, np.uint8))
        Image.fromarray(np.full(rgba.shape, (128, 128, 255, 255), np.uint8)).save(directory / f"{name}_normal.png")
    assert main([*map(str, sources), "--maps", "ao", "--depth-map", str(directory),
                 "--normal-map", str(directory), "--ao-device", device, "-o", str(tmp_path / "out")]) == 0
    assert captured == [device] * len(sources)


def test_cli_project_depth_can_be_overridden_by_png(tmp_path, monkeypatch):
    from smg import setup

    monkeypatch.setattr(setup, "prepare_environment", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Depth AI is unnecessary")))
    source = tmp_path / "sprite.png"
    rgba = _source(source)
    project = tmp_path / "sprite.ssoul"
    save_project(project, source, np.full(rgba.shape[:2], 0.75, np.float32),
                 20.0, "OpenGL")
    replacement = np.full(rgba.shape[:2], 64, np.uint8)
    depth_map = tmp_path / "replacement.png"
    assert cv2.imwrite(str(depth_map), replacement)

    output = tmp_path / "out"
    assert main([str(project), "-o", str(output), "--maps", "depth",
                 "--depth-map", str(depth_map)]) == 0
    depth = cv2.imread(str(output / "sprite_depth.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(depth[..., 0], np.full(rgba.shape[:2], 64 * 257, np.uint16))


def test_cli_rejects_depth_map_with_wrong_sprite_size(tmp_path, monkeypatch, capsys):
    from smg import setup

    monkeypatch.setattr(setup, "prepare_environment", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Invalid map reached model setup")))
    source = tmp_path / "sprite.png"
    _source(source)
    depth_map = tmp_path / "wrong_depth.png"
    assert cv2.imwrite(str(depth_map), np.zeros((7, 11), np.uint8))
    output = tmp_path / "out"
    assert main([str(source), "-o", str(output), "--maps", "ao",
                 "--depth-map", str(depth_map)]) == 1
    assert "11x7" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_irrelevant_or_unmatched_depth_map(tmp_path):
    source = tmp_path / "sprite.png"
    _source(source)
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--maps", "normal", "--depth-map", str(tmp_path / "depth.png")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(source), "--maps", "ao",
              "--depth-map", str(tmp_path / "depth.png")])
    assert exc.value.code == 2


@pytest.mark.parametrize("project_args", [[], ["--no-save-project"],
                                          ["--save-project", "--no-save-project"]])
@pytest.mark.parametrize("mode,expected_models,expected_suffixes", [
    ("both", ("depth", "ai", "clipseg"), {"depth", "normal"}),
    ("depth", ("depth",), {"depth"}),
    ("normal", ("ai", "clipseg"), {"normal"}),
    ("albedo", ("albedo",), {"albedo"}),
    ("ao", ("depth", "ai", "clipseg"), {"ao"}),
    ("roughness", ("roughness",), {"roughness"}),
    ("all", ("depth", "ai", "clipseg", "albedo", "roughness"),
     {"depth", "normal", "albedo", "ao", "roughness"}),
])
def test_cli_generates_selected_maps(tmp_path, monkeypatch, mode, expected_models,
                                     expected_suffixes, project_args):
    from smg import albedo_ai, export, normal_ai, roughness_ai, segmentation, setup
    from smg.depth import inference

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment",
                        lambda models, progress, events=None: prepared.append(tuple(models)))

    class FakeDepth:
        def generate(self, rgba, progress=None):
            return np.tile(np.arange(rgba.shape[1], dtype=np.float32), (rgba.shape[0], 1))

    class FakeNormal:
        def __init__(self, fov):
            assert fov == 60

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(inference, "DepthModel", FakeDepth)
    monkeypatch.setattr(normal_ai, "DSINENormalModel", FakeNormal)
    monkeypatch.setattr(segmentation, "detect_tree_crown", lambda rgba, progress: None)
    monkeypatch.setattr(albedo_ai, "generate_albedo", lambda rgba, progress, **kwargs: rgba.copy())
    monkeypatch.setattr(roughness_ai, "generate_roughness",
                        lambda rgba, progress: np.full(rgba.shape[:2], 0.5, np.float32))

    def unexpected_save(*args, **kwargs):
        raise AssertionError("Direct export must not save a project")

    monkeypatch.setattr(export, "save_project", unexpected_save)

    source = tmp_path / "sprite.png"
    _source(source)
    output = tmp_path / "out"
    assert main(["generate", str(source), "-o", str(output), "--maps", mode,
                 *project_args]) == 0
    assert prepared == [expected_models]
    assert {path.name for path in output.iterdir()} == {
        f"sprite_{suffix}.png" for suffix in expected_suffixes
    }


def test_cli_passes_albedo_settings(tmp_path, monkeypatch):
    from smg import albedo_ai, setup

    monkeypatch.setattr(setup, "prepare_environment", lambda *args, **kwargs: None)
    received = []

    def fake_albedo(rgba, progress, **kwargs):
        received.append(kwargs)
        return rgba.copy()

    monkeypatch.setattr(albedo_ai, "generate_albedo", fake_albedo)
    source = tmp_path / "sprite.png"
    _source(source)
    debug = tmp_path / "debug"
    assert main([str(source), "-o", str(tmp_path / "out"), "--maps", "albedo",
                 "--albedo-strength", "1.4", "--albedo-smooth", "3",
                 "--albedo-shadows", "0.6", "--albedo-debug", str(debug)]) == 0
    assert received == [{"strength": 1.4, "illumination_sigma": 3.0,
                         "shadow_strength": 0.6, "debug": True, "debug_dir": debug}]


def test_cli_crown_controls(tmp_path, monkeypatch):
    from smg import normal_ai, segmentation, setup

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment",
                        lambda models, progress, events=None: prepared.append(tuple(models)))

    class FakeNormal:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(normal_ai, "DSINENormalModel", FakeNormal)
    thresholds = []

    def detect(rgba, progress, threshold=0.5):
        thresholds.append(threshold)
        return None

    monkeypatch.setattr(segmentation, "detect_tree_crown", detect)
    source = tmp_path / "sprite.png"
    _source(source)
    assert main([str(source), "-o", str(tmp_path / "auto"), "--maps", "normal",
                 "--crown-threshold", "0.7"]) == 0
    assert main([str(source), "-o", str(tmp_path / "off"), "--maps", "normal",
                 "--crown-mode", "off"]) == 0
    assert prepared == [("ai", "clipseg"), ("ai",)]
    assert thresholds == [0.7]


def test_cli_batch_and_existing_output(tmp_path, monkeypatch):
    from smg.depth import inference
    from smg import setup

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: prepared.append(tuple(models)))

    class FakeDepth:
        def __init__(self, keep_loaded):
            assert keep_loaded

        def unload(self):
            pass

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
    assert prepared == [("depth",)]
    assert (output / "one_depth.png").exists()
    assert (output / "two_depth.png").exists()
    assert not (output / "one_normal.png").exists()
    assert main(command) == 1
    assert main(command + ["--overwrite"]) == 0


def test_cli_ai_only_skips_depth_model(tmp_path, monkeypatch):
    from smg import normal_ai, segmentation
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
    monkeypatch.setattr(segmentation, "detect_tree_crown", lambda rgba, progress: None)
    source = tmp_path / "sprite.png"
    rgba = _source(source)
    output = tmp_path / "out"
    assert main([str(source), "-o", str(output), "--maps", "normal",
                 "--normal-source", "ai", "--no-invert-ai-x"]) == 0
    assert prepared == [("ai", "clipseg")]
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


def test_cli_round_trip_preserves_project_foliage_mask(tmp_path):
    source = tmp_path / "foliage.png"
    rgba = _source(source)
    depth = np.full(rgba.shape[:2], 0.6, np.float32)
    mask = np.zeros(rgba.shape[:2], bool)
    mask[2:6, 3:9] = True
    project = tmp_path / "foliage.ssoul"
    save_project(project, source, depth, 20.0, "OpenGL", foliage_mask=mask)
    output = tmp_path / "out"

    assert main([str(project), "-o", str(output), "--maps", "depth", "--save-project"]) == 0
    _source_path, saved_depth, _strength, _convention, restored = load_project(
        output / "foliage.ssoul", with_foliage_mask=True
    )
    assert np.array_equal(saved_depth, depth)
    assert np.array_equal(restored, mask)

    output_without_crown = tmp_path / "out_without_crown"
    assert main([str(project), "-o", str(output_without_crown), "--maps", "depth",
                 "--crown-mode", "off", "--save-project"]) == 0
    *_, preserved = load_project(output_without_crown / "foliage.ssoul",
                                 with_foliage_mask=True)
    assert np.array_equal(preserved, mask)


def test_cli_applies_saved_crown_mask_to_ai_normal(tmp_path, monkeypatch):
    from smg import normal_ai, setup

    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, events=None: None)

    class FlatAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(normal_ai, "DSINENormalModel", FlatAI)
    source = tmp_path / "tree.png"
    rgba = np.full((54, 62, 4), 120, np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(source)
    mask = np.zeros(rgba.shape[:2], bool)
    mask[7:47, 9:53] = True
    project = tmp_path / "tree.ssoul"
    save_project(project, source, None, 20.0, "OpenGL", foliage_mask=mask)

    assert main([str(project), "-o", str(tmp_path / "out"), "--maps", "normal"]) == 0
    normal = open_png(tmp_path / "out" / "tree_normal.png")
    assert normal[27, 16, 0] < 100 < normal[27, 46, 0]
    assert normal[27, 4, 0] == 128


def test_cli_auto_detects_tree_and_saves_crown_mask(tmp_path, monkeypatch):
    from smg import normal_ai, segmentation, setup

    prepared = []
    monkeypatch.setattr(setup, "prepare_environment",
                        lambda models, progress, events=None: prepared.append(tuple(models)))

    class FlatAI:
        def __init__(self, fov):
            pass

        def generate(self, rgba, progress=None):
            vectors = np.zeros((*rgba.shape[:2], 3), np.float32)
            vectors[..., 2] = 1
            return vectors

    monkeypatch.setattr(normal_ai, "DSINENormalModel", FlatAI)
    source = tmp_path / "tree.png"
    rgba = np.full((54, 62, 4), 120, np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(source)
    mask = np.zeros(rgba.shape[:2], bool)
    mask[7:47, 9:53] = True
    monkeypatch.setattr(segmentation, "detect_tree_crown", lambda rgba, progress: mask)
    depth_map = tmp_path / "depth.png"
    assert cv2.imwrite(str(depth_map), np.full(rgba.shape[:2], 128, np.uint8))
    output = tmp_path / "out"
    assert main([str(source), "-o", str(output), "--maps", "normal",
                 "--depth-map", str(depth_map), "--save-project"]) == 0
    assert prepared == [("ai", "clipseg")]
    normal = open_png(output / "tree_normal.png")
    assert normal[27, 16, 0] < 100 < normal[27, 46, 0]
    *_, restored = load_project(output / "tree.ssoul", with_foliage_mask=True)
    assert np.array_equal(restored, mask)

    monkeypatch.setattr(segmentation, "detect_tree_crown", lambda rgba, progress: None)
    output2 = tmp_path / "other"
    assert main([str(source), "-o", str(output2), "--maps", "normal"]) == 0
    plain = open_png(output2 / "tree_normal.png")
    assert plain[27, 16, 0] == 128


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
    assert "--crown-threshold" in generate_help
    assert "--albedo-strength" in generate_help
    assert "--no-save-project" in generate_help
    assert "hybrid" not in generate_help.lower()
