from pathlib import Path
import sys

import numpy as np
import pytest

from smg.model_session import ModelSession


@pytest.fixture
def worker_session(tmp_path):
    worker = tmp_path / "worker.py"
    module_dir = Path(__file__).resolve().parents[1] / "smg"
    worker.write_text(
        f"import sys\nsys.path.insert(0, {str(module_dir)!r})\n"
        "from model_session import serve\n"
        "print('model loaded', flush=True)\n"
        "def request(args):\n"
        "    if args == ['error']: raise RuntimeError('bad request')\n"
        "    if args == ['crash']: sys.exit(7)\n"
        "    print('inference: ' + args[0], flush=True)\n"
        "serve(request)\n", encoding="utf-8",
    )
    session = ModelSession(Path(sys.executable), worker, tmp_path)
    yield session
    session.close()


def test_worker_loads_once_and_recovers_from_request_error(worker_session):
    progress = []
    worker_session.run(["one"], progress.append)
    process = worker_session.process
    with pytest.raises(RuntimeError, match="bad request"):
        worker_session.run(["error"], progress.append)
    worker_session.run(["two"], progress.append)
    assert worker_session.process is process
    assert progress.count("model loaded") == 1
    assert "inference: one" in progress and "inference: two" in progress
    worker_session.close()
    assert worker_session.process is None and process.poll() is not None
    assert process.stdin.closed and process.stdout.closed


def test_worker_crash_is_reported_and_session_can_restart(worker_session):
    with pytest.raises(RuntimeError, match="0x00000007"):
        worker_session.run(["crash"])
    assert worker_session.process is None
    worker_session.run(["after crash"])


def test_intrinsic_loader_keeps_model_for_multiple_requests(monkeypatch):
    from smg import intrinsic_worker

    calls = []
    sentinel = object()
    monkeypatch.setattr(intrinsic_worker, "load_model", lambda *args, **kwargs:
                        calls.append(args) or sentinel)
    loader = intrinsic_worker.cached_model_loader()
    assert loader("config", "checkpoint", "cuda") is sentinel
    assert loader("config", "checkpoint", "cuda") is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize("keep_loaded", [False, True])
@pytest.mark.parametrize("kind", ["depth", "normal"])
def test_in_process_models_release_once_per_batch(monkeypatch, keep_loaded, kind):
    torch = pytest.importorskip("torch")
    from smg.depth.inference import DepthModel
    from smg.normal_ai import DSINENormalModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cleanup = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleanup.append(True))
    model = DepthModel(keep_loaded) if kind == "depth" else DSINENormalModel(keep_loaded=keep_loaded)
    loads = []

    def load():
        if model.model is None:
            loads.append(True)
            if kind == "depth":
                model.model = object()
            else:
                return object()

    monkeypatch.setattr(model, "_load", load)
    monkeypatch.setattr(model, "_infer", lambda *args: np.ones((3, 5), np.float32))
    rgba = np.full((3, 5, 4), 255, np.uint8)
    model.generate(rgba)
    model.generate(rgba)
    assert len(loads) == (1 if keep_loaded else 2)
    if keep_loaded:
        assert model.model is not None and not cleanup
        model.unload()
    assert model.model is None
    assert len(cleanup) == (1 if keep_loaded else 2)


def test_roughness_passes_files_to_reusable_session(monkeypatch):
    from smg import roughness_ai
    from PIL import Image

    monkeypatch.setattr(roughness_ai, "model_available", lambda root: True)
    rgba = np.full((3, 5, 4), 173, np.uint8)
    calls = []

    class Session:
        def run(self, arguments, progress):
            source, output = map(Path, arguments)
            assert np.array_equal(np.asarray(Image.open(source)), rgba)
            np.save(output, np.full((3, 5), 0.37, np.float32))
            calls.append(arguments)

    session = Session()
    for _ in range(2):
        result = roughness_ai.generate_roughness(rgba, session=session)
        assert np.allclose(result, 0.37)
    assert len(calls) == 2


def test_albedo_passes_inference_args_to_reusable_session(monkeypatch):
    from smg import albedo_ai
    from PIL import Image

    monkeypatch.setattr(albedo_ai, "_experiment", lambda: (Path("intrinsic"), Path("python")))
    rgba = np.full((8, 11, 4), 173, np.uint8)
    calls = []

    class Session:
        def run(self, arguments, progress):
            assert arguments[0] == "--input_dir"
            input_dir = Path(arguments[1])
            output_dir = Path(arguments[3])
            for source in input_dir.glob("*.png"):
                Image.open(source).save(output_dir / source.name)
            calls.append(arguments)

    session = Session()
    for _ in range(2):
        result = albedo_ai.generate_albedo(rgba, session=session)
        assert result.shape == rgba.shape and np.array_equal(result[..., 3], rgba[..., 3])
    assert len(calls) == 2
