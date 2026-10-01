import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from smg import roughness_ai, setup, supermat_setup
from smg.cli import main
from smg.export import export_roughness


def test_export_roughness_preserves_linear_values_and_alpha(tmp_path):
    values = np.array([[0, 0.25, 0.5, 1]], np.float32)
    alpha = np.array([[0, 1, 128, 255]], np.uint8)
    path = export_roughness("sprite.png", values, alpha, tmp_path)
    assert path.name == "sprite_roughness.png"
    pixels = np.asarray(Image.open(path))
    assert np.array_equal(pixels[..., 3], alpha)
    assert np.array_equal(pixels[0, :, 0], [0, 64, 128, 255])
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 0], pixels[..., 2])


@pytest.mark.parametrize("values", [np.zeros((2, 2)), np.array([[np.nan]])])
def test_export_rejects_invalid_prediction(tmp_path, values):
    with pytest.raises(ValueError):
        export_roughness("sprite.png", values, np.ones((1, 1), np.uint8), tmp_path)


@pytest.mark.parametrize("rgba", [np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 4), np.float32)])
def test_generation_validates_rgba(rgba):
    with pytest.raises(ValueError, match="RGBA uint8"):
        roughness_ai.generate_roughness(rgba)


def test_transparent_sprite_does_not_launch_model(monkeypatch):
    monkeypatch.setattr(roughness_ai.subprocess, "Popen", lambda *a, **kw: pytest.fail("No inference needed"))
    result = roughness_ai.generate_roughness(np.zeros((3, 5, 4), np.uint8))
    assert result.shape == (3, 5) and result.dtype == np.float32
    assert not result.any()


@pytest.mark.parametrize("prediction", [np.full((3, 5), 0.37, np.float32),
                                       np.zeros((1, 1), np.float32),
                                       np.full((3, 5), np.inf, np.float32)])
def test_subprocess_prediction_is_loaded_and_validated(monkeypatch, prediction):
    monkeypatch.setattr(roughness_ai, "model_available", lambda root: True)
    source = np.full((3, 5, 4), 255, np.uint8)
    source[..., 3] = np.arange(15, dtype=np.uint8).reshape(3, 5)

    class Process:
        stdout = io.StringIO("SuperMat: test progress\n")

        def __init__(self, command, **kwargs):
            assert Path(command[1]).name == "supermat_worker.py"
            assert np.array_equal(np.asarray(Image.open(command[-2])), source)
            np.save(command[-1], prediction)

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(roughness_ai.subprocess, "Popen", Process)
    progress = []
    if prediction.shape != (3, 5) or not np.isfinite(prediction).all():
        with pytest.raises(ValueError, match="неверную карту"):
            roughness_ai.generate_roughness(source, progress.append)
    else:
        assert np.array_equal(roughness_ai.generate_roughness(source, progress.append), prediction)
    assert "SuperMat: test progress" in progress


def test_missing_supermat_reports_preparation_command(monkeypatch):
    monkeypatch.setattr(roughness_ai, "model_available", lambda root: False)
    with pytest.raises(RuntimeError, match="setup --models roughness"):
        roughness_ai.generate_roughness(np.full((3, 5, 4), 255, np.uint8))


@pytest.mark.parametrize("shared_dependencies", [False, True])
def test_availability_requires_all_components(tmp_path, monkeypatch, shared_dependencies):
    monkeypatch.setattr(roughness_ai, "CHECKPOINT_SIZE", 8)
    monkeypatch.setattr(roughness_ai, "BASE_WEIGHT_SIZES", {})
    files = [*roughness_ai.SOURCE_FILES, "checkpoints/supermat.pth",
             *[f"base_model/{name}" for name in roughness_ai.BASE_FILES]]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"12345678")
    python = roughness_ai.runtime_python(tmp_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    packages = python.parents[1] / (
        "Lib/site-packages" if roughness_ai.sys.platform == "win32"
        else f"lib/python{roughness_ai.sys.version_info.major}.{roughness_ai.sys.version_info.minor}/site-packages"
    )
    if shared_dependencies:
        (python.parents[1] / "runtime-ready.json").write_text("{}")
    else:
        for name in ("diffusers", "accelerate"):
            (packages / name).mkdir(parents=True)
    assert roughness_ai.model_available(tmp_path)
    (tmp_path / "base_model/tokenizer/vocab.json").unlink()
    assert not roughness_ai.model_available(tmp_path)


def test_setup_roughness_uses_cuda_and_routes_events(monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(setup, "ensure_cuda", lambda progress, events=None: calls.append("cuda"))
    monkeypatch.setattr(supermat_setup, "prepare_supermat",
                        lambda progress, event_callback: calls.append(event_callback))
    setup.prepare_environment(("roughness",), events=events.append)
    assert calls == ["cuda", events.append]
    assert events[-1]["phase"] == "setup" and events[-1]["status"] == "done"


def test_cli_setup_accepts_roughness(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, install_cuda, events:
                        calls.append((models, install_cuda)))
    assert main(["setup", "--models", "roughness", "--skip-cuda"]) == 0
    assert calls == [(("roughness",), False)]


def test_cached_preparation_never_contacts_hub(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(supermat_setup, "SUPERMAT_ROOT", tmp_path)
    monkeypatch.setattr(supermat_setup, "CHECKPOINT_SIZE", 8)
    monkeypatch.setattr(supermat_setup, "BASE_WEIGHT_SIZES", {})
    monkeypatch.setattr(supermat_setup, "_prepare_source", lambda *a: None)
    monkeypatch.setattr(supermat_setup, "_prepare_runtime", lambda *a: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("Cached weights must work offline"))
    for name in ["checkpoints/supermat.pth", *[f"base_model/{n}" for n in roughness_ai.BASE_FILES]]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"12345678")
    events = []
    supermat_setup.prepare_supermat(lambda m: None, events.append)
    assert events[-1]["current"] == events[-1]["total"] == 14
    assert events[-1]["status"] == "cached"


def test_fresh_preparation_pins_revisions_and_omits_base_unet(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(supermat_setup, "SUPERMAT_ROOT", tmp_path)
    monkeypatch.setattr(supermat_setup, "_prepare_source", lambda *a: None)
    monkeypatch.setattr(supermat_setup, "_prepare_runtime", lambda *a: None)
    queries, downloads = [], []

    class Api:
        def model_info(self, repo, revision, files_metadata):
            queries.append((repo, revision))
            names = ["supermat.pth"] if repo == roughness_ai.MODEL_REPO else roughness_ai.BASE_FILES
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=12_000_000,
                                                             lfs=SimpleNamespace(sha256="hash")) for n in names])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(supermat_setup, "download_ranges",
                        lambda repo, name, revision, target, size, digest, progress:
                        downloads.append((repo, name, revision)))
    supermat_setup.prepare_supermat(lambda m: None)
    assert queries == [(roughness_ai.MODEL_REPO, roughness_ai.MODEL_REVISION),
                       (roughness_ai.BASE_REPO, roughness_ai.BASE_REVISION)]
    assert len(downloads) == 12
    assert not any(name.startswith("unet/") and name != "unet/config.json" for _, name, _ in downloads)


@pytest.mark.parametrize("resume", [False, True])
def test_range_download_checks_hash_and_resumes(tmp_path, monkeypatch, resume):
    import requests

    payload = b"a complete, resumable checkpoint"
    target = tmp_path / "weights.pth"
    if resume:
        partial = target.with_suffix(".pth.part")
        partial.write_bytes(payload[:8] + b"\0" * (len(payload) - 8))
        target.with_suffix(".pth.parts.json").write_text("[0]")
    ranges = []

    def get(url, headers, **kwargs):
        start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        ranges.append((start, stop))
        class Response:
            status_code = 206
            headers = {"Content-Range": f"bytes {start}-{stop}/{len(payload)}"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield payload[start:stop + 1]
        return Response()

    monkeypatch.setattr(requests, "get", get)
    updates = []
    supermat_setup.download_ranges("repo", "weights.pth", "pinned", target, len(payload),
                                  hashlib.sha256(payload).hexdigest(),
                                  lambda current, total: updates.append((current, total)), chunk_size=8)
    assert target.read_bytes() == payload
    assert updates[-1] == (len(payload), len(payload))
    assert (0, 7) not in ranges if resume else (0, 7) in ranges
    assert not target.with_suffix(".pth.parts.json").exists()


def test_corrupted_resume_is_not_published_and_resets_journal(tmp_path):
    target = tmp_path / "weights.pth"
    payload = b"invalid checkpoint"
    target.with_suffix(".pth.part").write_bytes(payload)
    target.with_suffix(".pth.parts.json").write_text(json.dumps(list(range((len(payload) + 7) // 8))))
    with pytest.raises(RuntimeError, match="SHA256"):
        supermat_setup.download_ranges("repo", "weights.pth", "revision", target,
                                      len(payload), hashlib.sha256(b"expected checkpoint").hexdigest(),
                                      lambda current, total: None, chunk_size=8)
    assert not target.exists()
    assert not target.with_suffix(".pth.parts.json").exists()
