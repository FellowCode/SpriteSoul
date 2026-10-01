import io
import json
import zipfile

import pytest

from smg import setup
from smg import normal_ai
from smg.cli import main


def test_setup_downloads_selected_model_files(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "ensure_cuda", lambda progress, events=None: calls.append("cuda"))
    monkeypatch.setattr(setup, "_model_file", lambda repo, name: calls.append((repo, name)))
    monkeypatch.setattr(setup, "_source_root", lambda progress: calls.append("source"))
    monkeypatch.setattr(setup, "_ensure_geffnet", lambda progress, events=None: calls.append("geffnet"))
    setup.prepare_environment(("depth", "ai"))
    assert calls == [
        "cuda",
        (setup.MODEL_ID, "config.json"),
        (setup.MODEL_ID, "preprocessor_config.json"),
        (setup.MODEL_ID, "model.safetensors"),
        "geffnet", "source", (setup.CHECKPOINT_REPO, "dsine.pt"),
    ]


def test_missing_models_reports_only_unavailable(monkeypatch):
    monkeypatch.setattr(setup, "model_available", lambda model: model == "depth")
    assert setup.missing_models(("depth", "ai", "albedo")) == ("ai", "albedo")


def test_setup_prepares_albedo_on_demand(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "_prepare_intrinsic", lambda progress, events: calls.append("albedo"))
    setup.prepare_environment(("albedo",), install_cuda=False)
    assert calls == ["albedo"]


def test_setup_prepares_clipseg_on_demand_and_uses_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "ensure_cuda", lambda progress, events=None: calls.append("cuda"))
    monkeypatch.setattr(setup, "_model_file", lambda repo, name: calls.append((repo, name)))
    setup.prepare_environment(("clipseg",))
    assert calls == ["cuda", *((setup.CLIPSEG_MODEL_ID, name) for name in setup.CLIPSEG_FILES)]


def test_setup_installs_cuda_wheel_only_when_needed(monkeypatch):
    statuses = iter([{"error": "No module named torch"},
                     {"version": "2.6.0+cu126", "cuda": "12.6", "available": True}])
    monkeypatch.setattr(setup, "_torch_status", lambda: next(statuses))
    monkeypatch.setattr(setup, "_cuda_index", lambda: "cu126")
    commands = []

    class FakeProcess:
        stdout = io.StringIO("Successfully installed torch\n")

        def wait(self):
            return 0

    monkeypatch.setattr(setup.subprocess, "Popen", lambda command, **kwargs: commands.append(command) or FakeProcess())
    messages = []
    setup.ensure_cuda(messages.append)
    assert commands == [[setup.sys.executable, "-m", "pip", "install", "--upgrade",
                         "--force-reinstall", "--progress-bar", "on", "torch==2.6.0", "--index-url",
                         "https://download.pytorch.org/whl/cu126"]]
    assert "CUDA готова" in messages[-1]


def test_setup_cli_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, install_cuda, events=None:
                        calls.append((models, install_cuda)))
    assert main(["setup", "--models", "ai", "--skip-cuda"]) == 0
    assert calls == [(("ai",), False)]


def test_setup_cli_accepts_clipseg(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, install_cuda, events=None:
                        calls.append((models, install_cuda)))
    assert main(["setup", "--models", "clipseg", "--skip-cuda"]) == 0
    assert calls == [(("clipseg",), False)]


def test_setup_cli_all_includes_every_model(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, install_cuda, events=None:
                        calls.append(models))
    assert main(["setup", "--models", "all", "--skip-cuda"]) == 0
    assert calls == [("depth", "ai", "clipseg", "albedo")]


def test_setup_cli_accepts_albedo(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "prepare_environment", lambda models, progress, install_cuda, events=None:
                        calls.append(models))
    assert main(["setup", "--models", "albedo", "--skip-cuda"]) == 0
    assert calls == [("albedo",)]


def test_clipseg_availability_checks_every_required_file(monkeypatch):
    checked = []
    monkeypatch.setattr(setup, "_cached_model_file", lambda repo, name:
                        checked.append((repo, name)) or True)
    assert setup.model_available("clipseg")
    assert checked == [(setup.CLIPSEG_MODEL_ID, name) for name in setup.CLIPSEG_FILES]


def test_cuda_installed_but_unavailable_reports_driver_issue(monkeypatch):
    monkeypatch.setattr(setup, "_torch_status", lambda: {"version": "2.6.0+cu126",
                                                       "cuda": "12.6", "available": False})
    monkeypatch.setattr(setup, "_cuda_index", lambda: "cu126")
    monkeypatch.setattr(setup.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("PyTorch must not be reinstalled"))
    with pytest.raises(RuntimeError, match="GPU недоступна"):
        setup.ensure_cuda(lambda message: None)


def test_ai_dependency_installed_without_replacing_cuda_torch(monkeypatch):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    calls = []
    monkeypatch.setattr(setup, "_pip_install", lambda command, progress, label, events=None, component=None: calls.append(command))
    setup._ensure_geffnet(lambda message: None)
    assert calls == [[setup.sys.executable, "-m", "pip", "install", "--no-deps",
                      "--progress-bar", "on", "geffnet==1.0.2"]]


def test_pip_progress_emits_byte_counts(monkeypatch):
    class FakeProcess:
        stdout = io.StringIO("Downloading torch.whl (15.8 MB)\n"
                             "   ---------- 1.2/15.8 MB 3.0 MB/s eta 0:00:05\n"
                             "Successfully installed torch\n")

        def wait(self):
            return 0

    monkeypatch.setattr(setup.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    events = []
    setup._pip_install(["pip"], lambda message: None, "torch", events.append, "torch")
    updates = [event for event in events if event["status"] == "update"]
    assert updates == [{"event": "progress", "phase": "pip_install", "status": "update",
                        "message": "---------- 1.2/15.8 MB 3.0 MB/s eta 0:00:05",
                        "component": "torch", "package": "torch.whl",
                        "current": 1200000, "total": 15800000, "unit": "bytes"}]


def test_setup_json_lines_are_parseable(monkeypatch, capsys):
    def fake_prepare(models, progress, install_cuda, events):
        events({"event": "progress", "phase": "cuda", "status": "ready", "message": "CUDA готова"})

    monkeypatch.setattr(setup, "prepare_environment", fake_prepare)
    assert main(["setup", "--models", "none", "--progress", "json"]) == 0
    output = capsys.readouterr()
    lines = [json.loads(line) for line in output.out.splitlines()]
    assert output.err == ""
    assert lines == [
        {"schema_version": 1, "event": "progress", "phase": "cuda", "status": "ready", "message": "CUDA готова"},
        {"schema_version": 1, "event": "complete", "status": "success", "processed": 0, "failed": 0},
    ]


def test_dsine_source_download_reports_bytes(tmp_path, monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(f"DSINE-{normal_ai.DSINE_COMMIT}/models/dsine/v02.py", "# cached source")
    payload = data.getvalue()

    def response(*args, **kwargs):
        stream = io.BytesIO(payload)
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    monkeypatch.setattr(normal_ai, "DSINE_SOURCE_ROOT", tmp_path / f"DSINE-{normal_ai.DSINE_COMMIT}")
    monkeypatch.setattr(normal_ai.urllib.request, "urlopen", response)
    updates = []
    root = normal_ai._source_root(lambda current, total: updates.append((current, total)))
    assert (root / "models" / "dsine" / "v02.py").exists()
    assert updates[-1] == (len(payload), len(payload))
