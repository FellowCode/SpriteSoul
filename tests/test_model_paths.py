from pathlib import Path

from smg import model_paths


def test_windows_nested_install_uses_short_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    root = tmp_path / ("nested" * 24) / "IntrinsicAnything"
    environment = model_paths.intrinsic_environment(root)
    assert environment.parent == tmp_path / "cache/ss-venvs"
    assert environment == model_paths.intrinsic_environment(root)
    assert environment != model_paths.intrinsic_environment(root / "other")
    assert model_paths.intrinsic_python(root) == environment / "Scripts/python.exe"


def test_short_windows_install_keeps_local_environment(monkeypatch):
    monkeypatch.setattr(model_paths.sys, "platform", "win32")
    root = Path("C:/ia")
    assert model_paths.intrinsic_environment(root) == root / ".venv-intrinsic"


def test_non_windows_install_keeps_local_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(model_paths.sys, "platform", "linux")
    root = tmp_path / ("nested" * 24)
    assert model_paths.intrinsic_python(root) == root / ".venv-intrinsic/bin/python"
