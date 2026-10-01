import subprocess

import pytest

from smg.albedo_ai import _run_intrinsic
from smg.intrinsic_worker import inference_weights


def test_inference_uses_ema_and_owns_weights_without_training_state():
    torch = pytest.importorskip("torch")
    checkpoint = {
        "model.diffusion_model.layer.weight": torch.tensor([1.0, 2.0]),
        "model_ema.diffusion_modellayerweight": torch.tensor([3.0, 4.0]),
        "model_ema.decay": torch.tensor(0.9999),
        "model_ema.num_updates": torch.tensor(12),
        "first_stage_model.layer.weight": torch.tensor([5.0], dtype=torch.float16),
        "cond_stage_model.counter": torch.tensor(7, dtype=torch.int64),
    }
    weights = inference_weights(checkpoint)
    assert set(weights) == {
        "model.diffusion_model.layer.weight", "first_stage_model.layer.weight",
        "cond_stage_model.counter",
    }
    assert weights["model.diffusion_model.layer.weight"].tolist() == [3.0, 4.0]
    assert weights["model.diffusion_model.layer.weight"].dtype == torch.float16
    assert weights["cond_stage_model.counter"].dtype == torch.int64
    checkpoint["first_stage_model.layer.weight"].fill_(0)
    checkpoint["model_ema.diffusion_modellayerweight"].fill_(0)
    assert weights["first_stage_model.layer.weight"].item() == 5.0
    assert weights["model.diffusion_model.layer.weight"].tolist() == [3.0, 4.0]


def test_inference_rejects_unmatched_ema():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="unmatched EMA"):
        inference_weights({"model_ema.unknown": torch.tensor([1.0])})


@pytest.mark.parametrize("code", [0xC0000005, -1073741819])
def test_native_crash_preserves_log_and_exit_code(tmp_path, monkeypatch, code):
    monkeypatch.chdir(tmp_path)

    def crash(command, **kwargs):
        kwargs["stdout"].write("LightningDeprecationWarning\nWindows fatal exception: access violation\n")
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr("smg.albedo_ai.subprocess.run", crash)
    with pytest.raises(RuntimeError, match="0xC0000005") as error:
        _run_intrinsic(["python", "worker"], tmp_path)
    log, = (tmp_path / "generated/logs").glob("intrinsic-*.log")
    assert "Windows fatal exception" in log.read_text(encoding="utf-8")
    assert str(log) in str(error.value)
