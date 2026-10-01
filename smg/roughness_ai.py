"""SuperMat roughness inference in an isolated process and dependency runtime."""

from collections.abc import Callable
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from smg.model_paths import SUPERMAT_ROOT


SOURCE_COMMIT = "4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4"
MODEL_REPO = "oyiya/SuperMat"
MODEL_REVISION = "91ffb8edaf259a1e5bbcbcd53b702ad44d70b709"
BASE_REPO = "sd2-community/stable-diffusion-2-1"
BASE_REVISION = "bb2154823665391b4fb29b0b9cf82a198964ee05"
CHECKPOINT_SIZE = 3541172771
BASE_WEIGHT_SIZES = {
    "text_encoder/model.fp16.safetensors": 680821096,
    "vae/diffusion_pytorch_model.fp16.safetensors": 167335342,
}
BASE_FILES = (
    "feature_extractor/preprocessor_config.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "unet/config.json",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
)
SOURCE_FILES = (
    "src/models/supermat_unet_2d_condition.py",
    "src/pipelines/pipeline_supermat_stable_diffusion.py", "src/utils.py",
)


def runtime_python(root: Path = SUPERMAT_ROOT) -> Path:
    return root / ".venv-supermat" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )


def model_available(root: Path = SUPERMAT_ROOT) -> bool:
    checkpoint = root / "checkpoints/supermat.pth"
    packages = root / ".venv-supermat" / (
        "Lib/site-packages" if sys.platform == "win32"
        else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    return (
        all((root / name).is_file() for name in SOURCE_FILES)
        and checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_SIZE
        and all((root / "base_model" / name).is_file()
                and (root / "base_model" / name).stat().st_size > 0 for name in BASE_FILES)
        and all((root / "base_model" / name).stat().st_size == size
                for name, size in BASE_WEIGHT_SIZES.items())
        and runtime_python(root).is_file()
        and ((root / ".venv-supermat/runtime-ready.json").is_file()
             or all((packages / name).is_dir() for name in ("diffusers", "accelerate")))
    )


def generate_roughness(rgba: np.ndarray,
                       progress: Callable[[str], None] | None = None, session=None) -> np.ndarray:
    """Return linear roughness [0, 1], at source size; export supplies source alpha."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("SuperMat ожидает RGBA uint8")
    if not rgba.shape[0] or not rgba.shape[1]:
        raise ValueError("Изображение пустое")
    if not np.any(rgba[..., 3]):
        return np.zeros(rgba.shape[:2], np.float32)
    if not model_available(SUPERMAT_ROOT):
        raise RuntimeError("SuperMat не подготовлена. Запустите «Подготовить AI» или setup --models roughness.")
    report = progress or (lambda _message: None)
    report("SuperMat: загрузка модели и генерация Roughness…")
    with tempfile.TemporaryDirectory(prefix="sprite-soul-supermat-") as temporary:
        source = Path(temporary) / "input.png"
        output = Path(temporary) / "roughness.npy"
        Image.fromarray(rgba).save(source)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Only the child loads diffusers and allocates SuperMat CUDA tensors.
        worker = Path(__file__).with_name("supermat_worker.py")
        if session is not None:
            session.run([str(source), str(output)], report)
        else:
            _run_worker(worker, source, output, env, report)
        if not output.is_file():
            raise RuntimeError("SuperMat не сохранила Roughness")
        roughness = np.load(output, allow_pickle=False)
    if roughness.shape != rgba.shape[:2] or not np.isfinite(roughness).all():
        raise ValueError("SuperMat вернула неверную карту Roughness")
    return np.clip(roughness, 0, 1).astype(np.float32)


def _run_worker(worker, source, output, env, report):
    process = subprocess.Popen(
        [str(runtime_python(SUPERMAT_ROOT)), str(worker), str(SUPERMAT_ROOT), str(source), str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    tail = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                tail = (tail + [line])[-10:]
                report(line)
        if process.wait() != 0:
            raise RuntimeError("Ошибка SuperMat: " + " | ".join(tail))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
