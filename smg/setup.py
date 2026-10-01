"""Prepare CUDA PyTorch and cache the selected inference models."""

import io
import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

from smg.depth.inference import MODEL_ID
from smg.model_paths import HUGGINGFACE_HUB_CACHE, INTRINSIC_ROOT, MODELS_ROOT
from smg.normal_ai import CHECKPOINT_REPO, DSINE_SOURCE_ROOT, _source_root
from smg.segmentation import MODEL_FILES as CLIPSEG_FILES, MODEL_ID as CLIPSEG_MODEL_ID


Progress = Callable[[str], None]
Events = Callable[[dict], None]
TORCH_VERSION = "2.6.0"
DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
INTRINSIC_COMMIT = "5fa5ec07f7e09101710c80bc95738fabb0cbfc80"
INTRINSIC_MODEL_REPO = "LittleFrog/IntrinsicAnything"
MODEL_LABELS = {
    "depth": "Depth Anything V2",
    "ai": "DSINE",
    "albedo": "IntrinsicAnything (Albedo)",
    "clipseg": "CLIPSeg (определение кроны)",
    "roughness": "SuperMat (шероховатость)",
}
_PIP_BYTES = re.compile(r"(?P<current>\d+(?:\.\d+)?)/(?P<total>\d+(?:\.\d+)?) (?P<unit>kB|MB|GB)")
_PIP_UNITS = {"kB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3}


def _cached_model_file(repo: str, filename: str) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        hf_hub_download(
            repo, filename, cache_dir=HUGGINGFACE_HUB_CACHE, local_files_only=True
        )
        return True
    except LocalEntryNotFoundError:
        return False


def model_available(model: str) -> bool:
    """Return whether a model can be used without downloading anything."""
    if model == "roughness":
        from smg.roughness_ai import model_available as supermat_available
        return supermat_available()
    if model == "depth":
        return all(_cached_model_file(MODEL_ID, filename) for filename in (
            "config.json", "preprocessor_config.json", "model.safetensors"
        ))
    if model == "ai":
        return (
            (DSINE_SOURCE_ROOT / "models" / "dsine" / "v02.py").is_file()
            and _cached_model_file(CHECKPOINT_REPO, "dsine.pt")
        )
    if model == "albedo":
        python = INTRINSIC_ROOT / ".venv-intrinsic" / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        return (
            (INTRINSIC_ROOT / "inference.py").is_file()
            and python.is_file()
            and (INTRINSIC_ROOT / "weights/albedo/configs/albedo_project.yaml").is_file()
            and (INTRINSIC_ROOT / "weights/albedo/checkpoints/last.ckpt").is_file()
        )
    if model == "clipseg":
        return all(_cached_model_file(CLIPSEG_MODEL_ID, name) for name in CLIPSEG_FILES)
    raise ValueError(f"Неизвестная модель: {model}")


def missing_models(models: Iterable[str]) -> tuple[str, ...]:
    return tuple(model for model in models if not model_available(model))


def _emit(events: Events | None, phase: str, status: str, message: str, **details) -> None:
    if events is not None:
        events({"event": "progress", "phase": phase, "status": status,
                "message": message, **details})


def _torch_status() -> dict:
    script = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, "
        "'available': torch.cuda.is_available()}))\n"
        "except Exception as exc:\n"
        " print(json.dumps({'error': str(exc)}))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=60, check=False,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"error": (result.stderr or result.stdout).strip() or "PyTorch не запускается"}


def _cuda_index() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Не найден драйвер NVIDIA. Установите драйвер: {DRIVER_URL}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Не удалось обнаружить видеокарту NVIDIA. Проверьте драйвер: {DRIVER_URL}")
    try:
        parts = [int(part) for part in result.stdout.splitlines()[0].strip().split(".")[:3]]
        version = tuple((parts + [0, 0, 0])[:3])
    except ValueError as exc:
        raise RuntimeError("Не удалось определить версию драйвера NVIDIA") from exc
    if sys.platform == "win32":
        cu126, cu124 = (560, 76, 0), (551, 61, 0)
    else:
        cu126, cu124 = (560, 28, 3), (550, 54, 14)
    if version >= cu126:
        return "cu126"
    if version >= cu124:
        return "cu124"
    raise RuntimeError(f"Драйвер NVIDIA слишком старый для CUDA 12.4. Обновите его: {DRIVER_URL}")


def ensure_cuda(progress: Progress, events: Events | None = None) -> None:
    _emit(events, "cuda", "start", "Проверка CUDA")
    loaded = sys.modules.get("torch")
    if loaded is not None and loaded.cuda.is_available():
        message = f"CUDA готова: PyTorch {loaded.__version__}, CUDA {loaded.version.cuda}"
        progress(message)
        _emit(events, "cuda", "ready", message)
        return
    status = _torch_status()
    if status.get("available"):
        message = f"CUDA готова: PyTorch {status['version']}, CUDA {status['cuda']}"
        progress(message)
        _emit(events, "cuda", "ready", message)
        return
    index = _cuda_index()
    if status.get("cuda"):
        raise RuntimeError(
            f"PyTorch с CUDA {status['cuda']} установлен, но GPU недоступна. "
            f"Проверьте драйвер NVIDIA: {DRIVER_URL}"
        )
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch уже загружен в этом процессе. Перезапустите приложение и повторите подготовку.")
    message = f"Установка PyTorch {TORCH_VERSION} с CUDA ({index})..."
    progress(message)
    _emit(events, "pip_install", "start", message, component="torch")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
               "--progress-bar", "on", f"torch=={TORCH_VERSION}", "--index-url",
               f"https://download.pytorch.org/whl/{index}"]
    _pip_install(command, progress, "Установка CUDA-сборки PyTorch", events, "torch")
    status = _torch_status()
    if not status.get("available"):
        raise RuntimeError(
            f"PyTorch установлен, но CUDA недоступна: {status.get('error', status.get('cuda'))}. "
            f"Проверьте драйвер NVIDIA: {DRIVER_URL}"
        )
    message = f"CUDA готова: PyTorch {status['version']}, CUDA {status['cuda']}"
    progress(message)
    _emit(events, "pip_install", "done", message, component="torch")
    _emit(events, "cuda", "ready", message)


def _pip_install(command: list[str], progress: Progress, label: str,
                 events: Events | None = None, component: str = "package") -> None:
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except OSError as exc:
        raise RuntimeError(f"Не удалось запустить pip: {exc}") from exc
    tail = []
    package = component
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line:
            tail.append(line)
            tail = tail[-5:]
            progress(line)
            if line.startswith("Downloading "):
                package = line.split()[1]
            match = _PIP_BYTES.search(line)
            if match:
                factor = _PIP_UNITS[match["unit"]]
                _emit(events, "pip_install", "update", line, component=component,
                      package=package, current=int(float(match["current"]) * factor),
                      total=int(float(match["total"]) * factor), unit="bytes")
            else:
                _emit(events, "pip_install", "log", line, component=component)
    if process.wait() != 0:
        raise RuntimeError(f"{label} не удалась: " + " | ".join(tail))


def _replace_intrinsic_source(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) < count:
        raise RuntimeError(f"Не удалось применить совместимую правку IntrinsicAnything: {path.name}")
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def _download_intrinsic_source(progress: Progress, events: Events | None) -> None:
    if (INTRINSIC_ROOT / "inference.py").is_file():
        return
    message = "IntrinsicAnything: загрузка исходного кода"
    progress(message)
    _emit(events, "model_download", "start", message, model="albedo", file="source",
          current=0, total=1, unit="files")
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/zju3dv/IntrinsicAnything/archive/{INTRINSIC_COMMIT}.zip"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = io.BytesIO()
        size = response.headers.get("Content-Length")
        total = int(size) if size and size.isdigit() else None
        while chunk := response.read(1024 * 1024):
            data.write(chunk)
            details = {"model": "albedo", "file": "source", "current": data.tell(),
                       "unit": "bytes"}
            if total is not None:
                details["total"] = total
            _emit(events, "model_download", "update", message, **details)
    with tempfile.TemporaryDirectory(prefix="intrinsic-source-", dir=MODELS_ROOT) as temporary:
        with zipfile.ZipFile(data) as archive:
            archive.extractall(temporary)
        top_levels = {name.split("/", 1)[0] for name in archive.namelist() if "/" in name}
        if len(top_levels) != 1:
            raise RuntimeError("Архив IntrinsicAnything имеет неожиданный формат")
        extracted = Path(temporary) / top_levels.pop()
        shutil.copytree(extracted, INTRINSIC_ROOT, dirs_exist_ok=True)

    _replace_intrinsic_source(
        INTRINSIC_ROOT / "models/ldm/modules/encoders/modules.py",
        "self.model, _ = clip.load(name=model, device=device, jit=jit)",
        "self.model = clip.model.CLIP(768, 224, 24, 1024, 14, 77, 49408, 768, 12, 12)",
        count=2,
    )
    _replace_intrinsic_source(
        INTRINSIC_ROOT / "models/ldm/util.py",
        "from carvekit.api.high import HiInterface",
        "# HiInterface is imported lazily by create_carvekit_interface",
    )
    _replace_intrinsic_source(
        INTRINSIC_ROOT / "models/ldm/util.py",
        "def create_carvekit_interface():\n",
        "def create_carvekit_interface():\n    from carvekit.api.high import HiInterface\n",
    )
    _replace_intrinsic_source(
        INTRINSIC_ROOT / "models/matfusion.py",
        "torch.load(ckpt, map_location='cpu')",
        "torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)",
    )
    _emit(events, "model_download", "done", message, model="albedo", file="source",
          current=1, total=1, unit="files")


def _prepare_intrinsic_runtime(progress: Progress, events: Events | None) -> Path:
    python = INTRINSIC_ROOT / ".venv-intrinsic" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if python.is_file():
        check = subprocess.run(
            [str(python), "-c", (
                "import diffusers, einops, imageio, kornia, loguru, omegaconf, "
                "pytorch_lightning, taming, torch, torchvision, transformers"
            )],
            capture_output=True, text=True, check=False, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if check.returncode == 0:
            return python
    message = "IntrinsicAnything: создание отдельного Python-окружения"
    progress(message)
    _emit(events, "pip_install", "start", message, component="intrinsic-runtime")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(INTRINSIC_ROOT / ".venv-intrinsic")],
        capture_output=True, text=True, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode or not python.is_file():
        raise RuntimeError("Не удалось создать окружение IntrinsicAnything: " +
                           (result.stderr or result.stdout).strip())

    index = _cuda_index()
    _pip_install(
        [str(python), "-m", "pip", "install", "--upgrade", "--progress-bar", "on",
         f"torch=={TORCH_VERSION}", "torchvision==0.21.0", "--index-url",
         f"https://download.pytorch.org/whl/{index}"],
        progress, "Установка PyTorch для IntrinsicAnything", events, "intrinsic-torch",
    )
    packages = [
        "imageio>=2.9", "numpy>=1.26,<3", "Pillow>=10,<12",
        "opencv-python-headless>=4.9,<5", "pytorch-lightning==1.9.5",
        "torchmetrics==0.11.4", "omegaconf>=2.3,<3", "diffusers==0.31.0",
        "transformers>=4.48,<5", "einops>=0.8,<1", "loguru>=0.7,<1",
        "kornia>=0.6,<1", "scipy>=1.12,<2", "matplotlib>=3.8,<4",
        "taming-transformers-rom1504==0.0.6",
        "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
    ]
    _pip_install(
        [str(python), "-m", "pip", "install", "--progress-bar", "on", *packages],
        progress, "Установка библиотек IntrinsicAnything", events, "intrinsic-runtime",
    )
    check = subprocess.run(
        [str(python), "-c", (
            "import diffusers, einops, imageio, kornia, loguru, omegaconf, "
            "pytorch_lightning, taming, torch, torchvision, transformers"
        )],
        capture_output=True, text=True, check=False, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if check.returncode:
        raise RuntimeError("Окружение IntrinsicAnything установлено не полностью: " +
                           (check.stderr or check.stdout).strip())
    _emit(events, "pip_install", "done", "Окружение IntrinsicAnything готово",
          component="intrinsic-runtime")
    return python


def _prepare_intrinsic(progress: Progress, events: Events | None) -> None:
    from huggingface_hub import hf_hub_download

    if model_available("albedo"):
        progress("IntrinsicAnything уже установлена")
        return
    _download_intrinsic_source(progress, events)
    _prepare_intrinsic_runtime(progress, events)
    files = (
        "albedo/configs/albedo_project.yaml",
        "albedo/checkpoints/last.ckpt",
    )
    weights = INTRINSIC_ROOT / "weights"
    for index, filename in enumerate(files, 1):
        message = f"IntrinsicAnything: {Path(filename).name}"
        progress(message)
        _emit(events, "model_download", "start", message, model="albedo", file=filename,
              current=index - 1, total=len(files), unit="files")
        hf_hub_download(
            INTRINSIC_MODEL_REPO, filename, cache_dir=HUGGINGFACE_HUB_CACHE,
            local_dir=weights,
        )
        _emit(events, "model_download", "done", message, model="albedo", file=filename,
              current=index, total=len(files), unit="files")
    if not model_available("albedo"):
        raise RuntimeError("IntrinsicAnything загружена не полностью")


def _ensure_geffnet(progress: Progress, events: Events | None = None) -> None:
    if importlib.util.find_spec("geffnet") is not None:
        _emit(events, "pip_install", "cached", "geffnet уже установлен", component="geffnet")
        return
    message = "Установка библиотеки DSINE (geffnet)..."
    progress(message)
    _emit(events, "pip_install", "start", message, component="geffnet")
    _pip_install([sys.executable, "-m", "pip", "install", "--no-deps",
                  "--progress-bar", "on", "geffnet==1.0.2"],
                 progress, "Установка geffnet", events, "geffnet")
    _emit(events, "pip_install", "done", "geffnet установлен", component="geffnet")


def _model_file(repo: str, filename: str) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        hf_hub_download(
            repo, filename, cache_dir=HUGGINGFACE_HUB_CACHE, local_files_only=True
        )
        return True
    except LocalEntryNotFoundError:
        hf_hub_download(repo, filename, cache_dir=HUGGINGFACE_HUB_CACHE)
        return False


def prepare_environment(models: Iterable[str] = ("depth", "ai"), progress: Progress | None = None,
                        install_cuda: bool = True, events: Events | None = None) -> None:
    """Install CUDA PyTorch when needed, then cache exactly the files inference uses."""
    report = progress or (lambda _message: None)
    selected = set(models)
    if selected - {"depth", "ai", "albedo", "clipseg", "roughness"}:
        raise ValueError("Неизвестная модель для загрузки")
    if install_cuda and selected & {"depth", "ai", "clipseg", "roughness"}:
        ensure_cuda(report, events)
    if "depth" in selected:
        files = ("config.json", "preprocessor_config.json", "model.safetensors")
        for index, filename in enumerate(files, 1):
            message = f"Depth Anything V2: {filename}"
            report(message)
            _emit(events, "model_download", "start", message, model="depth", file=filename,
                  current=index - 1, total=len(files), unit="files")
            cached = _model_file(MODEL_ID, filename)
            _emit(events, "model_download", "cached" if cached else "done", message,
                  model="depth", file=filename, current=index, total=len(files), unit="files")
    if "ai" in selected:
        if install_cuda:
            _ensure_geffnet(report, events)
        report("DSINE: исходный код")
        _emit(events, "model_download", "start", "DSINE: исходный код", model="ai",
              file="source", current=0, total=2, unit="files")
        def source_progress(current: int, total: int | None) -> None:
            details = {"model": "ai", "file": "source", "current": current, "unit": "bytes"}
            if total is not None:
                details["total"] = total
            _emit(events, "model_download", "update", "Загрузка кода DSINE", **details)
            if total and (current == total or current % (5 * 1024 * 1024) < 1024 * 1024):
                report(f"DSINE: {current}/{total} байт")

        _source_root(source_progress)
        _emit(events, "model_download", "done", "DSINE: исходный код", model="ai",
              file="source", current=1, total=2, unit="files")
        report("DSINE: веса dsine.pt")
        _emit(events, "model_download", "start", "DSINE: веса dsine.pt", model="ai",
              file="dsine.pt", current=1, total=2, unit="files")
        cached = _model_file(CHECKPOINT_REPO, "dsine.pt")
        _emit(events, "model_download", "cached" if cached else "done", "DSINE: веса dsine.pt",
              model="ai", file="dsine.pt", current=2, total=2, unit="files")
    if "albedo" in selected:
        _prepare_intrinsic(report, events)
    if "clipseg" in selected:
        for index, filename in enumerate(CLIPSEG_FILES, 1):
            message = f"CLIPSeg: {filename}"
            report(message)
            _emit(events, "model_download", "start", message, model="clipseg",
                  file=filename, current=index - 1, total=len(CLIPSEG_FILES), unit="files")
            cached = _model_file(CLIPSEG_MODEL_ID, filename)
            _emit(events, "model_download", "cached" if cached else "done", message,
                  model="clipseg", file=filename, current=index,
                  total=len(CLIPSEG_FILES), unit="files")
    if "roughness" in selected:
        from smg.supermat_setup import prepare_supermat
        prepare_supermat(report, events)
    report("Подготовка завершена")
    _emit(events, "setup", "done", "Подготовка завершена")
