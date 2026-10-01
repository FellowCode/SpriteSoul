"""Pinned SuperMat sources, shared CUDA runtime, and resumable model downloads."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
import site
import subprocess
import sys
import threading
import urllib.request
import zipfile

from smg.model_paths import SUPERMAT_ROOT
from smg.roughness_ai import (
    BASE_FILES, BASE_REPO, BASE_REVISION, BASE_WEIGHT_SIZES, CHECKPOINT_SIZE,
    MODEL_REPO, MODEL_REVISION, SOURCE_COMMIT, SOURCE_FILES, runtime_python,
)


def download_ranges(repo, name, revision, target, size, expected_hash, progress,
                    chunk_size=16 * 1024**2):
    """Resume checked HTTP ranges without duplicating multi-GB files in HF cache."""
    import requests

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    journal = target.with_suffix(target.suffix + ".parts.json")
    total = (size + chunk_size - 1) // chunk_size
    done = set()
    if partial.exists() and partial.stat().st_size == size and journal.exists():
        try:
            saved = json.loads(journal.read_text(encoding="utf-8"))
            done = {index for index in saved if type(index) is int and 0 <= index < total}
        except (ValueError, TypeError):
            done = set()
    if not partial.exists() or partial.stat().st_size != size:
        with partial.open("wb") as handle:
            handle.truncate(size)
        done = set()

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    cancelled = threading.Event()

    def fetch(index):
        start, stop = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        for attempt in range(4):
            if cancelled.is_set():
                raise RuntimeError("Загрузка SuperMat остановлена")
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{stop}"},
                                  timeout=(15, 30), stream=True) as response:
                    response.raise_for_status()
                    if (response.status_code != 206 or
                            response.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}"):
                        raise RuntimeError("Сервер вернул неверный диапазон загрузки SuperMat")
                    count = 0
                    with partial.open("r+b") as handle:
                        handle.seek(start)
                        for data in response.iter_content(1024**2):
                            if count + len(data) > stop - start + 1:
                                raise RuntimeError("Ответ SuperMat превышает запрошенный диапазон")
                            handle.write(data)
                            count += len(data)
                    if count != stop - start + 1:
                        raise RuntimeError("Загрузка SuperMat прервалась до конца диапазона")
                return index
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                cancelled.wait(attempt + 1)

    def report():
        progress(sum(min(chunk_size, size - index * chunk_size) for index in done), size)

    report()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, index) for index in range(total) if index not in done]
        try:
            for future in as_completed(futures):
                done.add(future.result())
                pending_journal = journal.with_suffix(journal.suffix + ".tmp")
                pending_journal.write_text(json.dumps(sorted(done)), encoding="utf-8")
                pending_journal.replace(journal)
                report()
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        # Retry must re-fetch ranges, rather than trust the corrupted partial file.
        journal.unlink(missing_ok=True)
        raise RuntimeError(f"SuperMat: SHA256 не совпадает для {name}")
    partial.replace(target)
    journal.unlink(missing_ok=True)


def _prepare_source(progress, events):
    from smg.setup import _emit

    if all((SUPERMAT_ROOT / name).is_file() for name in SOURCE_FILES):
        return
    url = f"https://github.com/hyj542682306/SuperMat/archive/{SOURCE_COMMIT}.zip"
    data = io.BytesIO()
    with urllib.request.urlopen(url, timeout=120) as response:
        size = response.headers.get("Content-Length")
        total = int(size) if size and size.isdigit() else None
        while chunk := response.read(1024**2):
            data.write(chunk)
            _emit(events, "model_download", "update", "SuperMat: исходный код",
                  model="roughness", file="source", current=data.tell(), total=total, unit="bytes")
    # Copy only inference code and its license; do not import the multi-view adapter.
    with zipfile.ZipFile(data) as archive:
        prefix = f"SuperMat-{SOURCE_COMMIT}/"
        for item in archive.infolist():
            name = item.filename.removeprefix(prefix)
            if not item.filename.startswith(prefix) or item.is_dir():
                continue
            if not (name.startswith("src/") or name.startswith("LICENSE")):
                continue
            target = (SUPERMAT_ROOT / name).resolve()
            if not target.is_relative_to(SUPERMAT_ROOT.resolve()):
                raise RuntimeError("Некорректный путь в архиве SuperMat")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    if not all((SUPERMAT_ROOT / name).is_file() for name in SOURCE_FILES):
        raise RuntimeError("Архив SuperMat не содержит код модели")


def _prepare_runtime(progress, events):
    from smg.setup import _emit, _pip_install

    python = runtime_python(SUPERMAT_ROOT)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    if not python.is_file():
        result = subprocess.run([sys.executable, "-m", "venv", str(python.parents[1])],
                                capture_output=True, text=True, creationflags=flags, check=False)
        if result.returncode:
            raise RuntimeError("Не удалось создать окружение SuperMat: " + result.stderr.strip())
    packages = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True, creationflags=flags,
    ).strip()
    # Reuse the host's CUDA torch; downloading a second copy wastes several GB.
    (Path(packages) / "sprite_soul_base.pth").write_text(
        "\n".join(site.getsitepackages()) + "\n", encoding="utf-8",
    )
    script = (
        "import importlib.metadata as m; import torch, torchvision, diffusers, accelerate; "
        "from diffusers import AutoencoderKL; from transformers import CLIPTextModel; "
        "assert m.version('diffusers') == '0.31.0'; "
        "assert m.version('accelerate') == '1.1.1'"
    )
    marker = python.parents[1] / "runtime-ready.json"

    def mark_ready():
        marker.write_text(json.dumps({"diffusers": "0.31.0", "accelerate": "1.1.1",
                                      "host_python": sys.executable}), encoding="utf-8")

    def check():
        return subprocess.run([str(python), "-c", script], capture_output=True, text=True,
                              timeout=90, creationflags=flags, check=False)

    if check().returncode == 0:
        mark_ready()
        return
    marker.unlink(missing_ok=True)
    _emit(events, "pip_install", "start", "SuperMat: установка библиотек",
          component="supermat-runtime")
    _pip_install(
        [str(python), "-m", "pip", "install", "--progress-bar", "on",
         "diffusers==0.31.0", "accelerate==1.1.1"],
        progress, "Установка библиотек SuperMat", events, "supermat-runtime",
    )
    vision = subprocess.run([str(python), "-c", "import torchvision"], capture_output=True,
                            text=True, timeout=60, creationflags=flags, check=False)
    if vision.returncode:
        _pip_install(
            [str(python), "-m", "pip", "install", "--no-deps", "--progress-bar", "on",
             "torchvision==0.21.0"],
            progress, "Установка torchvision для SuperMat", events, "supermat-runtime",
        )
    result = check()
    if result.returncode:
        raise RuntimeError("Окружение SuperMat не готово: " + (result.stderr or result.stdout).strip())
    mark_ready()
    _emit(events, "pip_install", "done", "Библиотеки SuperMat готовы",
          component="supermat-runtime")


def prepare_supermat(progress, events=None):
    from huggingface_hub import HfApi, hf_hub_download
    from smg.model_paths import HUGGINGFACE_HUB_CACHE
    from smg.setup import _emit

    SUPERMAT_ROOT.mkdir(parents=True, exist_ok=True)
    files = [(MODEL_REPO, MODEL_REVISION, "supermat.pth", SUPERMAT_ROOT / "checkpoints/supermat.pth")]
    files += [(BASE_REPO, BASE_REVISION, name, SUPERMAT_ROOT / "base_model" / name)
              for name in BASE_FILES]
    total = len(files) + 2
    for index, (name, prepare) in enumerate((("source", _prepare_source),
                                            ("runtime", _prepare_runtime)), 1):
        message = f"SuperMat: {name}"
        progress(message)
        _emit(events, "model_download", "start", message, model="roughness", file=name,
              current=index - 1, total=total, unit="files")
        prepare(progress, events)
        _emit(events, "model_download", "done", message, model="roughness", file=name,
              current=index, total=total, unit="files")
    metadata = {}
    for index, (repo, revision, name, target) in enumerate(files, 3):
        message = f"SuperMat: {name}"
        progress(message)
        _emit(events, "model_download", "start", message, model="roughness", file=name,
              current=index - 1, total=total, unit="files")
        cached = target.is_file() and target.stat().st_size > 0
        # These sizes also reject a truncated pre-existing large checkpoint.
        expected_size = {"supermat.pth": CHECKPOINT_SIZE, **BASE_WEIGHT_SIZES}.get(name)
        if expected_size is not None and cached:
            cached = target.stat().st_size == expected_size
        if not cached:
            if repo not in metadata:
                info = HfApi().model_info(repo, revision=revision, files_metadata=True)
                metadata[repo] = {item.rfilename: item for item in info.siblings}
            item = metadata[repo][name]
            partial = target.with_suffix(target.suffix + ".part")
            needed = 0 if partial.exists() and partial.stat().st_size == item.size else item.size
            if shutil.disk_usage(SUPERMAT_ROOT).free < needed + 512 * 1024**2:
                raise RuntimeError(f"Недостаточно места для SuperMat: {name} ({item.size / 1024**3:.2f} ГиБ)")
            if item.size > 10_000_000:
                def report_bytes(current, size):
                    _emit(events, "model_download", "update", message, model="roughness",
                          file=name, current=current, total=size, unit="bytes")
                download_ranges(repo, name, revision, target, item.size, item.lfs.sha256, report_bytes)
            else:
                # Small configs use the hub client; large weights never get duplicated in its cache.
                hf_hub_download(repo, name, revision=revision, local_dir=SUPERMAT_ROOT / "base_model",
                                cache_dir=HUGGINGFACE_HUB_CACHE)
        _emit(events, "model_download", "cached" if cached else "done", message,
              model="roughness", file=name, current=index, total=total, unit="files")
