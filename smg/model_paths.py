"""Project-local storage paths for every model used by Sprite Soul."""

import os
import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
HUGGINGFACE_ROOT = MODELS_ROOT / "huggingface"
HUGGINGFACE_HUB_CACHE = HUGGINGFACE_ROOT / "hub"
HUGGINGFACE_XET_CACHE = HUGGINGFACE_ROOT / "xet"
INTRINSIC_ROOT = MODELS_ROOT / "IntrinsicAnything"
SUPERMAT_ROOT = MODELS_ROOT / "SuperMat"


def intrinsic_environment(root: Path) -> Path:
    """Keep torch's long header filenames below Windows MAX_PATH."""
    local = root / ".venv-intrinsic"
    header = (local / "Lib/site-packages/torch/include/ATen/ops" /
              "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_compositeexplicitautograd_dispatch.h")
    if sys.platform != "win32" or len(str(header.resolve())) < 260:
        return local
    cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    identity = hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()[:12]
    return cache / "ss-venvs" / identity


def intrinsic_python(root: Path) -> Path:
    return intrinsic_environment(root) / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )

# Keep Hugging Face metadata, blobs, and Xet download chunks beside the models.
# Explicit cache_dir arguments still protect model loading if these variables
# were read by another library before Sprite Soul was imported.
os.environ["HF_HOME"] = str(HUGGINGFACE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HUGGINGFACE_HUB_CACHE)
os.environ["HF_XET_CACHE"] = str(HUGGINGFACE_XET_CACHE)
