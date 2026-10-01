"""Project-local storage paths for every model used by Sprite Soul."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
HUGGINGFACE_ROOT = MODELS_ROOT / "huggingface"
HUGGINGFACE_HUB_CACHE = HUGGINGFACE_ROOT / "hub"
HUGGINGFACE_XET_CACHE = HUGGINGFACE_ROOT / "xet"
INTRINSIC_ROOT = MODELS_ROOT / "IntrinsicAnything"
SUPERMAT_ROOT = MODELS_ROOT / "SuperMat"

# Keep Hugging Face metadata, blobs, and Xet download chunks beside the models.
# Explicit cache_dir arguments still protect model loading if these variables
# were read by another library before Sprite Soul was imported.
os.environ["HF_HOME"] = str(HUGGINGFACE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HUGGINGFACE_HUB_CACHE)
os.environ["HF_XET_CACHE"] = str(HUGGINGFACE_XET_CACHE)
