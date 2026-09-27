"""On-demand DSINE inference using project-local source code and weights."""

import gc
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from smg.normal import dsine_to_opengl
from smg.model_paths import HUGGINGFACE_HUB_CACHE, MODELS_ROOT


# Pin upstream source so its constructor and checkpoint format remain compatible.
DSINE_COMMIT = "ef0c2afa32b4dd19cb8ca4567c652802cd92591c"
CHECKPOINT_REPO = "dylanebert/DSINE"
DSINE_SOURCE_ROOT = MODELS_ROOT / f"DSINE-{DSINE_COMMIT}"


def _source_root(progress=None) -> Path:
    root = DSINE_SOURCE_ROOT
    cache = root.parent
    if not (root / "models" / "dsine" / "v02.py").exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/baegwangbin/DSINE/archive/{DSINE_COMMIT}.zip"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = io.BytesIO()
            size = response.headers.get("Content-Length")
            total = int(size) if size and size.isdigit() else None
            while chunk := response.read(1024 * 1024):
                data.write(chunk)
                if progress:
                    progress(data.tell(), total)
            archive = zipfile.ZipFile(data)
            archive.extractall(cache)
        extracted = cache / f"DSINE-{DSINE_COMMIT}"
        if not (extracted / "models" / "dsine" / "v02.py").exists():
            raise RuntimeError("Не удалось загрузить исходный код DSINE")
    return root


class DSINENormalModel:
    def __init__(self, fov: float = 60.0, max_side: int = 512):
        if not 20 <= fov <= 120:
            raise ValueError("DSINE FOV должен быть от 20 до 120 градусов")
        self.fov = fov
        self.max_side = max_side

    def _load(self):
        import geffnet
        import torch
        from huggingface_hub import hf_hub_download

        source = _source_root()
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from models.dsine.v02 import DSINE_v02

        args = SimpleNamespace(
            NNET_encoder_B=5, NNET_decoder_NF=2048, NNET_decoder_BN=False,
            NNET_decoder_down=8, NNET_learned_upsampling=True,
            NRN_prop_ps=5, NRN_num_iter_train=5, NRN_num_iter_test=5,
            NRN_ray_relu=True, NNET_output_dim=3, NNET_feature_dim=64,
            NNET_hidden_dim=64,
        )
        # The official constructor requests separate ImageNet weights. The DSINE
        # checkpoint replaces every encoder weight, so that download is unnecessary.
        create_model = geffnet.create_model
        def without_pretraining(name, *args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(name, *args, **kwargs)
        geffnet.create_model = without_pretraining
        try:
            model = DSINE_v02(args)
        finally:
            geffnet.create_model = create_model
        checkpoint = hf_hub_download(
            CHECKPOINT_REPO, "dsine.pt", cache_dir=HUGGINGFACE_HUB_CACHE
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        return model.to("cuda").eval()

    def _infer(self, model, rgba: np.ndarray, max_side: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from utils.projection import intrins_from_fov

        h, w = rgba.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        width, height = max(1, round(w * scale)), max(1, round(h * scale))
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        rgb = np.rint(rgba[..., :3] * alpha + 127 * (1 - alpha)).astype(np.uint8)
        if scale < 1:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32) / 255
        left = (32 - width % 32) % 32 // 2
        right = (32 - width % 32) % 32 - left
        top = (32 - height % 32) % 32 // 2
        bottom = (32 - height % 32) % 32 - top
        # A constant black frame creates rectangular seams in DSINE near the
        # aligned 32-pixel boundary. Reflect the real image instead.
        pad_mode = (
            "reflect"
            if width > max(left, right) and height > max(top, bottom)
            else "replicate"
        )
        tensor = F.pad(tensor, (left, right, top, bottom), mode=pad_mode)
        mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[None, :, None, None]
        tensor = (tensor - mean) / std
        intrinsics = intrins_from_fov(self.fov, height, width, device="cuda").unsqueeze(0)
        intrinsics[:, 0, 2] += left
        intrinsics[:, 1, 2] += top
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(tensor, intrins=intrinsics)[-1]
        prediction = prediction[0, :, top:top + height, left:left + width]
        vectors = prediction.float().cpu().permute(1, 2, 0).numpy()
        if scale < 1:
            vectors = cv2.resize(vectors, (w, h), interpolation=cv2.INTER_LINEAR)
        # Cache OpenGL float vectors immediately; all later modes use this basis.
        return dsine_to_opengl(vectors)

    def generate(self, rgba: np.ndarray, progress=None) -> np.ndarray:
        import torch

        if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
            raise ValueError("DSINE ожидает RGBA uint8")
        if not torch.cuda.is_available():
            raise RuntimeError("Для DSINE нужна NVIDIA CUDA")
        model = None
        try:
            if progress:
                progress("Загрузка DSINE...")
            model = self._load()
            if progress:
                progress("Генерация AI Normal...")
            try:
                return self._infer(model, rgba, self.max_side)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if progress:
                    progress("Недостаточно VRAM; повтор с меньшим разрешением...")
                return self._infer(model, rgba, min(self.max_side, 384))
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("Недостаточно VRAM для DSINE. Закройте другие GPU-приложения.") from exc
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
