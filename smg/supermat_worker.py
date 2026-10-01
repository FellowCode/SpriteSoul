"""Run inside .venv-supermat; deliberately independent of app imports."""

import json
from pathlib import Path
import sys


def run(root: Path, source: Path, output: Path) -> None:
    import numpy as np
    from PIL import Image
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

    sys.path.insert(0, str(root))
    from src.models.supermat_unet_2d_condition import SuperMatUNet2DConditionModel
    from src.pipelines.pipeline_supermat_stable_diffusion import SuperMatStableDiffusionPipeline
    from src.utils import load_rgba_image_as_rgb_tensor

    if not torch.cuda.is_available():
        raise RuntimeError("SuperMat требует доступную CUDA-видеокарту NVIDIA")
    base = root / "base_model"
    dtype = torch.float16
    print("SuperMat: загрузка весов…", flush=True)
    config = json.loads((base / "unet/config.json").read_text(encoding="utf-8"))
    with init_empty_weights():
        unet = SuperMatUNet2DConditionModel.from_config(config)
        unet.replicate(replicate_num=2, shared_blocks_lora=False)
    weights = torch.load(root / "checkpoints/supermat.pth", map_location="cpu",
                         weights_only=True, mmap=True)
    unet.load_state_dict(weights, strict=True, assign=True)
    del weights
    unet = unet.to(dtype=dtype).eval()
    pipe = SuperMatStableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(base / "vae", variant="fp16",
                                         torch_dtype=dtype, local_files_only=True),
        text_encoder=CLIPTextModel.from_pretrained(base / "text_encoder", variant="fp16",
                                                 torch_dtype=dtype, local_files_only=True),
        tokenizer=CLIPTokenizer.from_pretrained(base / "tokenizer", local_files_only=True),
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(base / "scheduler", timestep_spacing="trailing",
                                                local_files_only=True),
        safety_checker=None, requires_safety_checker=False,
        feature_extractor=CLIPImageProcessor.from_pretrained(base / "feature_extractor",
                                                            local_files_only=True),
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    image = load_rgba_image_as_rgb_tensor(source, 512, torch.device("cuda")).to(dtype)
    print("SuperMat: Roughness, 512×512, один проход…", flush=True)
    with torch.inference_mode():
        prediction = pipe(prompt="", source_image=image, num_inference_steps=1,
                          output_type="pt", use_fp32_input=False)
    orm = prediction[1]
    if tuple(orm.shape) != (1, 3, 512, 512) or not torch.isfinite(orm).all().item():
        raise RuntimeError("SuperMat вернула некорректное предсказание ORM")
    # ORM green is linear roughness, with no gamma or per-image normalization.
    roughness = orm[0, 1].detach().float().cpu().clamp(0, 1).numpy()
    with Image.open(source) as original:
        restored = Image.fromarray(roughness).resize(original.size, Image.Resampling.BILINEAR)
    np.save(output, np.asarray(restored, np.float32), allow_pickle=False)
    print("SuperMat: Roughness готова", flush=True)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
