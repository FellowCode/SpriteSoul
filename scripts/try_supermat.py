"""Reproducible, standalone SuperMat experiment; does not change the app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "91ffb8edaf259a1e5bbcbcd53b702ad44d70b709"
BASE_REVISION = "bb2154823665391b4fb29b0b9cf82a198964ee05"
BASE_FILES = [
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def download(model_root: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    checkpoint = model_root / "checkpoints" / "supermat.pth"
    remaining = (0 if checkpoint.exists() or checkpoint.with_suffix(".pth.part").exists() else 3_542_000_000) + sum(
        680_821_096 if name.startswith("text_encoder/model") else
        167_335_342 if name.startswith("vae/diffusion") else 2_000_000
        for name in BASE_FILES if not (model_root / "base_model" / name).exists()
    )
    if shutil.disk_usage(model_root).free < remaining + 1_000_000_000:
        raise RuntimeError("Not enough free disk space for the selected weights plus 1 GB reserve")
    print("Downloading single-image SuperMat checkpoint", flush=True)
    api = HfApi()
    model_info = api.model_info("oyiya/SuperMat", revision=MODEL_REVISION, files_metadata=True)
    checkpoint_info = next(item for item in model_info.siblings if item.rfilename == "supermat.pth")
    download_ranges("oyiya/SuperMat", "supermat.pth", MODEL_REVISION, checkpoint,
                    checkpoint_info.size, checkpoint_info.lfs.sha256)
    base_info = api.model_info("sd2-community/stable-diffusion-2-1", revision=BASE_REVISION, files_metadata=True)
    for name in BASE_FILES:
        print(f"Downloading SD 2.1 component: {name}", flush=True)
        item = next(item for item in base_info.siblings if item.rfilename == name)
        if item.size > 10_000_000:
            download_ranges("sd2-community/stable-diffusion-2-1", name, BASE_REVISION,
                            model_root / "base_model" / name, item.size, item.lfs.sha256)
        else:
            hf_hub_download(
                "sd2-community/stable-diffusion-2-1", name, revision=BASE_REVISION,
                local_dir=model_root / "base_model",
            )


def download_ranges(repo: str, name: str, revision: str, target: Path,
                    size: int, expected_hash: str) -> None:
    """Resume bounded HTTP ranges when a full-file/Xet transfer stalls."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import hashlib
    import requests

    if target.exists() and target.stat().st_size == size:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() == expected_hash:
            print(f"Verified existing {target.name}", flush=True)
            return
        raise RuntimeError(f"Existing file failed SHA256 verification: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    progress_file = target.with_suffix(target.suffix + ".parts.json")
    chunk_size = 16 * 1024**2
    total = (size + chunk_size - 1) // chunk_size
    done = set(json.loads(progress_file.read_text()) if progress_file.exists() else [])
    if not partial.exists():
        with partial.open("wb") as handle:
            handle.truncate(size)
        done = set()
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"

    def fetch(index: int) -> int:
        start, stop = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{stop}"},
                                  timeout=(15, 30), stream=True) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{stop}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Unexpected range response for chunk {index}")
                    count = 0
                    with partial.open("r+b") as handle:
                        handle.seek(start)
                        for data in response.iter_content(1024**2):
                            if count + len(data) > stop - start + 1:
                                raise RuntimeError("Range response exceeds requested length")
                            handle.write(data)
                            count += len(data)
                    if count != stop - start + 1:
                        raise RuntimeError("Incomplete range response")
                return index
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, index) for index in range(total) if index not in done]
        for future in as_completed(futures):
            done.add(future.result())
            progress_file.write_text(json.dumps(sorted(done)))
            if len(done) % 10 == 0 or len(done) == total:
                print(f"{target.name}: {len(done)}/{total} chunks ({len(done)/total:.0%}), "
                      f"elapsed {time.perf_counter()-started:.1f}s", flush=True)
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {target.name}")
    partial.replace(target)
    progress_file.unlink()
    print(f"SHA256 verified: {target.name}", flush=True)


def run(model_root: Path, inputs: list[Path], output_root: Path) -> None:
    import importlib.metadata
    import numpy as np
    from PIL import Image
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

    sys.path.insert(0, str(model_root))
    from src.models.supermat_unet_2d_condition import SuperMatUNet2DConditionModel
    from src.pipelines.pipeline_supermat_stable_diffusion import SuperMatStableDiffusionPipeline
    from src.utils import load_rgba_image_as_rgb_tensor, orm_to_roughness_metallic, to_uint8_rgb

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    base = model_root / "base_model"
    dtype = torch.float16
    config = json.loads((base / "unet/config.json").read_text(encoding="utf-8"))
    print("Building SuperMat architecture without downloading unused base UNet weights", flush=True)
    with init_empty_weights():
        unet = SuperMatUNet2DConditionModel.from_config(config)
        unet.replicate(replicate_num=2, shared_blocks_lora=False)
    weights = torch.load(
        model_root / "checkpoints/supermat.pth", map_location="cpu",
        weights_only=True, mmap=True,
    )
    unet.load_state_dict(weights, strict=True, assign=True)
    del weights
    unet = unet.to(dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(
        base / "vae", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base / "text_encoder", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = SuperMatStableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder,
        tokenizer=CLIPTokenizer.from_pretrained(base / "tokenizer", local_files_only=True),
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            base / "scheduler", timestep_spacing="trailing", local_files_only=True,
        ),
        safety_checker=None,
        feature_extractor=CLIPImageProcessor.from_pretrained(base / "feature_extractor", local_files_only=True),
        requires_safety_checker=False,
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "supermat_revision": MODEL_REVISION,
        "base_revision": BASE_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "dtype": "float16",
        "resolution": 512, "steps": 1, "images": [],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("diffusers", "accelerate", "transformers", "huggingface-hub")},
    }
    for path in inputs:
        print(f"Running {path.name}", flush=True)
        source = Image.open(path).convert("RGBA")
        output_dir = output_root / path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        image_tensor = load_rgba_image_as_rgb_tensor(path, 512, torch.device("cuda")).to(dtype)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = pipe(
                prompt="", source_image=image_tensor, num_inference_steps=1,
                output_type="pt", use_fp32_input=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not all(torch.isfinite(value).all().item() for value in prediction):
            raise RuntimeError(f"Non-finite prediction for {path.name}")
        albedo = to_uint8_rgb(prediction[0])
        orm = to_uint8_rgb(prediction[1])
        roughness, metallic = orm_to_roughness_metallic(prediction[1])
        maps = {"albedo": albedo, "orm": orm, "roughness": roughness, "metallic": metallic}
        source.save(output_dir / "source.png")
        for name, values in maps.items():
            raw = Image.fromarray(values)
            raw.save(output_dir / f"{name}_512.png")
            restored = raw.resize(source.size, Image.Resampling.BILINEAR).convert("RGBA")
            restored.putalpha(source.getchannel("A"))
            restored.save(output_dir / f"{path.stem}_{name}.png")
        restored_roughness = np.asarray(Image.open(output_dir / f"{path.stem}_roughness.png"))
        mask = np.asarray(source.getchannel("A")) >= 128
        values = restored_roughness[..., 0][mask].astype(float) / 255
        item = {
            "input": str(path.resolve()), "size": list(source.size),
            "seconds": elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "roughness_percentiles": np.percentile(values, [0, 5, 50, 95, 100]).tolist(),
            "alpha_matches": np.array_equal(restored_roughness[..., 3], np.asarray(source.getchannel("A"))),
        }
        summary["images"].append(item)
        (output_root / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False), flush=True)
        del prediction
    comparison(output_root, [path.stem for path in inputs])


def comparison(output_root: Path, stems: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    cell_size = 384
    margin, label_height = 16, 36
    labels = ["Source", "Roughness", "Metallic", "Albedo"]
    names = ["source", "roughness", "metallic", "albedo"]
    canvas = Image.new("RGB", (4 * cell_size + 5 * margin,
                              len(stems) * (cell_size + label_height + margin) + margin), (32, 35, 40))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    for row, stem in enumerate(stems):
        for col, (label, name) in enumerate(zip(labels, names)):
            x = margin + col * (cell_size + margin)
            y = margin + row * (cell_size + label_height + margin)
            draw.text((x, y), f"{stem} - {label}", fill=(230, 233, 240), font=font)
            file_name = "source.png" if name == "source" else f"{stem}_{name}.png"
            content = Image.open(output_root / stem / file_name).convert("RGBA")
            content = ImageOps.contain(content, (cell_size, cell_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGBA", (cell_size, cell_size), (70, 73, 78, 255))
            for ty in range(0, cell_size, 16):
                for tx in range(0, cell_size, 16):
                    if (tx // 16 + ty // 16) % 2 == 0:
                        ImageDraw.Draw(tile).rectangle((tx, ty, tx + 15, ty + 15), fill=(60, 63, 68, 255))
            tile.alpha_composite(content, ((cell_size - content.width) // 2, (cell_size - content.height) // 2))
            canvas.paste(tile.convert("RGB"), (x, y + label_height))
    canvas.save(output_root / "comparison.png")
    canvas.crop((0, 0, 2 * cell_size + 3 * margin, canvas.height)).save(output_root / "roughness_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/SuperMat")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/supermat")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("inputs", type=Path, nargs="*")
    args = parser.parse_args()
    if args.download_only:
        download(args.model_root)
    else:
        run(args.model_root, args.inputs or [ROOT / "example_sprites/treesV2.png", ROOT / "example_sprites/well.png"], args.output)


if __name__ == "__main__":
    main()
