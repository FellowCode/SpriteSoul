"""Try a public CHORD demo on sprites and compare with SuperMat."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SPACE = "jasong/chord-minecraft-pbr"
MAP_NAMES = ("albedo", "normal", "roughness", "metallic", "relit")


def comparison(output_root: Path, stems: list[str]) -> None:
    cell, gap, label_height = 384, 16, 36
    columns = ("Source", "SuperMat Roughness (512)", "CHORD Roughness (1024)")
    board = Image.new("RGB", (3 * cell + 4 * gap,
                             len(stems) * (cell + gap + label_height) + gap), (32, 35, 40))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
    for row, stem in enumerate(stems):
        files = [output_root / stem / "source.png",
                 ROOT / "generated/supermat" / stem / f"{stem}_roughness.png",
                 output_root / stem / f"{stem}_roughness.png"]
        for col, (label, path) in enumerate(zip(columns, files)):
            x, y = gap + col * (cell + gap), gap + row * (cell + gap + label_height)
            draw.text((x, y), f"{stem} - {label}", font=font, fill=(230, 233, 240))
            tile = Image.new("RGBA", (cell, cell), (70, 73, 78, 255))
            tile_draw = ImageDraw.Draw(tile)
            for ty in range(0, cell, 16):
                for tx in range(0, cell, 16):
                    if (tx // 16 + ty // 16) % 2 == 0:
                        tile_draw.rectangle((tx, ty, tx + 15, ty + 15), fill=(60, 63, 68, 255))
            if path.exists():
                content = ImageOps.contain(Image.open(path).convert("RGBA"), (cell, cell), Image.Resampling.LANCZOS)
                tile.alpha_composite(content, ((cell - content.width) // 2, (cell - content.height) // 2))
            board.paste(tile.convert("RGB"), (x, y + label_height))
    board.save(output_root / "roughness_comparison.png")


def run(inputs: list[Path], output_root: Path, space: str) -> None:
    from gradio_client import Client, handle_file
    from huggingface_hub import HfApi

    output_root.mkdir(parents=True, exist_ok=True)
    client = Client(space, download_files=str((output_root / "demo_downloads").resolve()))
    summary = {
        "execution": "public Hugging Face demo; not local inference",
        "space": space, "space_revision": HfApi().space_info(space).sha,
        "model_resolution": 1024, "background_rgb": [128, 128, 128],
        "gradio_client": importlib.metadata.version("gradio-client"),
        "images": [],
    }
    for path in inputs:
        source = Image.open(path).convert("RGBA")
        target = output_root / path.stem
        target.mkdir(parents=True, exist_ok=True)
        source.save(target / "source.png")
        source_array = np.asarray(source)
        alpha = source_array[..., 3:4].astype(np.float32) / 255
        rgb = np.rint(source_array[..., :3] * alpha + 128 * (1 - alpha)).astype(np.uint8)
        input_file = target / "input_gray.png"
        Image.fromarray(rgb).save(input_file)
        print(f"Submitting {path.name} to {space}", flush=True)
        started = time.perf_counter()
        options = {}
        if space == "jasong/chord-minecraft-pbr":
            options = {"output_format": "bedrock", "seamless": False,
                       "include_height": False, "compute_porosity": False,
                       "compute_sss": False, "compute_emission": False,
                       "hardcoded_metal": "none"}
        job = client.submit(handle_file(str(input_file.resolve())), api_name="/inference", **options)
        last_status = None
        while not job.done():
            status = job.status()
            current = (str(status.code), status.rank)
            if current != last_status:
                print(f"{path.name}: status={status.code}, queue_rank={status.rank}", flush=True)
                last_status = current
            time.sleep(2)
        result = job.result()
        elapsed = time.perf_counter() - started
        names = ("albedo", "mer", "normal_directx", "relit") if options else MAP_NAMES
        if len(result) != len(names):
            raise RuntimeError("Unexpected number of maps from the CHORD demo")
        formats = {}
        maps = {}
        for name, value in zip(names, result):
            raw_path = Path(value["path"] if isinstance(value, dict) else value)
            raw = Image.open(raw_path)
            if raw.size != source.size:
                raise RuntimeError(f"Unexpected {name} dimensions: {raw.size}, expected {source.size}")
            formats[name] = {"format": raw.format, "mode": raw.mode}
            shutil.copy2(raw_path, target / f"{name}_demo{raw_path.suffix}")
            maps[name] = raw.copy()
        if options:
            maps["roughness"] = maps["mer"].getchannel("B").convert("RGB")
            maps["metallic"] = maps["mer"].getchannel("R").convert("RGB")
            formats["roughness"] = {"format": "PNG", "mode": "RGB", "extracted_from": "MER blue"}
            formats["metallic"] = {"format": "PNG", "mode": "RGB", "extracted_from": "MER red"}
            maps["roughness"].save(target / "roughness_demo.png")
            maps["metallic"].save(target / "metallic_demo.png")
        for name, raw in maps.items():
            restored = raw.convert("RGBA")
            restored.putalpha(source.getchannel("A"))
            restored.save(target / f"{path.stem}_{name}.png")
            if not np.array_equal(np.asarray(restored)[..., 3], source_array[..., 3]):
                raise RuntimeError(f"Alpha mismatch for {name}")
        roughness = np.asarray(Image.open(target / f"{path.stem}_roughness.png"))[..., 0]
        valid = source_array[..., 3] >= 128
        item = {
            "input": str(path.resolve()), "size": list(source.size),
            "request_seconds_including_queue_network_and_downloads": elapsed,
            "roughness_percentiles": np.percentile(roughness[valid] / 255, [0, 5, 50, 95, 100]).tolist(),
            "alpha_matches_all_maps": True, "received_formats": formats,
            "api_options": options,
        }
        summary["images"].append(item)
        (output_root / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False), flush=True)
    comparison(output_root, [path.stem for path in inputs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/chord")
    parser.add_argument("--space", default=SPACE, choices=[SPACE, "ksangk/chord-demo"])
    parser.add_argument("inputs", nargs="*", type=Path)
    args = parser.parse_args()
    run(args.inputs or [ROOT / "example_sprites/treesV2.png", ROOT / "example_sprites/well.png"], args.output, args.space)


if __name__ == "__main__":
    main()
