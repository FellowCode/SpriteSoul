"""Compare CPU/CUDA AO speed and numerical parity using existing maps only."""

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

# Also support running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smg.ao import ao_from_depth
from smg.export import export_ao
from smg.normal import decode_normals
from smg.pipeline import open_depth_png, open_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("generated/ao_gpu_benchmark"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius", type=int, default=24)
    args = parser.parse_args()
    if args.repeats < 1 or args.radius < 1:
        parser.error("repeats and radius must be positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA GPU is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "radius": args.radius, "gpu_repeats": args.repeats, "sprites": {}}
    for source in args.input:
        rgba = open_png(source)
        depth = open_depth_png(args.maps_dir / f"{source.stem}_depth.png", rgba.shape[:2])
        normal = decode_normals(open_png(args.maps_dir / f"{source.stem}_normal.png"))
        kwargs = dict(radius=args.radius, normals=normal)
        started = perf_counter()
        cpu = ao_from_depth(depth, rgba[..., 3], device="cpu", **kwargs)
        cpu_seconds = perf_counter() - started
        print(f"{source.name}: CPU {cpu_seconds:.3f}s", flush=True)

        # First call warms the context/allocator; the benchmark includes all
        # normal validation, component labelling and both transfer directions.
        started = perf_counter()
        ao_from_depth(depth, rgba[..., 3], device="cuda", **kwargs)
        first_gpu_seconds = perf_counter() - started
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            started = perf_counter()
            gpu = ao_from_depth(depth, rgba[..., 3], device="cuda", **kwargs)
            elapsed.append(perf_counter() - started)
        gpu_seconds = statistics.median(elapsed)
        errors = np.abs(cpu - gpu)
        quantized_error = np.abs(np.rint(cpu*255) - np.rint(gpu*255))
        values = {"shape": list(depth.shape), "cpu_seconds": cpu_seconds,
                  "first_gpu_seconds": first_gpu_seconds, "gpu_median_seconds": gpu_seconds,
                  "gpu_seconds": elapsed, "speedup": cpu_seconds / gpu_seconds,
                  "max_abs_error": float(errors.max()), "mean_abs_error": float(errors.mean()),
                  "max_png_difference": int(quantized_error.max()),
                  "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2}
        report["sprites"][source.stem] = values
        export_ao(source, gpu, rgba[..., 3], args.output)
        print(f"{source.name}: CUDA {gpu_seconds:.3f}s, speedup {values['speedup']:.1f}x, "
              f"max error {values['max_abs_error']:.8f}, VRAM {values['peak_gpu_mb']:.0f} MiB", flush=True)
        (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
