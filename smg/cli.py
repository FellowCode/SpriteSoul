"""Headless command-line entry point for Sprite Soul."""

import argparse
import glob
import json
import sys
from pathlib import Path


def _bounded_float(low: float, high: float):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("ожидается число") from exc
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"значение должно быть от {low} до {high}")
        return number
    return parse


class Reporter:
    def __init__(self, mode: str):
        self.mode = mode
        self.input: Path | None = None

    def emit(self, payload: dict) -> None:
        if self.mode == "json":
            item = {"schema_version": 1, **payload}
            if self.input is not None:
                item.setdefault("input", str(self.input))
            print(json.dumps(item, ensure_ascii=True), flush=True)

    def text(self, message: str) -> None:
        if self.mode == "text":
            prefix = f"[{self.input.name}] " if self.input is not None else ""
            print(prefix + message, file=sys.stderr, flush=True)

    def runtime(self, phase: str, message: str) -> None:
        self.text(message)
        self.emit({"event": "progress", "phase": phase, "status": "update", "message": message})

    def result(self, path: Path) -> None:
        if self.mode == "json":
            self.emit({"event": "result", "path": str(path)})
        else:
            print(path, flush=True)

    def error(self, message: str) -> None:
        if self.mode == "json":
            self.emit({"event": "error", "message": message})
        else:
            prefix = f"[{self.input}] " if self.input is not None else ""
            print(f"Ошибка {prefix}{message}", file=sys.stderr, flush=True)

    def complete(self, failed: int, processed: int) -> None:
        self.emit({"event": "complete", "status": "failed" if failed else "success",
                   "processed": processed, "failed": failed})


def _add_progress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress", choices=("text", "json", "none"), default="text",
                        help="формат прогресса: текст в stderr, JSON Lines в stdout или без прогресса")
    parser.add_argument("--quiet", action="store_true", help="синоним --progress none")


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", nargs="+", type=Path, metavar="PNG|SSOUL|DIR",
                        help="исходные PNG, проекты .ssoul, каталоги или шаблоны (*.png); пакет обрабатывается по моделям")
    parser.add_argument("-o", "--output", default=Path("generated"), type=Path, metavar="DIR",
                        help="каталог для экспортируемых карт (по умолчанию: ./generated)")
    parser.add_argument("--maps", choices=("both", "depth", "normal", "albedo", "ao", "roughness", "all"), default="both",
                        help="какие карты записать (по умолчанию: Depth и Normal; all — все пять)")
    parser.add_argument("--depth-map", type=Path, metavar="PNG|DIR",
                        help="готовая Depth: PNG для одного входа или каталог с <имя>_depth.png для пакета")
    parser.add_argument("--normal-map", type=Path, metavar="PNG|DIR",
                        help="готовая Normal для AO/Normal: PNG или каталог с <имя>_normal.png; ориентация --convention")
    parser.add_argument("--ao-depth-only", action="store_true",
                        help="рассчитать AO только из Depth, без генерации DSINE Normal")
    parser.add_argument("--ao-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="устройство расчёта AO: auto выбирает CUDA при наличии GPU (по умолчанию: auto)")
    parser.add_argument("--normal-source", choices=("ai",), default="ai",
                        help="источник нормалей: только AI/DSINE")
    parser.add_argument("--convention", choices=("opengl", "directx"), default="opengl",
                        help="ориентация зелёного канала Normal (по умолчанию: opengl)")
    parser.add_argument("--invert-depth", action="store_true", help="инвертировать глубину")
    parser.add_argument("--depth-range", type=_bounded_float(0.01, 3), default=1.0,
                        help="множитель диапазона глубины, 0.01–3 (по умолчанию: 1)")
    parser.add_argument("--depth-contrast", type=_bounded_float(0.25, 3), default=1.0,
                        help="контраст глубины, 0.25–3 (по умолчанию: 1)")
    parser.add_argument("--depth-smooth", type=_bounded_float(0, 8), default=0.0,
                        help="сглаживание глубины, sigma 0–8 (по умолчанию: 0)")
    parser.add_argument("--ao-radius", type=int, default=24,
                        help="радиус AO в пикселях, 1–128 (по умолчанию: 24)")
    parser.add_argument("--ao-strength", type=_bounded_float(0, 4), default=2.0,
                        help="сила AO, 0–4 (по умолчанию: 2)")
    parser.add_argument("--ai-smoothing", type=_bounded_float(0, 8), default=1.5,
                        help="сглаживание AI-нормалей, sigma 0–8 (по умолчанию: 1.5)")
    parser.add_argument("--ai-details", type=_bounded_float(0, 1.5), default=0.35,
                        help="детали AI-нормалей из исходника, 0–1.5 (по умолчанию: 0.35)")
    parser.add_argument("--dsine-fov", type=_bounded_float(20, 120), default=60.0,
                        help="поле зрения DSINE, 20–120 градусов (по умолчанию: 60)")
    parser.add_argument("--invert-ai-x", action=argparse.BooleanOptionalAction, default=True,
                        help="инвертировать X у AI нормалей (по умолчанию: включено)")
    parser.add_argument("--invert-ai-y", action=argparse.BooleanOptionalAction, default=True,
                        help="инвертировать Y у AI нормалей (по умолчанию: включено)")
    parser.add_argument("--crown-mode", choices=("auto", "off"), default="auto",
                        help="коррекция нормалей кроны: автоматически или выключена (по умолчанию: auto)")
    parser.add_argument("--crown-threshold", type=_bounded_float(0, 1), default=0.5,
                        help="порог CLIPSeg для кроны, 0–1 (по умолчанию: 0.5)")
    parser.add_argument("--albedo-strength", type=_bounded_float(0, 4), default=1.0,
                        help="сила коррекции Albedo, 0–4 (по умолчанию: 1)")
    parser.add_argument("--albedo-smooth", type=_bounded_float(0, 16), default=2.0,
                        help="сглаживание освещения Albedo, sigma 0–16 (по умолчанию: 2)")
    parser.add_argument("--albedo-shadows", type=_bounded_float(0, 4), default=1.0,
                        help="сила коррекции теней Albedo, 0–4 (по умолчанию: 1)")
    parser.add_argument("--save-project", action=argparse.BooleanOptionalAction, default=False,
                        help="сохранить итоговую глубину и маску кроны в .ssoul (по умолчанию: только экспорт PNG)")
    parser.add_argument("--albedo-debug", type=Path, metavar="DIR",
                        help="сохранить промежуточные карты Albedo в DIR")
    parser.add_argument("--overwrite", action="store_true", help="перезаписать существующие выходные файлы")
    _add_progress_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sprite-soul",
        description="Генерация Depth, Normal, Albedo, AO и Roughness; установка моделей без UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Примеры:\n"
                "  sprite-soul setup --models all\n"
                "  sprite-soul generate sprite.png --maps all -o exported --no-save-project\n"
                "  sprite-soul generate sprite.png --progress json\n\n"
                "  sprite-soul generate sprites/ --maps all -o exported\n\n"
                "Подробности: sprite-soul generate --help; sprite-soul setup --help."),
    )
    parser.add_argument("--version", action="version", version="sprite-soul 0.1.3")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    generate = commands.add_parser(
        "generate", help="создать карты из PNG или проекта .ssoul",
        description="Сгенерировать и сразу экспортировать выбранные карты в PNG одной командой. "
                    "Проект .ssoul по умолчанию не сохраняется; при необходимости модели готовятся автоматически.",
        epilog="Пример: sprite-soul generate sprite.png --maps all -o exported --no-save-project",
    )
    _add_generate_args(generate)
    setup = commands.add_parser(
        "setup", help="подготовить CUDA и скачать модели",
        description="Установить CUDA-сборку PyTorch и заранее скачать модели.",
        epilog="Пример: sprite-soul setup --models depth --progress json",
    )
    setup.add_argument("--models", choices=("all", "depth", "ai", "clipseg", "albedo", "roughness", "none"), default="all",
                       help="какие модели скачать (по умолчанию: all)")
    setup.add_argument("--skip-cuda", action="store_true",
                       help="скачать модели без проверки и установки CUDA")
    _add_progress_args(setup)
    return parser


def _setup_command(args: argparse.Namespace, reporter: Reporter) -> int:
    from smg.setup import prepare_environment

    models = ("depth", "ai", "clipseg", "albedo", "roughness") if args.models == "all" else (() if args.models == "none" else (args.models,))
    try:
        prepare_environment(models, reporter.text, install_cuda=not args.skip_cuda,
                            events=reporter.emit)
    except KeyboardInterrupt:
        reporter.error("Прервано пользователем")
        reporter.complete(1, 0)
        return 130
    except Exception as exc:
        reporter.error(str(exc))
        reporter.complete(1, 0)
        return 1
    reporter.complete(0, 0)
    return 0


def _read_input(path: Path, args: argparse.Namespace):
    from smg.export import load_project
    from smg.pipeline import open_depth_png, open_png

    if path.is_dir():
        raise ValueError(f"В каталоге нет PNG или .ssoul: {path}")
    if not path.exists() and glob.has_magic(str(path)):
        raise ValueError(f"Нет файлов по шаблону: {path}")
    if path.suffix.lower() not in (".png", ".ssoul"):
        raise ValueError("Ожидается PNG или .ssoul")
    if path.suffix.lower() == ".ssoul":
        source_path, base_depth, _, _, foliage_mask = load_project(path, with_foliage_mask=True)
    else:
        source_path, base_depth, foliage_mask = path, None, None
    rgba = open_png(source_path)
    if args.depth_map:
        depth_path = (args.depth_map / f"{source_path.stem}_depth.png"
                      if args.depth_map.is_dir() else args.depth_map)
        base_depth = open_depth_png(depth_path, rgba.shape[:2])
    return source_path, rgba, base_depth, foliage_mask


def _target_paths(source_path: Path, args: argparse.Namespace, reserved: set[Path]) -> list[Path]:
    kinds = {"both": ("depth", "normal"), "all": ("depth", "normal", "albedo", "ao", "roughness")}
    paths = [args.output / f"{source_path.stem}_{kind}.png"
             for kind in kinds.get(args.maps, (args.maps,))]
    if args.save_project:
        paths.append(args.output / f"{source_path.stem}.ssoul")
    for target in paths:
        if target.resolve() in reserved:
            raise ValueError(f"Повторяющееся имя выходного файла: {target}")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Файл уже существует: {target} (используйте --overwrite)")
    reserved.update(target.resolve() for target in paths)
    return paths


def _read_normal_input(source_path: Path, rgba, args: argparse.Namespace):
    from smg.normal import decode_normals
    from smg.pipeline import open_png

    if args.normal_map is None:
        return None
    path = (args.normal_map / f"{source_path.stem}_normal.png"
            if args.normal_map.is_dir() else args.normal_map)
    pixels = open_png(path)
    if pixels.shape[:2] != rgba.shape[:2]:
        raise ValueError(f"Normal {pixels.shape[1]}x{pixels.shape[0]} не совпадает с размером спрайта")
    return decode_normals(pixels, "OpenGL" if args.convention == "opengl" else "DirectX")


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    expanded = []
    for path in inputs:
        if path.is_dir():
            matches = sorted((item for item in path.iterdir()
                              if item.is_file() and item.suffix.lower() in (".png", ".ssoul")),
                             key=lambda item: item.name.lower())
        elif not path.exists() and glob.has_magic(str(path)):
            matches = [Path(item) for item in sorted(glob.glob(str(path)))]
        else:
            matches = [path]
        expanded.extend(matches or [path])
    return expanded


def _run_one(path: Path, args: argparse.Namespace, reserved: set[Path],
             reporter: Reporter) -> list[Path]:
    from smg.ao import ao_from_depth
    from smg.depth.inference import DepthModel
    from smg.albedo_ai import generate_albedo
    from smg.crown_normal import compose_crown_normals
    from smg.depth.processing import DepthSettings, process_depth
    from smg.export import export_albedo, export_ao, export_depth, export_normal, export_roughness, save_project
    from smg.roughness_ai import generate_roughness
    from smg.normal import ai_normal, orient_ai_vectors, postprocess_ai_vectors
    from smg.normal_ai import DSINENormalModel
    from smg.pipeline import generate_depth
    from smg.segmentation import detect_tree_crown
    from smg.setup import prepare_environment

    source_path, rgba, base_depth, foliage_mask = _read_input(path, args)
    output = args.output
    stem = source_path.stem
    wants_depth = args.maps in ("both", "depth", "all")
    wants_normal = args.maps in ("both", "normal", "all")
    wants_albedo = args.maps in ("albedo", "all")
    wants_ao = args.maps in ("ao", "all")
    wants_roughness = args.maps in ("roughness", "all")
    needs_depth = wants_depth or wants_ao or args.save_project
    needs_normal = wants_normal or (wants_ao and not args.ao_depth_only)
    vectors = _read_normal_input(source_path, rgba, args)
    paths = _target_paths(source_path, args, reserved)

    def progress(message: str) -> None:
        reporter.runtime("inference", message)

    models = []
    if needs_depth and base_depth is None:
        models.append("depth")
    if needs_normal and vectors is None:
        models.append("ai")
        if args.crown_mode == "auto" and foliage_mask is None:
            models.append("clipseg")
    if wants_albedo:
        models.append("albedo")
    if wants_roughness:
        models.append("roughness")
    if models:
        prepare_environment(models, reporter.text, events=reporter.emit)

    depth = None
    if needs_depth:
        if base_depth is None:
            progress("Генерация Depth...")
            base_depth = generate_depth(rgba, DepthModel(), progress)
        depth = process_depth(base_depth, rgba[..., 3], DepthSettings(
            args.invert_depth, args.depth_range, args.depth_contrast, args.depth_smooth,
        ))

    normal = None
    if needs_normal and vectors is None:
        if args.crown_mode == "auto" and foliage_mask is None:
            if args.crown_threshold == 0.5:
                foliage_mask = detect_tree_crown(rgba, progress)
            else:
                foliage_mask = detect_tree_crown(rgba, progress, threshold=args.crown_threshold)
        progress("Генерация AI Normal...")
        vectors = DSINENormalModel(args.dsine_fov).generate(rgba, progress)
        if vectors.shape != (*rgba.shape[:2], 3):
            raise ValueError("DSINE вернула неверный размер Normal")
        vectors = orient_ai_vectors(vectors, args.invert_ai_x, args.invert_ai_y)
        vectors = postprocess_ai_vectors(
            vectors, rgba, args.ai_smoothing, args.ai_details,
        )
        if args.crown_mode == "auto" and foliage_mask is not None:
            vectors = compose_crown_normals(vectors, foliage_mask, rgba[..., 3])
    if wants_normal:
        normal = ai_normal(vectors, rgba[..., 3], "OpenGL" if args.convention == "opengl" else "DirectX")

    if wants_depth:
        export_depth(source_path, depth, rgba[..., 3], output)
    if wants_ao:
        progress("Расчёт AO из Depth и Normal..." if vectors is not None else "Расчёт AO из Depth...")
        ao = ao_from_depth(depth, rgba[..., 3], args.ao_radius, args.ao_strength,
                           normals=None if args.ao_depth_only else vectors, device=args.ao_device)
        export_ao(source_path, ao, rgba[..., 3], output)
    if wants_normal:
        export_normal(source_path, normal, output)
    if wants_albedo:
        progress("Генерация Albedo...")
        albedo = generate_albedo(
            rgba, progress, strength=args.albedo_strength,
            illumination_sigma=args.albedo_smooth, shadow_strength=args.albedo_shadows,
            debug=args.albedo_debug is not None, debug_dir=args.albedo_debug,
        )
        export_albedo(source_path, albedo, output)
    if wants_roughness:
        progress("Генерация Roughness...")
        roughness = generate_roughness(rgba, progress)
        export_roughness(source_path, roughness, rgba[..., 3], output)
    if args.save_project:
        output.mkdir(parents=True, exist_ok=True)
        save_project(output / f"{stem}.ssoul", source_path, depth,
                     20.0, "OpenGL" if args.convention == "opengl" else "DirectX",
                     foliage_mask=foliage_mask)
    return paths


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    if argv[0] not in ("generate", "setup", "-h", "--help", "--version"):
        argv.insert(0, "generate")  # Compatibility with the original positional CLI.
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.quiet and args.progress != "text":
        parser.error("--quiet нельзя сочетать с --progress")
    reporter = Reporter("none" if args.quiet else args.progress)
    if args.command == "setup":
        if args.skip_cuda and args.models == "none":
            parser.error("--skip-cuda с --models none ничего не делает")
        return _setup_command(args, reporter)
    if args.depth_map and args.maps not in ("both", "depth", "ao", "all") and not args.save_project:
        parser.error("--depth-map применяется только при генерации Depth, AO или сохранении проекта")
    args.input = _expand_inputs(args.input)
    if args.normal_map and args.maps not in ("both", "normal", "ao", "all"):
        parser.error("--normal-map применяется только при генерации Normal или AO")
    if args.normal_map and len(args.input) > 1 and not args.normal_map.is_dir():
        parser.error("для нескольких входов --normal-map должен указывать каталог с <имя>_normal.png")
    if args.ao_depth_only and (args.maps not in ("ao", "all") or args.normal_map):
        parser.error("--ao-depth-only применяется к AO и несовместим с --normal-map")
    if args.depth_map and len(args.input) > 1 and not args.depth_map.is_dir():
        parser.error("для нескольких входов --depth-map должен указывать каталог с <имя>_depth.png")
    if args.albedo_debug and args.maps not in ("albedo", "all"):
        parser.error("--albedo-debug применяется только при экспорте Albedo")
    if args.ao_radius < 1 or args.ao_radius > 128:
        parser.error("--ao-radius должен быть от 1 до 128")
    if len(args.input) > 1:
        from smg.batch import run_batch

        return run_batch(args, reporter)
    failed = 0
    processed = 0
    reserved: set[Path] = set()
    try:
        for path in args.input:
            reporter.input = path
            try:
                written = _run_one(path, args, reserved, reporter)
                for target in written:
                    reporter.result(target)
                processed += 1
            except Exception as exc:
                # Model downloads and third-party inference can raise their own error types.
                reporter.error(str(exc))
                failed += 1
    except KeyboardInterrupt:
        reporter.error("Прервано пользователем")
        reporter.complete(failed + 1, processed)
        return 130
    reporter.input = None
    reporter.complete(failed, processed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
