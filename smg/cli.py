"""Headless command-line entry point for Sprite Soul."""

import argparse
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
    parser.add_argument("input", nargs="+", type=Path, metavar="PNG|SSOUL",
                        help="исходные PNG или сохранённые проекты .ssoul")
    parser.add_argument("-o", "--output", default=Path("generated"), type=Path, metavar="DIR",
                        help="каталог для экспортируемых карт (по умолчанию: ./generated)")
    parser.add_argument("--maps", choices=("both", "depth", "normal", "albedo", "ao", "all"), default="both",
                        help="какие карты записать (по умолчанию: both; all добавляет AO и Albedo)")
    parser.add_argument("--depth-map", type=Path, metavar="PNG",
                        help="готовая серая карта глубины для одного исходного PNG; пропускает Depth AI")
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
    parser.add_argument("--save-project", action="store_true",
                        help="сохранить итоговую глубину в .ssoul для дальнейшего редактирования")
    parser.add_argument("--albedo-debug", type=Path, metavar="DIR",
                        help="сохранить промежуточные карты Albedo в DIR")
    parser.add_argument("--overwrite", action="store_true", help="перезаписать существующие выходные файлы")
    _add_progress_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sprite-soul",
        description="Генерация карт глубины и нормалей; установка CUDA и моделей без UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Примеры:\n"
                "  sprite-soul setup --models all\n"
                "  sprite-soul generate sprite.png\n"
                "  sprite-soul generate sprite.png --progress json\n\n"
                "Подробности: sprite-soul generate --help; sprite-soul setup --help."),
    )
    parser.add_argument("--version", action="version", version="sprite-soul 0.1.0")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    generate = commands.add_parser(
        "generate", help="создать карты из PNG или проекта .ssoul",
        description="Сгенерировать Depth и Normal; при необходимости подготовить AI.",
        epilog="Пример: sprite-soul generate sprite.png --maps normal --ai-smoothing 1.5",
    )
    _add_generate_args(generate)
    setup = commands.add_parser(
        "setup", help="подготовить CUDA и скачать модели",
        description="Установить CUDA-сборку PyTorch и заранее скачать модели.",
        epilog="Пример: sprite-soul setup --models depth --progress json",
    )
    setup.add_argument("--models", choices=("all", "depth", "ai", "clipseg", "none"), default="all",
                       help="какие модели скачать (по умолчанию: all)")
    setup.add_argument("--skip-cuda", action="store_true",
                       help="скачать модели без проверки и установки CUDA")
    _add_progress_args(setup)
    return parser


def _setup_command(args: argparse.Namespace, reporter: Reporter) -> int:
    from smg.setup import prepare_environment

    models = ("depth", "ai", "clipseg") if args.models == "all" else (() if args.models == "none" else (args.models,))
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


def _run_one(path: Path, args: argparse.Namespace, reserved: set[Path],
             reporter: Reporter) -> list[Path]:
    from smg.ao import ao_from_depth
    from smg.depth.inference import DepthModel
    from smg.albedo_ai import generate_albedo
    from smg.crown_normal import compose_crown_normals
    from smg.depth.processing import DepthSettings, process_depth
    from smg.export import export_albedo, export_ao, export_depth, export_normal, load_project, save_project
    from smg.normal import ai_normal, orient_ai_vectors, postprocess_ai_vectors
    from smg.normal_ai import DSINENormalModel
    from smg.pipeline import generate_depth, open_depth_png, open_png
    from smg.segmentation import detect_tree_crown
    from smg.setup import prepare_environment

    if path.suffix.lower() == ".ssoul":
        source_path, base_depth, _, _, foliage_mask = load_project(path, with_foliage_mask=True)
        rgba = open_png(source_path)
    else:
        source_path = path
        rgba = open_png(path)
        base_depth = open_depth_png(args.depth_map, rgba.shape[:2]) if args.depth_map else None
        foliage_mask = None

    output = args.output
    stem = source_path.stem
    wants_depth = args.maps in ("both", "depth", "all")
    wants_normal = args.maps in ("both", "normal", "all")
    wants_albedo = args.maps in ("albedo", "all")
    wants_ao = args.maps in ("ao", "all")
    needs_depth = wants_depth or wants_ao or args.save_project
    paths = []
    if wants_depth:
        paths.append(output / f"{stem}_depth.png")
    if wants_normal:
        paths.append(output / f"{stem}_normal.png")
    if wants_albedo:
        paths.append(output / f"{stem}_albedo.png")
    if wants_ao:
        paths.append(output / f"{stem}_ao.png")
    if args.save_project:
        paths.append(output / f"{stem}.ssoul")
    for target in paths:
        key = target.resolve()
        if key in reserved:
            raise ValueError(f"Повторяющееся имя выходного файла: {target}")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Файл уже существует: {target} (используйте --overwrite)")
    reserved.update(target.resolve() for target in paths)

    def progress(message: str) -> None:
        reporter.runtime("inference", message)

    models = []
    if needs_depth and base_depth is None:
        models.append("depth")
    if wants_normal:
        models.append("ai")
        if foliage_mask is None:
            models.append("clipseg")
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
    if wants_normal:
        if foliage_mask is None:
            foliage_mask = detect_tree_crown(rgba, progress)
        convention = "OpenGL" if args.convention == "opengl" else "DirectX"
        progress("Генерация AI Normal...")
        vectors = DSINENormalModel(args.dsine_fov).generate(rgba, progress)
        if vectors.shape != (*rgba.shape[:2], 3):
            raise ValueError("DSINE вернула неверный размер Normal")
        vectors = orient_ai_vectors(vectors, args.invert_ai_x, args.invert_ai_y)
        vectors = postprocess_ai_vectors(
            vectors, rgba, args.ai_smoothing, args.ai_details,
        )
        if foliage_mask is not None:
            vectors = compose_crown_normals(vectors, foliage_mask, rgba[..., 3])
        normal = ai_normal(vectors, rgba[..., 3], convention)

    if wants_depth:
        export_depth(source_path, depth, rgba[..., 3], output)
    if wants_ao:
        progress("Расчёт AO из Depth...")
        ao = ao_from_depth(depth, rgba[..., 3], args.ao_radius, args.ao_strength)
        export_ao(source_path, ao, rgba[..., 3], output)
    if wants_normal:
        export_normal(source_path, normal, output)
    if wants_albedo:
        progress("Генерация Albedo...")
        albedo = generate_albedo(rgba, progress, debug=args.albedo_debug is not None,
                                 debug_dir=args.albedo_debug)
        export_albedo(source_path, albedo, output)
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
    if args.depth_map and (len(args.input) != 1 or args.input[0].suffix.lower() == ".ssoul"):
        parser.error("--depth-map допустим только с одним исходным PNG")
    if args.albedo_debug and args.maps not in ("albedo", "all"):
        parser.error("--albedo-debug применяется только при экспорте Albedo")
    if args.ao_radius < 1 or args.ao_radius > 128:
        parser.error("--ao-radius должен быть от 1 до 128")
    failed = 0
    processed = 0
    reserved: set[Path] = set()
    try:
        for path in args.input:
            reporter.input = path
            try:
                if path.suffix.lower() not in (".png", ".ssoul"):
                    raise ValueError("Ожидается PNG или .ssoul")
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
