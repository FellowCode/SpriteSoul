"""CLI map-major processing, with one resident model and disk-backed sprite state."""

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from smg.cli import _read_input, _target_paths


@dataclass
class Job:
    input: Path
    source: Path
    cache: Path
    needs_depth_model: bool
    needs_crown: bool
    failed: bool = False


def run_batch(args, reporter) -> int:
    from smg import albedo_ai, roughness_ai
    from smg.ao import ao_from_depth
    from smg.crown_normal import compose_crown_normals
    from smg.depth.inference import DepthModel
    from smg.depth.processing import DepthSettings, process_depth
    from smg.export import export_albedo, export_ao, export_depth, export_normal, export_roughness, save_project
    from smg.model_session import ModelSession
    from smg.normal import ai_normal, orient_ai_vectors, postprocess_ai_vectors
    from smg.normal_ai import DSINENormalModel
    from smg.pipeline import generate_depth
    from smg.segmentation import CrownModel, tree_crown_mask
    from smg.setup import prepare_environment

    kinds = {"both": ("depth", "normal"), "all": ("depth", "normal", "albedo", "ao", "roughness")}
    selected = kinds.get(args.maps, (args.maps,))
    needs_depth = "depth" in selected or "ao" in selected or args.save_project
    convention = "OpenGL" if args.convention == "opengl" else "DirectX"
    jobs = []
    failed = processed = 0
    reserved = set()
    generated = dict.fromkeys(selected, 0)
    total = len(args.input)

    def map_progress(kind):
        current = generated[kind]
        label = {"depth": "Depth", "normal": "Normal", "albedo": "Albedo",
                 "ao": "AO", "roughness": "Roughness"}[kind]
        message = f"{label}: сгенерировано {current} из {total} карт"
        reporter.text(message)
        reporter.emit({"event": "progress", "phase": "batch", "status": "update",
                       "message": message, "map": kind, "current": current,
                       "total": total, "unit": "maps"})

    def progress(message):
        reporter.runtime("inference", message)

    def fail(job, exc):
        nonlocal failed
        reporter.input = job.input
        reporter.error(str(exc))
        job.failed = True
        failed += 1

    def stage(name, operation, factory=None, predicate=lambda job: True, persist=False):
        active = [job for job in jobs if not job.failed and predicate(job)]
        if not active:
            return
        resource = None
        reporter.input = None
        reporter.runtime("batch", f"{name}: проход по {len(active)} спрайтам")
        try:
            if factory:
                resource = factory()
            for job in active:
                reporter.input = job.input
                try:
                    with np.load(job.cache, allow_pickle=False) as cached:
                        state = {key: cached[key] for key in cached.files}
                    operation(job, state, resource)
                    if persist:
                        np.savez(job.cache, **state)
                except Exception as exc:
                    fail(job, exc)
        except Exception as exc:
            for job in active:
                if not job.failed:
                    fail(job, exc)
        finally:
            if resource is not None:
                if hasattr(resource, "unload"):
                    resource.unload()
                else:
                    resource.close()
                reporter.input = None
                reporter.runtime("batch", f"{name}: модель выгружена")

    def write(job, kind, values, alpha=None):
        exporters = {"depth": export_depth, "normal": export_normal,
                     "albedo": export_albedo, "ao": export_ao, "roughness": export_roughness}
        exporter = exporters[kind]
        if alpha is None:
            target = exporter(job.source, values, args.output)
        else:
            target = exporter(job.source, values, alpha, args.output)
        reporter.result(target)
        generated[kind] += 1
        map_progress(kind)

    def depth_pass(job, state, model):
        rgba = state["rgba"]
        base = state.get("depth")
        if base is None:
            progress("Генерация Depth...")
            base = generate_depth(rgba, model, progress)
        depth = process_depth(base, rgba[..., 3], DepthSettings(
            args.invert_depth, args.depth_range, args.depth_contrast, args.depth_smooth,
        ))
        if "depth" in selected:
            write(job, "depth", depth, rgba[..., 3])
        if "ao" in selected:
            progress("Расчёт AO из Depth...")
            write(job, "ao", ao_from_depth(depth, rgba[..., 3], args.ao_radius, args.ao_strength), rgba[..., 3])
        if args.save_project:
            state["depth"] = depth
        else:
            state.pop("depth", None)

    def crown_pass(job, state, model):
        rgba = state["rgba"]
        if np.any(rgba[..., 3]):
            mask = tree_crown_mask(model.predict(rgba, progress), rgba, args.crown_threshold)
            if mask is not None:
                state["foliage"] = mask
            progress("Дерево: крона найдена" if mask is not None else "Крона меньше 40%: обычная Normal")

    def normal_pass(job, state, model):
        rgba = state["rgba"]
        vectors = model.generate(rgba, progress)
        if vectors.shape != (*rgba.shape[:2], 3):
            raise ValueError("DSINE вернула неверный размер Normal")
        vectors = orient_ai_vectors(vectors, args.invert_ai_x, args.invert_ai_y)
        vectors = postprocess_ai_vectors(vectors, rgba, args.ai_smoothing, args.ai_details)
        if args.crown_mode == "auto" and "foliage" in state:
            vectors = compose_crown_normals(vectors, state["foliage"], rgba[..., 3])
        write(job, "normal", ai_normal(vectors, rgba[..., 3], convention))

    def albedo_session():
        root, python = albedo_ai._experiment()
        return ModelSession(python, Path(albedo_ai.__file__).with_name("intrinsic_worker.py"), root)

    def albedo_pass(job, state, session):
        debug_dir = args.albedo_debug / job.source.stem if args.albedo_debug else None
        albedo = albedo_ai.generate_albedo(
            state["rgba"], progress, strength=args.albedo_strength,
            illumination_sigma=args.albedo_smooth, shadow_strength=args.albedo_shadows,
            debug=debug_dir is not None, debug_dir=debug_dir, session=session,
        )
        write(job, "albedo", albedo)

    def roughness_session():
        root = roughness_ai.SUPERMAT_ROOT
        return ModelSession(roughness_ai.runtime_python(root),
                            Path(roughness_ai.__file__).with_name("supermat_worker.py"), root)

    def roughness_pass(job, state, session):
        rgba = state["rgba"]
        write(job, "roughness", roughness_ai.generate_roughness(rgba, progress, session=session), rgba[..., 3])

    def finish(job, state, _resource):
        nonlocal processed
        if args.save_project:
            args.output.mkdir(parents=True, exist_ok=True)
            target = args.output / f"{job.source.stem}.ssoul"
            save_project(target, job.source, state["depth"], 20.0, convention,
                         foliage_mask=state.get("foliage"))
            reporter.result(target)
        processed += 1

    try:
        reporter.input = None
        for kind in selected:
            map_progress(kind)
        with tempfile.TemporaryDirectory(prefix="sprite-soul-batch-") as temporary:
            for index, path in enumerate(args.input):
                reporter.input = path
                try:
                    source, rgba, depth, foliage = _read_input(path, args)
                    _target_paths(source, args, reserved)
                    cache = Path(temporary) / f"{index}.npz"
                    state = {"rgba": rgba}
                    if depth is not None:
                        state["depth"] = depth
                    if foliage is not None:
                        state["foliage"] = foliage
                    np.savez(cache, **state)
                    jobs.append(Job(path, source, cache, needs_depth and depth is None,
                                    "normal" in selected and args.crown_mode == "auto" and foliage is None))
                except Exception as exc:
                    reporter.error(str(exc))
                    failed += 1
            # Do not retain the last preflight image in RAM throughout the batch.
            source = rgba = depth = foliage = state = None
            models = []
            if any(job.needs_depth_model for job in jobs):
                models.append("depth")
            if jobs and "normal" in selected:
                models.append("ai")
                if any(job.needs_crown for job in jobs):
                    models.append("clipseg")
            models.extend(kind for kind in ("albedo", "roughness") if jobs and kind in selected)
            if models:
                reporter.input = None
                try:
                    prepare_environment(models, reporter.text, events=reporter.emit)
                except Exception as exc:
                    for job in jobs:
                        fail(job, exc)
            if needs_depth:
                stage("Depth / AO", depth_pass,
                      (lambda: DepthModel(keep_loaded=True)) if "depth" in models else None,
                      persist=args.save_project)
            stage("CLIPSeg", crown_pass, CrownModel, lambda job: job.needs_crown, persist=True)
            if "normal" in selected:
                stage("Normal", normal_pass, lambda: DSINENormalModel(args.dsine_fov, keep_loaded=True))
            if "albedo" in selected:
                stage("Albedo", albedo_pass, albedo_session)
            if "roughness" in selected:
                stage("Roughness", roughness_pass, roughness_session)
            stage("Завершение пакета", finish)
    except KeyboardInterrupt:
        reporter.error("Прервано пользователем")
        reporter.input = None
        reporter.complete(failed + 1, processed)
        return 130
    reporter.input = None
    reporter.complete(failed, processed)
    return 1 if failed else 0
