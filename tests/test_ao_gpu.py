"""CUDA/CPU parity, bounded chunking, and optional-device fallbacks."""

import numpy as np
import pytest

from smg.ao import ao_from_depth
from smg.normal import normalize_vectors


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is unavailable")
    return torch


def scene(shape=(47, 61)):
    y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    relief = -3 * np.exp(-((x-24)**2 + (y-22)**2) / 18)
    gy, gx = np.gradient(relief)
    normals = normalize_vectors(np.dstack((-gx, gy, np.ones_like(gx))))
    depth = np.full(shape, 0.5, np.float32)
    depth[12:32, 33:43] = 0.8
    alpha = np.full(shape, 255, np.uint8)
    alpha[:3] = 0
    alpha[-3:] = 0
    alpha[:, 45:48] = 0  # Separate atlas components.
    alpha[5:8] = 64
    depth[alpha == 0] = np.nan
    normals[alpha == 0] = np.nan
    return depth, alpha, normals


@pytest.mark.parametrize("radius", [1, 8, 24, 64, 128])
@pytest.mark.parametrize("with_normals", [False, True])
def test_cuda_matches_cpu_for_contacts_relief_transparency_and_atlases(cuda, radius, with_normals):
    depth, alpha, normals = scene()
    kwargs = dict(radius=radius, normals=normals if with_normals else None)
    cpu = ao_from_depth(depth, alpha, device="cpu", **kwargs)
    gpu = ao_from_depth(depth, alpha, device="cuda", **kwargs)
    assert gpu.dtype == np.float32
    assert np.all(np.isfinite(gpu)) and np.all((gpu >= 0) & (gpu <= 1))
    assert np.all(gpu[alpha == 0] == 1)
    assert np.allclose(gpu, cpu, atol=2e-5, rtol=0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
def test_cuda_handles_tiny_and_single_axis_images(cuda, shape):
    depth = np.full(shape, 0.5, np.float32)
    normals = np.broadcast_to(np.array((0, 0, 1), np.float32), (*shape, 3))
    gpu = ao_from_depth(depth, np.full(shape, 255, np.uint8), normals=normals, device="cuda")
    assert np.all(gpu == 1)


def test_cuda_chunk_boundaries_do_not_change_ao(cuda, monkeypatch):
    from smg import ao_gpu

    depth, alpha, normals = scene()
    full = ao_from_depth(depth, alpha, normals=normals, device="cuda")
    monkeypatch.setattr(ao_gpu, "RECEIVER_CHUNK_SIZE", 137)
    chunked = ao_from_depth(depth, alpha, normals=normals, device="cuda")
    assert np.allclose(chunked, full, atol=2e-6, rtol=0)


def test_cuda_remains_float32_inside_autocast(cuda):
    depth, alpha, normals = scene()
    expected = ao_from_depth(depth, alpha, normals=normals, device="cuda")
    with cuda.autocast("cuda", dtype=cuda.float16):
        actual = ao_from_depth(depth, alpha, normals=normals, device="cuda")
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=2e-6, rtol=0)


def test_cpu_mode_does_not_probe_or_import_cuda(monkeypatch):
    from smg import ao

    monkeypatch.setattr(ao, "_cuda_runtime", lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be probed")))
    depth, alpha, normals = scene((81, 81))
    assert np.all(np.isfinite(ao_from_depth(depth, alpha, normals=normals, device="cpu")))


def test_auto_works_without_torch_or_cuda_and_explicit_cuda_reports_error(monkeypatch):
    from smg import ao

    monkeypatch.setattr(ao, "_cuda_runtime", lambda: None)
    depth, alpha, normals = scene((81, 81))
    cpu = ao_from_depth(depth, alpha, normals=normals, device="cpu")
    assert np.array_equal(ao_from_depth(depth, alpha, normals=normals), cpu)
    with pytest.raises(RuntimeError, match="CUDA"):
        ao_from_depth(depth, alpha, normals=normals, device="cuda")


def test_auto_uses_gpu_for_large_maps(monkeypatch):
    torch = pytest.importorskip("torch")
    from smg import ao, ao_gpu

    monkeypatch.setattr(ao, "_cuda_runtime", lambda: torch)
    depth, alpha, normals = scene((81, 81))
    marker = np.full(depth.shape, 0.6, np.float32)
    calls = []

    def calculate(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(ao_gpu, "ao_cuda", calculate)
    assert ao_from_depth(depth, alpha, normals=normals) is marker
    assert len(calls) == 1


def test_auto_falls_back_only_for_gpu_oom_and_explicit_cuda_does_not(monkeypatch):
    torch = pytest.importorskip("torch")
    from smg import ao, ao_gpu

    monkeypatch.setattr(ao, "_cuda_runtime", lambda: torch)
    cleanup = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleanup.append(True))

    def out_of_memory(*args):
        raise torch.cuda.OutOfMemoryError("test GPU OOM")

    monkeypatch.setattr(ao_gpu, "ao_cuda", out_of_memory)
    depth, alpha, normals = scene((81, 81))
    cpu = ao_from_depth(depth, alpha, normals=normals, device="cpu")
    assert np.array_equal(ao_from_depth(depth, alpha, normals=normals), cpu)
    assert cleanup == [True]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        ao_from_depth(depth, alpha, normals=normals, device="cuda")

    def unexpected_error(*args):
        raise RuntimeError("unrelated CUDA error")

    monkeypatch.setattr(ao_gpu, "ao_cuda", unexpected_error)
    with pytest.raises(RuntimeError, match="unrelated"):
        ao_from_depth(depth, alpha, normals=normals)


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="auto, cpu или cuda"):
        ao_from_depth(np.zeros((1, 1)), np.full((1, 1), 255), device="invalid")
