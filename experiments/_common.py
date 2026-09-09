"""Shared Modal image, volume and timing helper for the experiment runners."""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent

cutedsl_cache = modal.Volume.from_name("cutedsl-cache", create_if_missing=True)
ENV = {"CUTE_DSL_CACHE_DIR": "/root/.cache/cutedsl"}
VOLUMES = {"/root/.cache/cutedsl": cutedsl_cache}

base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "nvidia-cutlass-dsl[cu13]",
    "torch",
    extra_index_url="https://download.pytorch.org/whl/cu130",
)


def bench(fn, iters=30):
    """Mean seconds per call, after warmup. Runs on the GPU side."""
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters * 1e-3
