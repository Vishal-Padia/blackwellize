"""gemm_09 vs cuBLAS across problem shapes.

    modal run experiments/shape_sweep.py

One container, one compile per shape. Results 2026-09-02:

                 shape       ours  TFLOP/s      torch  TFLOP/s  ratio
  4096x4096x4096        107.7   1275.6       94.4   1456.1   0.88x
  8192x8192x8192        846.1   1299.5      779.9   1409.8   0.92x
  2048x2048x2048         18.8    912.5       14.4   1189.9   0.77x
  4096x4096x512          33.2    517.4       14.5   1184.3   0.44x   <-- worst
  8192x1024x4096         57.2   1201.2       48.3   1422.8   0.84x
  1024x8192x4096         57.3   1199.2       48.5   1418.3   0.85x
  4096x4096x16384       402.2   1366.7      369.8   1486.7   0.92x   <-- best

Deep K is where we're strong, short kernels are where we lose. At k=512 there
are only 8 K-tiles against 6 pipeline stages, so the pipeline never reaches
steady state and prologue + drain + epilogue are most of the runtime. That is a
fixed-cost problem, i.e. what a persistent kernel would fix.
"""

import pathlib

import modal

from _common import ENV, VOLUMES, base_image

REPO = pathlib.Path(__file__).parent.parent

app = modal.App(
    "gemm09-shape-sweep",
    image=base_image.add_local_python_source("_common").add_local_dir(
        str(REPO / "kernels"), "/root/kernels"
    ),
)

SHAPES = [
    (4096, 4096, 4096),  # baseline
    (8192, 8192, 8192),  # large
    (2048, 2048, 2048),  # small
    (4096, 4096, 512),  # shallow K
    (8192, 1024, 4096),  # narrow N
    (1024, 8192, 4096),  # narrow M
    (4096, 4096, 16384),  # deep K
]


@app.function(gpu="B200", volumes=VOLUMES, env=ENV, timeout=1800)
def sweep(iters: int):
    import sys

    sys.path.insert(0, "/root")
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from _common import bench
    from kernels.gemm_09_2cta_tcgen05 import gemm_2cta_tcgen05_host

    cutlass.cuda.initialize_cuda_context()
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} sm_{p.major}{p.minor}\n", flush=True)
    print(
        f"{'shape':>22}  {'ours':>9} {'TFLOP/s':>8}  {'torch':>9} {'TFLOP/s':>8}  ratio",
        flush=True,
    )

    for m, n, k in SHAPES:
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
        at, bt, ct = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
        try:
            fn = cute.compile(gemm_2cta_tcgen05_host, at, bt, ct)
            fn(at, bt, ct)
            torch.cuda.synchronize()
            ref = (a.float() @ b.float().t()).to(torch.float16)
            err = (c.float() - ref.float()).abs().max().item()
            ours = bench(lambda: fn(at, bt, ct), iters)
            them = bench(lambda: torch.matmul(a, b.t()), iters)
            fl = 2 * m * n * k
            ok = "ok" if err < 1.0 else f"ERR {err:.1f}"
            print(
                f"{m:>6}x{n}x{k:<6}  {ours * 1e6:9.1f} {fl / ours / 1e12:8.1f}  "
                f"{them * 1e6:9.1f} {fl / them / 1e12:8.1f}  {them / ours:5.2f}x  {ok}",
                flush=True,
            )
        except Exception as ex:
            print(
                f"{m:>6}x{n}x{k:<6}  FAILED {type(ex).__name__}: {str(ex)[:70]}",
                flush=True,
            )
        del a, b, c
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(iters: int = 30):
    sweep.remote(iters=iters)
