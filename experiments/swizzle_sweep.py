"""Threadblock swizzle: grouped rasterisation over the (pair_m, n) tile grid.

    modal run experiments/swizzle_sweep.py
    modal run experiments/swizzle_sweep.py --groups 1,4,8,16,32 --shapes 8192x8192x8192

The kernel is kernel_swizzled.py -- a copy of kernels/gemm_09_2cta_tcgen05.py
with one change: the tile coordinate goes through a grouped rasterisation
instead of straight (bidx, bidy). `swizzle_m` is a Constexpr argument, so one
kernel serves every group size and cute.compile specialises per value.

Instead of CTA (bidx, bidy) taking output tile (bidx, bidy), walk `swizzle_m`
pair-tiles down M before stepping along N, so neighbouring CTAs share B tiles in
L2. The CTA pair must stay intact, so only the (pair_m, n) grid is permuted and
bidx % 2 is left alone. swizzle_m=1 is exactly the identity mapping.

Results 2026-09-02:

  8192x8192x8192   torch  739.6 us  1486.6 TFLOP/s
     swizzle_m=1      848.2 us  1296.2   0.87x
     swizzle_m=4      803.0 us  1369.3   0.92x
     swizzle_m=8      785.6 us  1399.5   0.94x   <-- optimum
     swizzle_m=16     802.8 us  1369.7   0.92x
     swizzle_m=32     846.6 us  1298.7   0.87x

  4096x4096x16384  torch  402.5 us  1365.8 TFLOP/s
     swizzle_m=8      401.4 us  1369.6   1.00x   <-- parity with cuBLAS

  4096x4096x4096   no effect (107.4 / 107.8 / 108.5 for g1 / g4 / g8): only 256
  pair-tiles over 148 SMs, so there aren't enough waves for order to matter.

CAVEAT: the mapping assumes pairs_m % swizzle_m == 0, where pairs_m = m / 256.
Violate it and CTAs walk off the end of M and produce wrong results (g32 at
m=4096 gives max |c - ref| = 772). The runner flags the non-divisible cases.
A real version needs the clamp the Triton-style grouped ordering has.
"""

import modal

from _common import ENV, VOLUMES, base_image

app = modal.App(
    "gemm09-swizzle-sweep",
    image=base_image.add_local_python_source(
        "swizzle_sweep", "kernel_swizzled", "_common"
    ),
)

GROUPS = (1, 4, 8, 16, 32)
SHAPES = ((4096, 4096, 4096), (8192, 8192, 8192))


@app.function(gpu="B200", volumes=VOLUMES, env=ENV, timeout=1800)
def go(groups: list[int], shapes: list[tuple[int, int, int]], iters: int):
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from _common import bench
    from kernel_swizzled import gemm_swizzled_host, mma_tiler_mnk

    cutlass.cuda.initialize_cuda_context()
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} sm_{p.major}{p.minor}\n", flush=True)

    for m, n, k in shapes:
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        ref = (a.float() @ b.float().t()).to(torch.float16)
        them = bench(lambda: torch.matmul(a, b.t()), iters)
        fl = 2 * m * n * k
        print(
            f"{m}x{n}x{k}   torch {them * 1e6:8.1f} us  {fl / them / 1e12:7.1f} TFLOP/s",
            flush=True,
        )
        pairs_m = m // mma_tiler_mnk[0]
        for g in groups:
            note = "" if pairs_m % g == 0 else f"  (pairs_m={pairs_m} not divisible!)"
            c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
            at, bt, ct = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
            try:
                fn = cute.compile(gemm_swizzled_host, at, bt, ct, g)
                fn(at, bt, ct)
                torch.cuda.synchronize()
                err = (c.float() - ref.float()).abs().max().item()
                ours = bench(lambda: fn(at, bt, ct), iters)
                ok = "ok" if err < 1.0 else f"ERR {err:.1f}"
                print(
                    f"   swizzle_m={g:<3} {ours * 1e6:9.1f} us  "
                    f"{fl / ours / 1e12:7.1f} TFLOP/s  {them / ours:5.2f}x  {ok}{note}",
                    flush=True,
                )
            except Exception as ex:
                print(
                    f"   swizzle_m={g:<3} FAILED {type(ex).__name__}: {str(ex)[:60]}",
                    flush=True,
                )
            del c
            torch.cuda.empty_cache()
        print("\n", flush=True)
        del a, b, ref
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(groups: str = "", shapes: str = "", iters: int = 30):
    """--groups 1,4,8   --shapes 4096x4096x4096,8192x8192x8192"""
    gs = [int(x) for x in groups.split(",")] if groups else list(GROUPS)
    sh = (
        [tuple(int(v) for v in s.split("x")) for s in shapes.split(",")]
        if shapes
        else list(SHAPES)
    )
    go.remote(groups=gs, shapes=sh, iters=iters)
