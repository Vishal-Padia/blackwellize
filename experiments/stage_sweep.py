import modal

from _common import ENV, VOLUMES, base_image

app = modal.App(
    "gemm09-stage-sweep",
    image=base_image.add_local_python_source(
        "stage_sweep", "kernel_stages", "_common"
    ),
)

STAGES = (2, 3, 4, 5, 6, 7)
SHAPES = ((4096, 4096, 4096), (8192, 8192, 8192))


@app.function(gpu="B200", volumes=VOLUMES, env=ENV, timeout=1800)
def go(stages: list[int], shapes: list[tuple[int, int, int]], iters: int):
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from _common import bench
    from kernel_stages import gemm_2cta_tcgen05_host

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
        k_tiles = k // 64
        for s in stages:
            c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
            at, bt, ct = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
            smem = s * (128 + 128) * 64 * 2 // 1024
            note = "" if k_tiles >= s else f"  (only {k_tiles} k-tiles!)"
            try:
                fn = cute.compile(gemm_2cta_tcgen05_host, at, bt, ct, s)
                fn(at, bt, ct)
                torch.cuda.synchronize()
                err = (c.float() - ref.float()).abs().max().item()
                ours = bench(lambda: fn(at, bt, ct), iters)
                ok = "ok" if err < 1.0 else f"ERR {err:.1f}"
                print(
                    f"   stages={s} ({smem:3d} KiB) {ours * 1e6:9.1f} us  "
                    f"{fl / ours / 1e12:7.1f} TFLOP/s  {them / ours:5.2f}x  {ok}{note}",
                    flush=True,
                )
            except Exception as ex:
                print(
                    f"   stages={s} ({smem:3d} KiB) FAILED "
                    f"{type(ex).__name__}: {str(ex)[:60]}",
                    flush=True,
                )
            del c
            torch.cuda.empty_cache()
        del a, b, ref
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(stages: str = "", shapes: str = "", iters: int = 30):
    """--stages 2,4,6,7   --shapes 4096x4096x4096,8192x8192x8192"""
    st = [int(x) for x in stages.split(",")] if stages else list(STAGES)
    sh = (
        [tuple(int(v) for v in s.split("x")) for s in shapes.split(",")]
        if shapes
        else list(SHAPES)
    )
    go.remote(stages=st, shapes=sh, iters=iters)
