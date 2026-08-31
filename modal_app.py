import modal

cutedsl_cache = modal.Volume.from_name("cutedsl-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nvidia-cutlass-dsl[cu13]",
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_python_source("kernels")
)

app = modal.App("cutedsl-b200", image=image)

GPU = "B200"
VOLUMES = {"/root/.cache/cutedsl": cutedsl_cache}
ENV = {"CUTE_DSL_CACHE_DIR": "/root/.cache/cutedsl"}


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def smoke():
    """Sanity check that the DSL compiles and runs on this GPU."""
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack
    from kernels.test import elem_add

    cutlass.cuda.initialize_cuda_context()

    props = torch.cuda.get_device_properties(0)
    print(props.name, f"sm_{props.major}{props.minor}")  # expect sm_100

    n = 1 << 20
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    args = [from_dlpack(t) for t in (a, b, out)]
    compiled = cute.compile(elem_add, *args)
    compiled(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a + b)
    print("ok")

    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=1800)
def fp16_gemm(m: int = 8192, n: int = 8192, k: int = 8192, iters: int = 20):
    """Correctness-check + benchmark the hand-rolled Blackwell fp16 GEMM."""
    import cutlass
    import torch
    from kernels.fp16_gemm_manual import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    res = run(m=m, n=n, k=k, iters=iters)
    print(f"mnk               : {res['mnk']}")
    print(f"cute (manual)     : {res['ours_ms']:8.3f} ms   {res['ours_tflops']:8.1f} TFLOP/s")
    print(f"torch.matmul      : {res['torch_ms']:8.3f} ms   {res['torch_tflops']:8.1f} TFLOP/s")
    print(f"ratio             : {res['ours_tflops'] / res['torch_tflops']:.2f}x of torch")

    cutedsl_cache.commit()
    return res


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_01_one_cta(m: int = 128, n: int = 256, k: int = 64):
    """Run the one-CTA CPAsync load kernel (smoke test)."""
    import cutlass
    import torch
    from kernels.gemm_01_one_cta import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_02_multi_cta(m: int = 256, n: int = 512, k: int = 16):
    """One UMMA per CTA across a grid of tiles."""
    import cutlass
    import torch
    from kernels.gemm_02_multi_cta import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_03_k_loop(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A K-loop of UMMAs per CTA, accumulating in TMEM."""
    import cutlass
    import torch
    from kernels.gemm_03_k_loop import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_04_pipelined(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A multi-stage pipelined K-loop, accumulating in TMEM."""
    import cutlass
    import torch
    from kernels.gemm_04_pipelined import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_05_warp_specialized(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A warp-specialized GEMM implementation."""
    import cutlass
    import torch
    from kernels.gemm_05_warp_specialized import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_06_swizzling(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A 128B-swizzled SMEM layout with a 64-deep K tile."""
    import cutlass
    import torch
    from kernels.gemm_06_swizzling import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_07_epilogue_pipelining(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A pipelined epilogue kernel."""
    import cutlass
    import torch
    from kernels.gemm_07_epilogue_pipelining import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, env=ENV, timeout=900)
def gemm_08_tma_multicast(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    """A TMA multicast kernel."""
    import cutlass
    import torch
    from kernels.gemm_08_tma_multicast import run

    cutlass.cuda.initialize_cuda_context()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    run(m=m, n=n, k=k, iters=iters)
    cutedsl_cache.commit()

@app.local_entrypoint()
def main(
    m: int = 8192,
    n: int = 8192,
    k: int = 8192,
    iters: int = 20,
    smoke_test: bool = False,
    gemm_01: bool = False,
    gemm_02: bool = False,
    gemm_03: bool = False,
    gemm_04: bool = False,
    gemm_05: bool = False,
    gemm_06: bool = False,
    gemm_07: bool = False,
    gemm_08: bool = False,
):
    if smoke_test:
        smoke.remote()
    elif gemm_01:
        gemm_01_one_cta.remote(m=m, n=n, k=k)
    elif gemm_02:
        gemm_02_multi_cta.remote(m=m, n=n, k=k)
    elif gemm_03:
        gemm_03_k_loop.remote(m=m, n=n, k=k, iters=iters)
    elif gemm_04:
        gemm_04_pipelined.remote(m=m, n=n, k=k, iters=iters)
    elif gemm_05:
        gemm_05_warp_specialized.remote(m=m, n=n, k=k, iters=iters)
    elif gemm_06:
        gemm_06_swizzling.remote(m=m, n=n, k=k, iters=iters)
    elif gemm_07:
        gemm_07_epilogue_pipelining.remote(m=m, n=n, k=k, iters=iters)
    elif gemm_08:
        gemm_08_tma_multicast.remote(m=m, n=n, k=k, iters=iters)
    else:
        fp16_gemm.remote(m=m, n=n, k=k, iters=iters)
