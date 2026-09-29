"""Every rung (gemm_03 .. gemm_10) and torch across prefill and decode shapes.

    modal run experiments/rung_sweep.py

Writes results/rung_sweep.json. One container, one compile per (rung, shape).
Shapes are Llama-3-70B linear layers (hidden 8192, MLP 28672, fused qkv 10240).
M is the number of tokens in flight: 8192 for prefill, 256 for decode. 256 is
the smallest M gemm_09/10 accept (one CTA pair covers 256 rows), so torch is
also timed at the true decode batch sizes to show what padding costs.
"""

import json
import pathlib

import modal

from _common import ENV, VOLUMES, base_image

REPO = pathlib.Path(__file__).parent.parent

app = modal.App(
    "rung-sweep",
    image=base_image.add_local_python_source("_common").add_local_dir(
        str(REPO / "kernels"), "/root/kernels"
    ),
)

RUNGS = {
    "gemm_03_k_loop": "gemm_k_loop_host",
    "gemm_04_pipelined": "gemm_pipelined_host",
    "gemm_05_warp_specialized": "gemm_warp_specialized_host",
    "gemm_06_swizzling": "gemm_swizzling_host",
    "gemm_07_epilogue_pipelining": "gemm_epilogue_pipelining_host",
    "gemm_08_tma_multicast": "gemm_tma_multicast_host",
    "gemm_09_2cta_tcgen05": "gemm_2cta_tcgen05_host",
    "gemm_10_persistent_kernel": "gemm_persistent_kernel_host",
}

SHAPES = {
    "square_4096": (4096, 4096, 4096),
    "prefill_o_proj": (8192, 8192, 8192),
    "prefill_mlp_up": (8192, 28672, 8192),
    "decode_o_proj": (256, 8192, 8192),
    "decode_qkv": (256, 10240, 8192),
    "decode_mlp_up": (256, 28672, 8192),
    "decode_mlp_down": (256, 8192, 28672),
}

# torch only, to show the cost of padding real decode batches up to 256
TORCH_ONLY = [(m, 8192, 8192) for m in (1, 8, 16, 32, 64, 128)]


def _iters(m, n, k):
    flops = 2 * m * n * k
    return 5 if flops > 3e12 else 30


@app.function(gpu="B200", volumes=VOLUMES, env=ENV, timeout=3600)
def sweep(rungs: list[str], shapes: list[str]):
    import importlib
    import sys

    sys.path.insert(0, "/root")
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from _common import bench

    cutlass.cuda.initialize_cuda_context()
    out = {"gpu": torch.cuda.get_device_properties(0).name, "rows": []}

    def record(kernel, shape_name, m, n, k, secs, err=None):
        flops = 2 * m * n * k
        nbytes = 2 * (m * k + n * k + m * n)
        row = dict(
            kernel=kernel, shape=shape_name, m=m, n=n, k=k,
            us=secs * 1e6, tflops=flops / secs / 1e12,
            intensity=flops / nbytes, gbps=nbytes / secs / 1e9, err=err,
        )
        out["rows"].append(row)
        print(json.dumps(row), flush=True)

    for name in shapes:
        m, n, k = SHAPES[name]
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        iters = _iters(m, n, k)
        record("torch", name, m, n, k, bench(lambda: torch.matmul(a, b.t()), iters))
        ref = torch.matmul(a, b.t())
        for rung in rungs:
            c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
            at, bt, ct = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
            try:
                host = getattr(importlib.import_module(f"kernels.{rung}"), RUNGS[rung])
                fn = cute.compile(host, at, bt, ct)
                fn(at, bt, ct)
                torch.cuda.synchronize()
                err = (c.float() - ref.float()).abs().max().item()
                record(rung, name, m, n, k, bench(lambda: fn(at, bt, ct), iters), err)
            except Exception as ex:
                print(json.dumps(dict(kernel=rung, shape=name, failed=f"{type(ex).__name__}: {str(ex)[:120]}")), flush=True)
                out["rows"].append(dict(kernel=rung, shape=name, m=m, n=n, k=k, failed=str(ex)[:120]))
            del c
        del a, b, ref
        torch.cuda.empty_cache()

    return out


@app.function(gpu="B200", volumes=VOLUMES, env=ENV, timeout=900)
def torch_small():
    import torch

    from _common import bench

    rows = []
    for m, n, k in TORCH_ONLY:
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, k, dtype=torch.float16, device="cuda")
        secs = bench(lambda: torch.matmul(a, b.t()), 50)
        nbytes = 2 * (m * k + n * k + m * n)
        rows.append(dict(kernel="torch", shape=f"m{m}", m=m, n=n, k=k, us=secs * 1e6,
                         tflops=2 * m * n * k / secs / 1e12, gbps=nbytes / secs / 1e9))
        print(json.dumps(rows[-1]), flush=True)
    return rows


@app.local_entrypoint()
def main(rungs: str = ",".join(RUNGS), shapes: str = ",".join(SHAPES)):
    res = sweep.remote(rungs.split(","), shapes.split(","))
    res["torch_small"] = torch_small.remote()
    path = REPO / "results" / "rung_sweep.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")
