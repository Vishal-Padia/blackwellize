"""ncu-profile a kernel across one or more shapes, drop each report in a volume.

    modal run experiments/profile_ncu.py --kernel gemm_02_multi_cta
    modal run experiments/profile_ncu.py --kernel gemm_06_swizzling --shapes "128x8192x8192,4096x4096x4096,8192x8192x8192"

One .ncu-rep per shape, named "{kernel}_{m}x{n}x{k}". Runs shapes back to
back on one B200 container rather than fanning out, since GPU containers can
be slow to schedule concurrently.

Then locally:

    modal volume get ncu-profiles gemm_09_2cta_tcgen05_4096x4096x4096.ncu-rep

and open it in the Nsight Compute GUI.

ncu needs --clock-control none on Modal (it cannot lock GPU clocks in the
container). Results carry a "data collection happened without fixed GPU
frequencies" warning, which is expected here.
"""

import pathlib

import modal

REPO = pathlib.Path(__file__).parent.parent
profiles = modal.Volume.from_name("ncu-profiles", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install(
        "nvidia-cutlass-dsl[cu13]",
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir(str(REPO / "kernels"), "/root/kernels")
)

app = modal.App("ncu-profile", image=image)

DRIVER = '''
import sys
sys.path.insert(0, "/root")
import cutlass, torch, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from kernels.{kernel} import {host}

m, n, k = {m}, {n}, {k}
cutlass.cuda.initialize_cuda_context()
torch.manual_seed(42)
a = torch.randn(m, k, dtype=torch.float16, device="cuda")
b = torch.randn(n, k, dtype=torch.float16, device="cuda")
c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
at, bt, ct = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
fn = cute.compile({host}, at, bt, ct)
for _ in range(8):          # 0-3 warm up, ncu captures one steady-state launch
    fn(at, bt, ct)
torch.cuda.synchronize()
print("driver ok", flush=True)
'''

HOSTS = {
    "gemm_01_one_cta": "gemm_one_cta_host",
    "gemm_02_multi_cta": "gemm_multi_cta_host",
    "gemm_03_k_loop": "gemm_k_loop_host",
    "gemm_04_pipelined": "gemm_pipelined_host",
    "gemm_05_warp_specialized": "gemm_warp_specialized_host",
    "gemm_06_swizzling": "gemm_swizzling_host",
    "gemm_07_epilogue_pipelining": "gemm_epilogue_pipelining_host",
    "gemm_08_tma_multicast": "gemm_tma_multicast_host",
    "gemm_09_2cta_tcgen05": "gemm_2cta_tcgen05_host",
    "gemm_10_persistent_kernel": "gemm_persistent_kernel_host",
}


@app.function(gpu="B200", volumes={"/profiles": profiles}, timeout=3600)
def profile(kernel: str, shapes: list[tuple[int, int, int]], sets: str):
    import subprocess

    for m, n, k in shapes:
        name = f"{kernel}_{m}x{n}x{k}"
        pathlib.Path("/root/driver.py").write_text(
            DRIVER.format(kernel=kernel, host=HOSTS[kernel], m=m, n=n, k=k)
        )

        cmd = (
            f"ncu --clock-control none --set {sets} "
            f"--launch-skip 4 --launch-count 1 --target-processes all "
            f"-f -o /profiles/{name} python /root/driver.py"
        )
        print(f"$ {cmd}\n", flush=True)
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/root")
        print((p.stdout + p.stderr)[-6000:], flush=True)
        print(f"[{name}] exit {p.returncode}", flush=True)

    profiles.commit()
    for f in sorted(pathlib.Path("/profiles").iterdir()):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB", flush=True)


DEFAULT_SHAPES = "128x8192x8192,512x8192x8192,4096x4096x4096,8192x8192x8192"


@app.local_entrypoint()
def main(
    kernel: str = "gemm_09_2cta_tcgen05",
    shapes: str = DEFAULT_SHAPES,
    sets: str = "full",
):
    parsed = [tuple(int(x) for x in s.split("x")) for s in shapes.split(",")]
    profile.remote(kernel=kernel, shapes=parsed, sets=sets)
    for m, n, k in parsed:
        print(f"  modal volume get ncu-profiles {kernel}_{m}x{n}x{k}.ncu-rep")


SECTIONS = "SpeedOfLight,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,Occupancy,WarpStateStats,SchedulerStats"


@app.function(volumes={"/profiles": profiles}, timeout=900)
def report(name: str, sections: str):
    """Text-dump an existing .ncu-rep from the volume. No GPU needed."""
    import subprocess

    sec = " ".join(f"--section {s}" for s in sections.split(",") if s)
    cmd = f"ncu --import /profiles/{name}.ncu-rep --page details {sec}"
    print(f"$ {cmd}\n", flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(p.stdout + p.stderr, flush=True)


@app.local_entrypoint()
def show(name: str = "gemm_09_2cta_tcgen05_4096x4096x4096", sections: str = SECTIONS):
    report.remote(name=name, sections=sections)
