# modal_app.py
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


@app.function(
    gpu="B200",
    # env={"CUTE_DSL_LOG_TO_CONSOLE": "1"},
    volumes={"/root/.cache/cutedsl": cutedsl_cache},
    timeout=900,
)
def run():
    import cutlass
    import torch, cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from kernels.test import elem_add

    cutlass.cuda.initialize_cuda_context()

    props = torch.cuda.get_device_properties(0)
    print(props.name, f"sm_{props.major}{props.minor}")   # expect sm_100

    
    # host side
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


@app.local_entrypoint()
def main():
    run.remote()