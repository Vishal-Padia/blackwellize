import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elem_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    i = bx * bdx + tx
    if i < out.shape[0]:
        out[i] = a[i] + b[i]

@cute.jit
def elem_add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    n = out.shape[0]
    tpb = 128
    blocks = (n + tpb - 1) // tpb

    elem_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),
        block=(tpb, 1, 1),
    )

# host side
n = 1 << 20
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn(n, device="cuda", dtype=torch.float32)
out = torch.empty_like(a)

compiled = cute.compile(elem_add, *[from_dlpack(t) for t in (a, b, out)])
compiled(from_dlpack(a), from_dlpack(b), from_dlpack(out))
torch.testing.assert_close(out, a + b)