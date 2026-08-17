import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

io_dtype = cutlass.Float16
threads_per_cta = 128


@cute.kernel
def gemm_one_cta(
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    a_elems = cute.size(mA_mk)
    b_elems = cute.size(mB_nk)

    a_smem_ptr = cute.arch.alloc_smem(io_dtype, a_elems, 128)
    b_smem_ptr = cute.arch.alloc_smem(io_dtype, b_elems, 128)
    ab_full_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(ab_full_mbar, 1)

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # A bulk copy moves a flat run of bytes and knows nothing about rows or
    # columns, so both sides are just "n elements".
    gA = cute.make_tensor(mA_mk.iterator, cute.make_layout(a_elems))
    gB = cute.make_tensor(mB_nk.iterator, cute.make_layout(b_elems))
    sA = cute.make_tensor(a_smem_ptr, cute.make_layout(a_elems))
    sB = cute.make_tensor(b_smem_ptr, cute.make_layout(b_elems))

    bulk = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), io_dtype, num_bits_per_copy=128)
    nbytes = (a_elems + b_elems) * (io_dtype.width // 8)

    if warp_idx == 0:
        # one lane announces the byte count for both copies
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full_mbar, nbytes)
        # ...but the whole warp issues the copies, never inside elect_one
        cute.copy(bulk, gA, sA, mbar_ptr=ab_full_mbar)
        cute.copy(bulk, gB, sB, mbar_ptr=ab_full_mbar)

    # every thread waits for the data to land
    cute.arch.mbarrier_wait(ab_full_mbar, 0)

    if tidx == 0:
        for i in cutlass.range_constexpr(4):
            cute.printf("sA[{}]={}  sB[{}]={}", i, sA[i], i, sB[i])


@cute.jit
def gemm_one_cta_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    gemm_one_cta(a, b, c).launch(grid=(1, 1, 1), block=(threads_per_cta, 1, 1))


def run(m: int = 128, n: int = 256, k: int = 16):
    """Load only: check the printed SMEM values against the host tensors."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    print("host A[:4]", a.flatten()[:4].tolist())
    print("host B[:4]", b.flatten()[:4].tolist())

    a_t, b_t, c_t = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
    compiled = cute.compile(gemm_one_cta_host, a_t, b_t, c_t)
    compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()

    print("gemm_01_one_cta: load ok")
