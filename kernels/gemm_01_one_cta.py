import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
threads_per_cta = 128
mma_inst_shape_mnk = (128, 256, 16)
tmem_cols = 512
SMEM_ATOM = (8, 8)


@cute.kernel
def gemm_one_cta(
    tiled_mma: cute.TiledMma,
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.Layout,
    b_smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    a_elems = cute.size(mA_mk)
    b_elems = cute.size(mB_nk)

    a_smem_ptr = cute.arch.alloc_smem(io_dtype, a_elems, 128)
    b_smem_ptr = cute.arch.alloc_smem(io_dtype, b_elems, 128)
    ab_full_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)
    mma_done_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)
    tmem_addr_slot = cute.arch.alloc_smem(cutlass.Int32, 1, 16)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(ab_full_mbar, 1)
            cute.arch.mbarrier_init(mma_done_mbar, 1)
        cute.arch.alloc_tmem(tmem_cols, tmem_addr_slot)

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_addr_slot
    )

    # A bulk copy moves a flat run of bytes and knows nothing about rows or
    # columns, so both sides are just "n elements"
    gA = cute.make_tensor(mA_mk.iterator, cute.make_layout(a_elems))
    gB = cute.make_tensor(mB_nk.iterator, cute.make_layout(b_elems))
    sA = cute.make_tensor(a_smem_ptr, cute.make_layout(a_elems))
    sB = cute.make_tensor(b_smem_ptr, cute.make_layout(b_elems))
    sA_mma = cute.make_tensor(a_smem_ptr, a_smem_layout)
    sB_mma = cute.make_tensor(b_smem_ptr, b_smem_layout)

    bulk = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), io_dtype, num_bits_per_copy=128)
    nbytes = (a_elems + b_elems) * (io_dtype.width // 8)

    tCrA = tiled_mma.make_fragment_A(sA_mma)
    tCrB = tiled_mma.make_fragment_B(sB_mma)
    thr_mma = tiled_mma.get_slice(0)  # UMMA is issued by a single thread
    tCgC = thr_mma.partition_C(mC_mn)  # mC_mn already IS the 128x256 tile
    tCtAcc = tiled_mma.make_fragment_C(
        tiled_mma.partition_shape_C(mma_inst_shape_mnk[:2])
    )
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    if warp_idx == 0:
        # one lane announces the byte count for both copies
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ab_full_mbar, nbytes)
        # ...but the whole warp issues the copies, never inside elect_one
        cute.copy(bulk, gA, sA, mbar_ptr=ab_full_mbar)
        cute.copy(bulk, gB, sB, mbar_ptr=ab_full_mbar)

    # every thread waits for the data to land
    cute.arch.mbarrier_wait(ab_full_mbar, 0)

    if warp_idx == 0:
        cute.gemm(tiled_mma, tCtAcc, tCrA[None, None, 0], tCrB[None, None, 0], tCtAcc)
        # A dedicated barrier for MMA completion, used once, so phase 0 both times
        # and no parity reasoning anywhere. Reusing ab_full_mbar for this hung
        with cute.arch.elect_one():
            tcgen05.commit(mma_done_mbar)
        cute.arch.mbarrier_wait(mma_done_mbar, 0)
        cute.arch.relinquish_tmem_alloc_permit()

    cute.arch.sync_threads()

    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // 4),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), acc_dtype
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tmem_src = tmem_thr_copy.partition_S(tCtAcc_epi)
    gmem_dst = tmem_thr_copy.partition_D(gC_epi)

    acc_frag = cute.make_rmem_tensor(gmem_dst[None, None, 0].shape, acc_dtype)
    out_frag = cute.make_rmem_tensor(gmem_dst[None, None, 0].shape, io_dtype)

    for i in cutlass.range_constexpr(cute.size(tmem_src, mode=[2])):
        cute.copy(tmem_tiled_copy, tmem_src[None, None, i], acc_frag)
        out_frag.store(acc_frag.load().to(io_dtype))
        cute.autovec_copy(out_frag, gmem_dst[None, None, i])

    cute.arch.sync_threads()
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem_ptr, tmem_cols)


@cute.jit
def gemm_one_cta_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_inst_shape_mnk, (1, None, 1))
    )
    b_smem_shape = tiled_mma.partition_shape_B(
        cute.dice(mma_inst_shape_mnk, (None, 1, 1))
    )
    smem_atom = cute.make_layout(SMEM_ATOM, stride=(SMEM_ATOM[1], 1))
    a_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, a_smem_shape, order=(1, 2))
    b_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, b_smem_shape, order=(1, 2))
    gemm_one_cta(tiled_mma, a, b, c, a_smem_layout, b_smem_layout).launch(
        grid=(1, 1, 1),
        block=(threads_per_cta, 1, 1),
    )


def run(m: int = 128, n: int = 256, k: int = 16):
    """One CTA, one UMMA, checked against torch."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    if (m, n, k) != mma_inst_shape_mnk:
        raise ValueError(f"mnk must be {mma_inst_shape_mnk}; got {(m, n, k)}")

    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    def pack(t, rows):
        return t.view(rows, k // 8, 8).permute(1, 0, 2).contiguous()

    a_p, b_p = pack(a, m), pack(b, n)

    a_t, b_t, c_t = (from_dlpack(t, assumed_align=16) for t in (a_p, b_p, c))
    compiled = cute.compile(gemm_one_cta_host, a_t, b_t, c_t)
    compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()).to(torch.float16)

    def row(t, i, j, n=6):
        return "  ".join(f"{v:9.4f}" for v in t[i, j : j + n].tolist())

    for i, j in ((0, 0), (63, 100), (127, 250)):
        print(f"c  [{i:3d},{j:3d}:] {row(c, i, j)}")
        print(f"ref[{i:3d},{j:3d}:] {row(ref, i, j)}")
    print(f"max |c - ref| = {(c.float() - ref.float()).abs().max().item():.6f}")

    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-3)

    print("gemm_01_one_cta: ok")
