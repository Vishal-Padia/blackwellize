import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
# One UMMA instruction: 128x256x16. One CTA K-step: 128x256x64. The ratio is why
# the mainloop issues 64/16 = 4 MMAs per K-tile.
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128
ab_stages = 1  # single buffer: the loader and the MMA take turns
tmem_cols = 512


# Plain K-major SMEM tiling unit: 8 rows x 8 fp16 = 16B, the minimum contiguity
# UMMA accepts. No swizzle at all, so MMA reads and TMA writes keep hitting the
# same SMEM banks.
SMEM_ATOM = (8, 8)


@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.Layout,
    b_smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # SMEM: one A tile + one B tile = 16 + 32 = 48 KiB
    a_smem_ptr = cute.arch.alloc_smem(io_dtype, mma_tiler_mnk[0] * mma_tiler_mnk[2], 1024)
    b_smem_ptr = cute.arch.alloc_smem(io_dtype, mma_tiler_mnk[1] * mma_tiler_mnk[2], 1024)
    # Two barriers, one stage each: data-arrived, and MMA-finished-reading.
    ab_full_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)
    mma_done_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)
    tmem_addr_slot = cute.arch.alloc_smem(cutlass.Int32, 1, 16)

    sA = cute.make_tensor(a_smem_ptr, a_smem_layout)
    sB = cute.make_tensor(b_smem_ptr, b_smem_layout)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(ab_full_mbar, 1)  # TMA transaction
            cute.arch.mbarrier_init(mma_done_mbar, 1)  # one tcgen05.commit
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cute.arch.alloc_tmem(tmem_cols, tmem_addr_slot)  # warp-wide, not elected

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_addr_slot
    )

    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)  # UMMA is issued by a single thread
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(sA)  # SMEM descriptors, not registers
    tCrB = tiled_mma.make_fragment_B(sB)
    tCtAcc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(mma_tiler_mnk[:2]))
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)  # repoint at our TMEM

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )

    # ab_full is a transaction barrier, so A and B can share it
    bytes_per_stage = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

    num_k_tiles = cute.size(gA, mode=[2])
    num_k_blocks = cute.size(tCrA, mode=[2])

    if warp_idx == 0:
        # One buffer reused every iteration, so each barrier completes exactly
        # once per K-tile and its phase just alternates.
        full_phase = cutlass.Int32(0)
        done_phase = cutlass.Int32(0)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            # load
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ab_full_mbar, bytes_per_stage)
            cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, 0)], tma_bar_ptr=ab_full_mbar)
            cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, 0)], tma_bar_ptr=ab_full_mbar)

            # wait for it -- nothing else to do meanwhile, there is no other buffer
            cute.arch.mbarrier_wait(ab_full_mbar, full_phase)
            full_phase ^= 1

            for k_block in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma, tCtAcc,
                    tCrA[(None, None, k_block, 0)],
                    tCrB[(None, None, k_block, 0)],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)  # first MMA overwrites

            # the next iteration overwrites sA/sB, so the MMA must be done reading
            with cute.arch.elect_one():
                tcgen05.commit(mma_done_mbar)
            cute.arch.mbarrier_wait(mma_done_mbar, done_phase)
            done_phase ^= 1

        cute.arch.relinquish_tmem_alloc_permit()

    # Epilogue, all 128 threads: TMEM -> RMEM -> GMEM. The last mma_done wait
    # above already guarantees the accumulator is complete; this just publishes
    # it to the other three warps.
    cute.arch.sync_threads()

    # 4 chunks because the tcgen05.ld atom below moves 64 fp32 per thread and the
    # accumulator is 128x256.
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // 4),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), acc_dtype)
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
def gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
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

    a_smem_shape = cute.append(
        tiled_mma.partition_shape_A(cute.dice(mma_tiler_mnk, (1, None, 1))), ab_stages
    )
    b_smem_shape = cute.append(
        tiled_mma.partition_shape_B(cute.dice(mma_tiler_mnk, (None, 1, 1))), ab_stages
    )
    # order=(1,2,3): K first, then MN, then stages -> one stage is contiguous K-major
    smem_atom = cute.make_layout(SMEM_ATOM, stride=(SMEM_ATOM[1], 1))
    a_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, a_smem_shape, order=(1, 2, 3))
    b_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, b_smem_shape, order=(1, 2, 3))

    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, cute.select(a_smem_layout, mode=[0, 1, 2]), mma_tiler_mnk, tiled_mma
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, cute.select(b_smem_layout, mode=[0, 1, 2]), mma_tiler_mnk, tiled_mma
    )

    gemm_kernel(
        tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, c,
        a_smem_layout, b_smem_layout,
    ).launch(
        grid=cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2]),
        block=(threads_per_cta, 1, 1),
    )


def run(m: int = 8192, n: int = 8192, k: int = 8192, iters: int = 20, tolerance: float = 1e-1):
    """Compile, verify against torch, benchmark."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    if m % mma_tiler_mnk[0] or n % mma_tiler_mnk[1] or k % mma_tiler_mnk[2]:
        raise ValueError(f"mnk must be multiples of {mma_tiler_mnk}; got {(m, n, k)}")

    torch.manual_seed(1111)

    def make(mn, kk):
        return (
            torch.empty(mn, kk, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=torch.float16, device="cuda")
        )

    a, b = make(m, k), make(n, k)
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    a_t, b_t, c_t = (from_dlpack(t, assumed_align=32) for t in (a, b, c))

    compiled = cute.compile(gemm_host, a_t, b_t, c_t)
    compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=tolerance, rtol=1e-3)  # rtol >= fp16 ulp

    def bench(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    flops = 2.0 * m * n * k
    ours_ms = bench(lambda: compiled(a_t, b_t, c_t))
    torch_ms = bench(lambda: torch.matmul(a, b.t()))

    return {
        "mnk": (m, n, k),
        "ours_ms": ours_ms,
        "ours_tflops": flops / (ours_ms * 1e-3) / 1e12,
        "torch_ms": torch_ms,
        "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
    }
