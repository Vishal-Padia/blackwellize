import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
threads_per_cta = 128
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
tmem_cols = 512
num_stages = 4
num_k_blocks = mma_tiler_mnk[2] // mma_inst_shape_mnk[2]


@cute.kernel
def gemm_epilogue_pipelining(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    mma_coord_mnk = (bidx, bidy, None)

    # 128B swizzle
    a_smem_ptr = cute.arch.alloc_smem(
        io_dtype, num_stages * mma_tiler_mnk[0] * mma_tiler_mnk[2], 1024
    )
    b_smem_ptr = cute.arch.alloc_smem(
        io_dtype, num_stages * mma_tiler_mnk[1] * mma_tiler_mnk[2], 1024
    )

    # full[s] -> slot s has been loaded, safe to read
    # empty[s] -> the mma that was reading slot s has retried, safe to overwrite
    ab_full_mbar = cute.arch.alloc_smem(cutlass.Int64, num_stages, 8)
    ab_empty_mbar = cute.arch.alloc_smem(cutlass.Int64, num_stages, 8)

    # use exactly once, after the loop to drain the MMAs still in flight
    mma_done_mbar = cute.arch.alloc_smem(cutlass.Int64, 1, 8)
    tmem_addr_slot = cute.arch.alloc_smem(cutlass.Int32, 1, 16)

    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(num_stages):
                cute.arch.mbarrier_init(ab_full_mbar + s, 1)
                cute.arch.mbarrier_init(ab_empty_mbar + s, 1)
                cute.arch.mbarrier_arrive(
                    ab_empty_mbar + s
                )  # here the slot starts empty
            cute.arch.mbarrier_init(mma_done_mbar, 1)

        # pull the tensormaps into cache before the first copy needs them
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cute.arch.alloc_tmem(tmem_cols, tmem_addr_slot)
        # nothing more to allocate: release the permit now so other CTAs can
        # allocate. This does not release our columns -- dealloc_tmem does.
        cute.arch.relinquish_tmem_alloc_permit()

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_addr_slot
    )

    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    # The swizzle rides on the pointer, shape/stride on the layout. make_fragment_A
    # rejects a composed layout, so .inner (the swizzle) goes to recast_ptr and
    # .outer (the affine part) carries the appended stage mode
    sA = cute.make_tensor(
        cute.recast_ptr(a_smem_ptr, a_smem_layout.inner, dtype=io_dtype),
        cute.append(
            a_smem_layout.outer,
            cute.make_layout(num_stages, stride=cute.cosize(a_smem_layout.outer)),
        ),
    )
    sB = cute.make_tensor(
        cute.recast_ptr(b_smem_ptr, b_smem_layout.inner, dtype=io_dtype),
        cute.append(
            b_smem_layout.outer,
            cute.make_layout(num_stages, stride=cute.cosize(b_smem_layout.outer)),
        ),
    )

    nbytes = (
        mma_tiler_mnk[0] * mma_tiler_mnk[2] + mma_tiler_mnk[1] * mma_tiler_mnk[2]
    ) * (io_dtype.width // 8)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCtAcc = tiled_mma.make_fragment_C(
        tiled_mma.partition_shape_C(mma_inst_shape_mnk[:2])
    )
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    num_k_tiles = cute.size(gA, mode=[2])  # k // 64, dynamic

    if warp_idx == 0:
        stage = cutlass.Int32(0)
        empty_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_mbar + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ab_full_mbar + stage, nbytes)
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile)],
                tAsA[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile)],
                tBsB[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
            )

            stage += 1
            if stage == num_stages:
                stage = cutlass.Int32(0)
                empty_phase ^= 1

    elif warp_idx == 1:
        stage = cutlass.Int32(0)
        full_phase = cutlass.Int32(0)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            cute.arch.mbarrier_wait(ab_full_mbar + stage, full_phase)

            # cute.gemm does not toggle ACCUMULATE across the MMA_K mode, so walk
            # the four instructions here: the first overwrites TMEM, the rest add
            for k_block in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[(None, None, k_block, stage)],
                    tCrB[(None, None, k_block, stage)],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            with cute.arch.elect_one():
                tcgen05.commit(ab_empty_mbar + stage)  # slot is free again

            stage += 1
            if stage == num_stages:
                stage = cutlass.Int32(0)
                full_phase ^= 1

        # drain
        with cute.arch.elect_one():
            tcgen05.commit(mma_done_mbar)
        cute.arch.mbarrier_wait(mma_done_mbar, 0)

    # warps 2-3 skipped the loop entirely and have been parked here
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

    num_epi = cute.size(tmem_src, mode=[2])  # 4
    shape = gmem_dst[None, None, 0].shape
    acc_frag = [cute.make_rmem_tensor(shape, acc_dtype) for _ in range(2)]
    out_frag = [cute.make_rmem_tensor(shape, io_dtype) for _ in range(2)]

    # prologue: get chunk 0 moving
    cute.copy(tmem_tiled_copy, tmem_src[None, None, 0], acc_frag[0])

    for i in cutlass.range_constexpr(num_epi):
        cur = i % 2
        if i + 1 < num_epi:
            # independent registers, so this issues immediately and its latency
            # hides behind the convert + store below
            cute.copy(tmem_tiled_copy, tmem_src[None, None, i + 1], acc_frag[1 - cur])

        out_frag[cur].store(acc_frag[cur].load().to(io_dtype))
        cute.autovec_copy(out_frag[cur], gmem_dst[None, None, i])

    cute.arch.sync_threads()
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem_ptr, tmem_cols)


@cute.jit
def gemm_epilogue_pipelining_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    m, _ = a.shape
    n, _ = b.shape
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
    smem_atom = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
    )
    a_smem_shape = tiled_mma.partition_shape_A(cute.dice(mma_tiler_mnk, (1, None, 1)))
    b_smem_shape = tiled_mma.partition_shape_B(cute.dice(mma_tiler_mnk, (None, 1, 1)))
    a_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, a_smem_shape, order=(1, 2))
    b_smem_layout = tcgen05.tile_to_mma_shape(smem_atom, b_smem_shape, order=(1, 2))
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_smem_layout, mma_tiler_mnk, tiled_mma
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_smem_layout, mma_tiler_mnk, tiled_mma
    )
    gemm_epilogue_pipelining(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=(m // mma_tiler_mnk[0], n // mma_tiler_mnk[1], 1),
        block=(threads_per_cta, 1, 1),
    )


def run(m: int = 256, n: int = 512, k: int = 4096, iters: int = 50):
    import torch
    from cutlass.cute.runtime import from_dlpack

    tile_m, tile_n, tile_k = mma_tiler_mnk
    if k % tile_k or m % tile_m or n % tile_n:
        raise ValueError(
            f"need k%{tile_k}==0, m%{tile_m}==0, n%{tile_n}==0; got {(m, n, k)}"
        )

    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    a_t, b_t, c_t = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
    compiled = cute.compile(gemm_epilogue_pipelining_host, a_t, b_t, c_t)
    compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()).to(torch.float16)

    def row(t, i, j, n=6):
        return "  ".join(f"{v:9.4f}" for v in t[i, j : j + n].tolist())

    for i, j in ((0, 0), (m // 2, n // 2), (m - 1, n - 6)):
        print(f"c  [{i:3d},{j:3d}:] {row(c, i, j)}")
        print(f"ref[{i:3d},{j:3d}:] {row(ref, i, j)}")
    print(f"max |c - ref| = {(c.float() - ref.float()).abs().max().item():.6f}")

    torch.testing.assert_close(c, ref, atol=3e-1, rtol=1e-3)

    print("gemm_07_epilogue_pipelining: ok")

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters * 1e-3

    flops = 2 * m * n * k
    ours = bench(lambda: compiled(a_t, b_t, c_t))
    theirs = bench(lambda: torch.matmul(a, b.t()))

    print(f"\n{m}x{n}x{k}, {iters} iters")
    for name, secs in (("gemm_07", ours), ("torch", theirs)):
        print(f"  {name:8s} {secs * 1e6:9.1f} us  {flops / secs / 1e12:8.1f} TFLOP/s")
    print(f"  ratio    {theirs / ours:9.2f}x")
