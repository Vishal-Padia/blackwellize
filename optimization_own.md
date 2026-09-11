# Which optimizations were made

1. K-loop, 1 stage (baseline)
2. 4-stage pipeline
3. Warp specialization
4. 128B swizzle
5. Pipelined epilogue
6. TMA multicast
7. 2-CTA tcgen05

The K-loop is the **baseline**, not an improvement: it is the first version that handles an arbitrary K, and everything is measured relative to it. Each rung has its own PR, so the diff for any single change is readable on its own.

Three of the six changes after the baseline measured **no improvement**. They are kept, reported, and labelled as such, because they turned out to be the more informative half of the project: the null at rung 3 is what forced the ablations that found the one change worth 4x.

# How was the performance measured?

Each kernel's `run()` uses CUDA events to time execution:

- record a start event
- run the kernel `iters` times (20-50)
- record a stop event, synchronize
- mean time per iteration = `start.elapsed_time(stop) / iters`

FLOPs use the standard GEMM count `2 * m * n * k`. `torch.matmul` on the same tensors, in the same process, on the same GPU, goes through the identical timing path. `torch.matmul` on fp16 CUDA tensors dispatches to cuBLAS (cuBLASLt), not cuDNN.

Output per kernel: time in microseconds, TFLOP/s, and the ratio against torch.

# The Starting Point

The kernel before the first rung (see [PR #3](https://github.com/blackwellize/blackwellize/pull/3)) is the starting point, and we started making optimizations on top of it. It's very minimal, and here each CTA loads a single MMA tile of A and B from global memory to shared memory via `cute.copy()` behind an mbarrier, issues one UMMA through `cute.gemm()`, and writes to accumulator. 

# Optimization Rungs

## 1. K-loop, 1 stage (baseline)

Each block previously performed one MMA, which would only work if K fits in the single `k=16` tile. This rung splits k into chunks of 16, and performs one MMA per chunk and accumulates the results.

$$
C_{tile} = A_{0}B_{0}^{T} + A_{1}B_{1}^{T} + ... + A_{k/16}B_{k/16}^{T}
$$

Nothing else changes compared to the baseline. Both mbarriers are now reused once per iteration. The first MMA overwrites the TMEM and every later ones added. 

This is not exactly an optimization, and it's not that fast. It's the first version that computes GMEMM and at 0.11x of cuBLAS. 

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_03 | 837.7 us | 164.1   | 0.11x |
| torch   | 93.5 us  | 1470.4  |       |

## 2. 4-stage pipeline

**`num_stages` is the number of SMEM buffers, not a split of K**. The two are independent.

The loops still runs like 256 times, the change is how we load from global memory.

We fill the buffer with tiles 0,1,2,3 and then iteration `k` waits on on `ab_full_mbar + stage`, and issues the MMA, and refills that same slot with tile ` k + 4`. Since the refill targets the slot just consumed, and the MMA-done wait sits above it, no separate "empty" barrier is needed yet.

Each barrier is now reused once per lap around the ring rather than once per iteration, so full `full_phase` flips when `stage` wraps to 0, while `done_phase` still flips every iteration. 

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_04 | 451.7 us | 304.3   | 0.21x |
| torch   | 95.0 us  | 1447.1  |       |

## 3. Warp Specialization

Previously only one warp did the job that in a sequence (ie not parallel), and the rest of the warps were idle until the epilogue. And in this rung, we split the roles between the warps.

```
warp 0 (producer): wait empty[s] -> issue TMA into slot s -> repeat
warp 1 (consumer): wait full[s] -> MMA on slot s -> commit empty [s] -> repeat
warp 2-3: idle until the epilogue
```

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_05 | 451.4 us | 304.4   | 0.21x |
| torch   | 94.4 us  | 1456.4  |       |

## 4. 128B Swizzle

Shared memory is 32 banks of 4 bytes, wrapping every 128 bytes. With `tile_k = 64`, one row of A is 64 fp16 = exactly 128 bytes, and the row stride is also 128 bytes, so **every row starts at bank 0**. TMA writes many rows concurrently and they all pile onto the same banks in the same order.

So to fix this, we chop each 128-byte row into eight 16 byte chunks and XORs the chunk index with the row index

- **A 128B swizzle requires `tile_k >= 64`**, because it needs 64 contiguous fp16. At `tile_k = 16` the atom does not fit the tile and the layout is rejected outright. `tile_k = 64` on its own was measured and did nothing (448.5 us). Only the pair works, which is why this took until rung 4 to find: each half alone looks like a dead end.

- **The swizzle rides on the pointer, not the layout.** `make_fragment_A` rejects a composed layout, so `.inner` (the swizzle) goes to `recast_ptr` and `.outer` (the affine part) carries the appended stage mode. The base pointer needs 1024-byte alignment, since the pattern spans 8 x 128 B.

Results:

| Problem size     | Kernel  | Time (us) | TFLOP/s | Ratio |
|------------------|---------|-----------|---------|-------|
| 4096^3, 50 iters | gemm_06 | 109.3     | 1256.9  | 0.86x |
|                  | torch   | 93.8      | 1465.5  |       |
| 8192^3, 20 iters | gemm_06 | 862.9     | 1274.2  | 0.83x |
|                  | torch   | 719.4     | 1528.4  |       |


## 5. Pipelined Epilogue

The epilogue moves the accumulator out in four chunks, each TMEM -> RMEM -> convert -> GMEM. Rungs 1-4 reuse a single `acc_frag`/`out_frag` pair for all four, which looks like a write-after-read hazard: chunk `i+1`'s TMEM read wants the registers chunk `i`'s conversion is still using. This rung double-buffers them and issues chunk `i+1`'s read before converting chunk `i`.

For scale: the whole epilogue is at most ~13 us of the 109, so even a perfect version could not have been worth more than ~12%.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_07 | 109.8 us | 1251.6  | 0.86x |
| torch   | 94.6 us  | 1452.5  |       |

## 6. TMA Multicast

At 4096^3 the grid is 32 x 16 = 512 CTAs. Each CTA loads the A tile selected by `bidx` and the B tile selected by `bidy`, so the 16 CTAs sharing a `bidx` each independently request the same A tile. 

Multicast makes one TMA request deliver into several CTAs' SMEM at once, which requires them to be in a cluster. With `cluster = (2, 2)`, `tma_partition` splits each tile across the multicast group so every CTA *issues* half a tile and *receives* a whole one. Both A and B get 2x, halving the bytes requested from L2.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_08 | 108.9 us | 1262.5  | 0.86x |
| torch   | 93.5 us  | 1469.6  |       |

## 7. 2-CTA tcgen05

With `CtaGroup.TWO`, a **pair** of CTAs issues one 256-wide MMA together. Each supplies half the operands from its own SMEM and holds half the accumulator in its own TMEM. `cluster = (2, 1)`, just the pair, no multicast.

There is **one `full` barrier per pair** (the leader's) armed for the whole pair's bytes (`tx_count = (A + B) x 2`), and both CTAs' copies land on it. Only the leader arms it and only the leader waits on it, because only the leader issues the MMA. Treating the pair as two CTAs with private half-sized pipelines was the mis-model underneath most of the debugging.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_09 | 106.4 us | 1291.6  | 0.89x |
| torch   | 94.4 us  | 1455.7  |       |