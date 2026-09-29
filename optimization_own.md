# Which optimizations were made

1. K-loop, 1 stage (baseline)
2. 4-stage pipeline
3. Warp specialization
4. 128B swizzle
5. Pipelined epilogue
6. TMA multicast
7. 2-CTA tcgen05
8. Persistent kernel with double-buffered TMEM accumulator

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

# Prerequisites

## mbarrier

An mbarrier (memory barrier object) is a 64-bit synchronization object that lives in the shared memory. A regular `__syncthreads()` basically means "every threads in the block stop here until all have arrived", but an mbarrier is more like a counter that tracks both the threads and the bytes. Its states has 4 parts:
- expected arrival count is set at init like 1 for a single producer thread or 128 for a warpgroup
- pending arrival count starts at the expected count and it's decremented by each `arrive`
- transaction count is the number of bytes that's expected to land
- phase bit is a single bit that flips each time the barrier completes

A single phase is completed when the arrival reaches 0 and transaction count reaches 0, after this the phase bit flips and the count resets, and anyone waiting on the phase is released. Here the waiters don't wait for a count, they wait for the phase parity to change, which is why we always pass a phase bit to wait.

High level overview of why we need mbarrier is for syncing GMEM to SMEM loads. TMA is asynchronous and runs outside the threads.

With TMA, one thread issues the copy and moves on immediately, the data later gets written to the SMEM by the TMA unit through "async proxy". No thread knows when the data has landed that's why `__syncthreads()` can't help here. Every thread could pass the barrier while the bytes are still in flight. This is what mbarrier basically solves
- The producer thread says, "I've arrived and expect N more bytes"
- It issues the TMA with that barrier attached
- As the bytes-land, the tma hardware decrements the transaction count
- when the transaction count hits 0, the phase flips and consumers waiting wake up with a guarantee that data is visible to SMEM


## TMA
It's a feature introduced in the hopper architecture gpus, to speed up the data transfer from gmem (global memory) to smem (shared memory) within the CTA (cooperative thread arrays / blocks). 

TMA offers number of improvements like (1) improving the gpu utilization via warp-specialized kernels (2) handling the computation of auxiliary data in a single threaded manner via the TMA descriptor, which is more register efficient and handles the necessary predication.

TMA operations: 
- TMA load
- TMA Stores

Basically, TMA load copies ("loads") the data from GPU's GMEM into one of it's CTA's SMEM and TMA store copies ("stores") the data from CTA's SMEM back to the GPU's GMEM

Also TMA is asynchronous operation (excuted in async proxy), we use certain memory consistency enforcement tools like async memory barrier (`mbarrier`) async memory fence (ie `fence.proxy.async`) so that kernel behavior is correct. 

|              | TMA load            | TMA store            |
| ------------ | ------------------- | -------------------- |
| Direction    | GMEM - > SMEM       | SMEM - > GMEM        |
| Sync Method  | Memory barrier      | ProxyFence           |
| When to sync | After the operation | Before the operation |

## TMEM
Tensor Memory or TMEM is an on-chip memory on blackwell (sm_100), separate from shared memory and registers. Each SM has 256kb of it and it exists for tensor cores too. We can think of tmem as a 2D grid:
- 128 lanes (rows) x 512 columns, where each cell is of 32 bits
- An FP32 accumulator tile of 128×N takes 128 lanes × N columns.

Why does tmem exists?
On Hopper `wgmma` kept accumulators in the registers and that caused these problems:
-  Accumulators ate most of the register file, which limited how big our tiles could be.
- A whole warpgroup (128 threads) had to own the MMA and hold those registers.
- The epilogue competed with the mainloop for the same registers.

Blackwell moves accumlators out from registers and into tmem. The tensor core owns the result, so a single thread can issue the MMA, and our registers are free for other work.  We use it like this:
- Allocate. One warp calls `tcgen05.alloc` for some number of columns (a power of 2, at least 32). The hardware writes the TMEM address into SMEM so other warps can read it. At the end of the kernel we `dealloc`. It's like `malloc`/ `free`, but on-chip.
- MMA writes into it. The MMA warp issues `tcgen05.mma` with the TMEM address as the accumulator. This is where the `tcgen05.commit` and `mbarrier` from before come in: they signal "the accumulator is done."
- The epilogue reads it out. Registers can't do math directly on TMEM. Epilogue warps use `tcgen05.ld` to copy TMEM into registers, then convert and scale, then store to GMEM (usually through SMEM and TMA).

TMEM is a dedicated 128×512 scratchpad where tensor cores keep accumulators. It frees our registers, lets one thread drive the MMA, and gives us room to double-buffer. We read it out 32 lanes per warp.

# The Starting Point

The kernel before the first rung (see [PR #3](https://github.com/Vishal-Padia/blackwellize/pull/3)) is the starting point, and we started making optimizations on top of it. It's very minimal, and here each CTA loads a single MMA tile of A and B from global memory to shared memory via `cute.copy()` behind an mbarrier, issues one UMMA through `cute.gemm()`, and writes to accumulator. 

### What the profile says

The warp state statistics say: "On average, each warp of this workload spends 19.5 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation." That stall is about 48.8% of the 40.1 cycles between issuing two instructions.

Why: this is the mbarrier prereq made concrete. Every thread in the CTA calls `cute.arch.mbarrier_wait(ab_full_mbar, 0)` after warp 0 issues the two TMA loads, and later warp 0 alone calls `cute.arch.mbarrier_wait(mma_done_mbar, 0)` after the single `cute.gemm()`. An mbarrier wait is implemented as a poll on a shared-memory location, which routes through the L1TEX/LSU pipe. So "stalled on L1TEX scoreboard" is just threads sitting at those two `mbarrier_wait` calls, doing nothing while the TMA hardware and tensor core finish their work. That is also the root of the occupancy gap: even with several blocks resident per SM, all of them hit the same wait at roughly the same time (one load, one MMA, no loop to stagger them), so no other warp has independent work ready to fill the gap. Everyone stalls together.

What fixes it: other work has to be available during the stall, and this kernel has none. Two mechanisms fix exactly this shape of problem and they are the next rungs:
- Pipelining (rung 2): only pays off once there is a K-loop. With multiple SMEM stages, iteration i+1's load is in flight while iteration i's MMA runs.
- Warp specialization (rung 3): warp 0 and warp 1 get separate roles, so one warp's stall does not block the other's progress.

# Optimization Rungs

## 1. K-loop, 1 stage (baseline)

[PR #4](https://github.com/Vishal-Padia/blackwellize/pull/4)

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

Across shapes, gemm_03 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 840.1 | 94.9 | 0.11x |
| 8192, 8192, 8192 | 5977.0 | 740.6 | 0.12x |
| 8192, 28672, 8192 | 20962.1 | 2406.8 | 0.11x |
| 256, 8192, 8192 | 449.9 | 32.3 | 0.07x |
| 256, 10240, 8192 | 445.4 | 38.2 | 0.09x |
| 256, 28672, 8192 | 872.7 | 110.9 | 0.13x |
| 256, 8192, 28672 | 1567.8 | 112.0 | 0.07x |

### What the profile says

Cycles between issued instructions jumped from 40 (starting point) to 167.6, and 73.8% of that (123.7 cycles) is "stalled waiting for sibling warps at a CTA barrier". This is not the L1TEX stall from before.

Why: the entire K-loop sits under `if warp_idx == 0`, so warps 1, 2 and 3 do nothing for the whole mainloop while warp 0 alone runs all 256 iterations of load, wait, MMA, wait.

What fixes it: it points at one thing. The rest of the warps need work. That is warp specialization, but it only helps if the loop is first fast enough for the extra warps to matter, so pipelining comes first.

## 2. 4-stage pipeline

[PR #5](https://github.com/Vishal-Padia/blackwellize/pull/5)

**`num_stages` is the number of SMEM buffers, not a split of K**. The two are independent.

The loops still runs like 256 times, the change is how we load from global memory.

We fill the buffer with tiles 0,1,2,3 and then iteration `k` waits on on `ab_full_mbar + stage`, and issues the MMA, and refills that same slot with tile ` k + 4`. Since the refill targets the slot just consumed, and the MMA-done wait sits above it, no separate "empty" barrier is needed yet.

Each barrier is now reused once per lap around the ring rather than once per iteration, so full `full_phase` flips when `stage` wraps to 0, while `done_phase` still flips every iteration. 

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_04 | 451.7 us | 304.3   | 0.21x |
| torch   | 95.0 us  | 1447.1  |       |

Across shapes, gemm_04 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 448.2 | 94.9 | 0.21x |
| 8192, 8192, 8192 | 3069.5 | 740.6 | 0.24x |
| 8192, 28672, 8192 | 10685.6 | 2406.8 | 0.23x |
| 256, 8192, 8192 | 222.0 | 32.3 | 0.15x |
| 256, 10240, 8192 | 224.9 | 38.2 | 0.17x |
| 256, 28672, 8192 | 453.1 | 110.9 | 0.24x |
| 256, 8192, 28672 | 761.5 | 112.0 | 0.15x |

### What the profile says

Duration dropped 856 us to 471 us and warp cycles per issued instruction dropped 167.6 to 59.63. But look at Speed of Light: L1/TEX cache throughput jumped to 98.90%, almost the ceiling, while DRAM throughput is 2.26% and L2 hit rate is 95.62%.

Why: the CTA barrier is still about 73% of the cycles, and this rung did not touch that. It made warp 0's own mainloop faster per iteration, with 4 SMEM slots in flight instead of 1, so there is less waiting for any single tile to land.

The L1/TEX number is the more interesting one, and it is a trap if read at face value. 98.9% looks like the shared-memory pipe is moving real bandwidth, but DRAM is barely touched and L2 is hitting 95.6% of the time, so almost no bytes are actually moving. Near-100% L1/TEX with near-0% DRAM is the signature of a pipe saturated by request count rather than bytes: many small, badly distributed SMEM accesses, i.e. bank conflicts. Swizzling is what solves it.

The whole speedup of this rung came from pipeline overlap, not from touching the SMEM access pattern.

## 3. Warp Specialization

[PR #7](https://github.com/Vishal-Padia/blackwellize/pull/7)

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

Across shapes, gemm_05 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 449.4 | 94.9 | 0.21x |
| 8192, 8192, 8192 | 3072.7 | 740.6 | 0.24x |
| 8192, 28672, 8192 | 10685.3 | 2406.8 | 0.23x |
| 256, 8192, 8192 | 221.6 | 32.3 | 0.15x |
| 256, 10240, 8192 | 223.7 | 38.2 | 0.17x |
| 256, 28672, 8192 | 452.8 | 110.9 | 0.24x |
| 256, 8192, 28672 | 758.6 | 112.0 | 0.15x |

### What the profile says

Not much changed. Duration is flat (471 us to 477 us), L1/TEX is still at 98%, DRAM still at 2.29%. The one thing that moved is the CTA-barrier stall, which shrank from 43.3 to 37.1 cycles.

Why: warp 0 now runs its own producer loop and warp 1 runs its own consumer loop, so 2 of 4 warps are active during the mainloop instead of 1. Warps 2 and 3 still hit `sync_threads()` and idle the whole time, just with fewer siblings waiting alongside them. So the barrier stall really did shrink.

But wall-clock did not move, because the L1/TEX pipe was already the binding constraint (about 99% since rung 2). Warp 0 and warp 1's loops were gated by how fast SMEM/TMA traffic could clear L1/TEX, and shaving idle time off warps 2 and 3 does nothing for them. We optimized a stall that was not the bottleneck.

What fixes it: this rung is the textbook case of fixing a real, measurable inefficiency that turns out not to be on the critical path once you check what the actual bottleneck is doing. The bottleneck is L1/TEX saturation, so we swizzle.

## 4. 128B Swizzle

[PR #8](https://github.com/Vishal-Padia/blackwellize/pull/8)

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

Across shapes, gemm_06 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 110.0 | 94.9 | 0.86x |
| 8192, 8192, 8192 | 865.2 | 740.6 | 0.86x |
| 8192, 28672, 8192 | 3062.9 | 2406.8 | 0.79x |
| 256, 8192, 8192 | 54.4 | 32.3 | 0.59x |
| 256, 10240, 8192 | 59.8 | 38.2 | 0.64x |
| 256, 28672, 8192 | 127.4 | 110.9 | 0.87x |
| 256, 8192, 28672 | 165.9 | 112.0 | 0.67x |

### What the profile says

Duration went from 477 us to 116.2 us, a 4.1x speedup. L1/TEX cache throughput is 68.79% now and compute throughput jumped from 19.77% to 64.8%.

Why: at rung 2 we already knew the 99% L1/TEX was never real bandwidth pressure (DRAM was at 2%), it was SMEM bank conflicts eating the pipe's request slots. Every row of the A/B tile was landing on bank 0 (128-byte rows, 128-byte stride), so the TMA's concurrent writes across rows all serialized onto the same banks. The swizzle spreads those writes across banks. Once the self-inflicted SMEM bottleneck is gone, the tensor cores actually get fed, and that is why compute throughput jumped.


## 5. Pipelined Epilogue

[PR #9](https://github.com/Vishal-Padia/blackwellize/pull/9)

The epilogue moves the accumulator out in four chunks, each TMEM -> RMEM -> convert -> GMEM. Rungs 1-4 reuse a single `acc_frag`/`out_frag` pair for all four, which looks like a write-after-read hazard: chunk `i+1`'s TMEM read wants the registers chunk `i`'s conversion is still using. This rung double-buffers them and issues chunk `i+1`'s read before converting chunk `i`.

For scale: the whole epilogue is at most ~13 us of the 109, so even a perfect version could not have been worth more than ~12%.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_07 | 109.8 us | 1251.6  | 0.86x |
| torch   | 94.6 us  | 1452.5  |       |

Across shapes, gemm_07 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 110.2 | 94.9 | 0.86x |
| 8192, 8192, 8192 | 864.3 | 740.6 | 0.86x |
| 8192, 28672, 8192 | 3053.9 | 2406.8 | 0.79x |
| 256, 8192, 8192 | 54.2 | 32.3 | 0.60x |
| 256, 10240, 8192 | 59.6 | 38.2 | 0.64x |
| 256, 28672, 8192 | 127.3 | 110.9 | 0.87x |
| 256, 8192, 28672 | 165.8 | 112.0 | 0.68x |

### What the profile says

Nothing moved outside measurement noise. Duration 116.2 us to 112.93 us, compute throughput 64.68% to 64.15%, L1/TEX 68.79% to 68.32%, warp cycles per issued instruction 31.40 to 31.64.

Why: the whole epilogue is at most about 13 us of a 113 to 116 us kernel, so even eliminating it completely could not move total time by more than about 12%. This rung fixes a write-after-read hazard inside that 13 us slice, but the mainloop, the other 100 us and where all the Speed of Light numbers are dominated, is untouched.

## 6. TMA Multicast

[PR #10](https://github.com/Vishal-Padia/blackwellize/pull/10)

At 4096^3 the grid is 32 x 16 = 512 CTAs. Each CTA loads the A tile selected by `bidx` and the B tile selected by `bidy`, so the 16 CTAs sharing a `bidx` each independently request the same A tile. 

Multicast makes one TMA request deliver into several CTAs' SMEM at once, which requires them to be in a cluster. With `cluster = (2, 2)`, `tma_partition` splits each tile across the multicast group so every CTA *issues* half a tile and *receives* a whole one. Both A and B get 2x, halving the bytes requested from L2.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_08 | 108.9 us | 1262.5  | 0.86x |
| torch   | 93.5 us  | 1469.6  |       |

Across shapes, gemm_08 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 109.4 | 94.9 | 0.87x |
| 8192, 8192, 8192 | 918.2 | 740.6 | 0.81x |
| 8192, 28672, 8192 | 3205.3 | 2406.8 | 0.75x |
| 256, 8192, 8192 | 54.5 | 32.3 | 0.59x |
| 256, 10240, 8192 | 59.8 | 38.2 | 0.64x |
| 256, 28672, 8192 | 126.3 | 110.9 | 0.88x |
| 256, 8192, 28672 | 167.0 | 112.0 | 0.67x |

### What the profile says

Duration is flat again (112.93 us to 113.41 us). But L1/TEX cache throughput dropped from 68.32% to 59.22%, and the Speed of Light summary flipped from "Compute and Memory are well-balanced" to "Compute is more heavily utilized than Memory."

Why: multicast does exactly what it is supposed to. It halves the redundant bytes each CTA requests and that shows up as real relief on the L1/TEX pipe. So this is not a null result in the way rung 3 was, the mechanism worked. It is a null result because the kernel was never memory-bound at this shape: DRAM throughput has sat at 9 to 10% since rung 4, and compute throughput has been the higher number (64 to 65%) since the swizzle fixed the real bottleneck. Multicast relieves memory-traffic pressure on a kernel that is already compute-bound, which makes a resource that was not the constraint even less constrained.

## 7. 2-CTA tcgen05

[PR #11](https://github.com/Vishal-Padia/blackwellize/pull/11)

With `CtaGroup.TWO`, a **pair** of CTAs issues one 256-wide MMA together. Each supplies half the operands from its own SMEM and holds half the accumulator in its own TMEM. `cluster = (2, 1)`, just the pair, no multicast.

There is **one `full` barrier per pair** (the leader's) armed for the whole pair's bytes (`tx_count = (A + B) x 2`), and both CTAs' copies land on it. Only the leader arms it and only the leader waits on it, because only the leader issues the MMA. Treating the pair as two CTAs with private half-sized pipelines was the mis-model underneath most of the debugging.

Results (4096^3, 50 iters):

| Kernel  | Time     | TFLOP/s | Ratio |
|---------|----------|---------|-------|
| gemm_09 | 106.4 us | 1291.6  | 0.89x |
| torch   | 94.4 us  | 1455.7  |       |

Across shapes, gemm_09 against torch:

| Problem size (M, N, K) | Time (us) | torch (us) | Ratio |
|---|---|---|---|
| 4096, 4096, 4096 | 107.3 | 94.9 | 0.88x |
| 8192, 8192, 8192 | 847.8 | 740.6 | 0.87x |
| 8192, 28672, 8192 | 3009.7 | 2406.8 | 0.80x |
| 256, 8192, 8192 | 47.7 | 32.3 | 0.68x |
| 256, 10240, 8192 | 51.8 | 38.2 | 0.74x |
| 256, 28672, 8192 | 116.6 | 110.9 | 0.95x |
| 256, 8192, 28672 | 137.9 | 112.0 | 0.81x |

### What the profile says

Compute throughput is 61.96% and memory throughput 39.43%. Nothing is above 62%, which means nothing is saturated: the kernel is latency-bound. Occupancy tells the same story. Theoretical occupancy is 6.25%, only one block resident per SM, because static shared memory per block is 197.63 KB of the 200.70 KB available (6 pipeline stages of 32 KB fill SMEM almost to the brim).

Why: with only one block per SM, there is no second block to keep the MMA pipe fed while this block's epilogue drains. The CTA-barrier stall (19.5 of 35.86 cycles, 54.4%) is that drain: the mainloop finishes, `sync_threads()` plus cluster wait, 128 KB gets written out, and during that whole stretch the tensor core is idle because nothing else is scheduled on the SM. Waves per SM is 3.46, which compounds it: three full waves plus a 47%-full partial wave, so the tail wave wastes over half an SM's worth of work every time.

What fixes it: the problem is no longer parallelism, it is overlap. The fix is a persistent kernel with a double-buffered TMEM accumulator: launch exactly 148 blocks (74 pairs), loop over output tiles inside the kernel, so tile i+1's mainloop is already issuing MMAs while tile i's epilogue drains. That removes both the barrier stall and the tail wave in one change.


## 8. Persistent kernel with double-buffered TMEM accumulator

`gemm_10_persistent_kernel.py`. Background is in `persistent_kernel.md`: a wave is one batch of thread blocks filling all SMs at once, a persistent kernel means we control the scheduling of tile coordinates instead of the hardware, and TMEM is split into two accumulator slots so tile i goes into slot `i % 2` while the epilogue drains slot `(i - 1) % 2`.

Results:

| Problem size (M, N, K)   | Kernel  | Time (us) | TFLOP/s | Ratio |
|--------------------------|---------|-----------|---------|-------|
| 4096, 4096, 4096         | gemm_10 | 98.0      | 1401.7  | 0.97x |
|                          | torch   | 94.9      | 1448.9  |       |
| 8192, 8192, 8192         | gemm_10 | 815.9     | 1347.7  | 0.91x |
|                          | torch   | 740.6     | 1484.6  |       |
| 8192, 28672, 8192        | gemm_10 | 2933.6    | 1311.8  | 0.82x |
|                          | torch   | 2406.8    | 1598.9  |       |
| 256, 8192, 8192          | gemm_10 | 47.9      | 717.6   | 0.67x |
|                          | torch   | 32.3      | 1064.5  |       |
| 256, 10240, 8192         | gemm_10 | 51.9      | 827.0   | 0.74x |
|                          | torch   | 38.2      | 1123.3  |       |
| 256, 28672, 8192         | gemm_10 | 113.3     | 1061.0  | 0.98x |
|                          | torch   | 110.9     | 1084.6  |       |
| 256, 8192, 28672         | gemm_10 | 138.2     | 870.1   | 0.81x |
|                          | torch   | 112.0     | 1073.9  |       |


Timing is CUDA events, 5 to 30 iterations depending on shape (`experiments/rung_sweep.py`, raw data in `results/rung_sweep.json`). On the decode shapes with N=8192 or 10240 there are already fewer tiles than SM pairs (32 and 40 tiles for 74 pairs), so there is no second tile to overlap and the persistent kernel changes nothing. The gain is in the prefill shapes and in decode MLP up, where 112 tiles are spread over 74 pairs.

### What the profile says

The grid is 148 blocks and waves per SM went from 3.46 to 1. Compute throughput went from 61.96% to 74.58%. The top stall at rung 7 was the CTA barrier (19.5 of 35.9 cycles). Now it is L1TEX scoreboard, 42.3 of 55.2 cycles.

Why: at rung 7, `sync_threads` plus `cluster_wait` before every epilogue left the tensor core idle during the drain. Now the MMA warp moves straight to the other TMEM slot, so the tensor core is busy about 74.6% of the time instead of 62%.

"L1TEX scoreboard" is warps waiting in `mbarrier_wait`, the same signature as the starting point. In a warp-specialized kernel each role spends most of its time waiting for the next role: the epilogue on `acc_full`, the TMA warp on `ab_empty`. That is the handoff working, not a bottleneck. The number that matters is tensor core utilization, and the Source page in the ncu GUI should show those stalls sitting on the `mbarrier_wait` lines.

Limits (from `persistent_kernel.md`): two accumulator slots must fit in 512 TMEM columns so N is at most 256 per slot; the epilogue is only fully hidden if it is shorter than the mainloop, which fails at small K; the first tile's mainloop and the last tile's epilogue cannot overlap with anything.
