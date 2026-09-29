Based on the profiling output of the previous kernel (2cta tcgen05), we concluded that we don't need more parallelism, what we need is overlap. So basically, we need to the tile i + 1's mainloop issuing MMAs while tile i's epilogue drains.

# What exactly is persistent kernel?
To understand what persistent kernel is, we need to understand what a wave is. In GPU, a wave is a batch of threads blocks that can be assigned to all the available SMs (Streaming Multiprocessors) in the GPU at once. So, kernel execute can be viewed as sequence of waves processed one after the other until all the waves are completed. A persistent kernel means that we would control the scheduling of the block tile coords rather than the hardware.

# TMEM Double Buffering
On blackwell, `tcgen05` doesn't accumulate into registers the way hoppers `wgmma` did. It accumulates into TMEM (Tiling Memory) instead. The epilogue warps then use tcgen05.ld to copy the finished accumulator out of TMEM into registers, and from there it goes to smem and global memory. So, we can split TMEM into two independent accumulator regions, slot 0 and slot 1, and alternate between them: tile `i` goes into slot `i % 2`.

```
Double buffer (two accumulator slots)

MMA warp:  [ tile 0 -> slot 0 ][ tile 1 -> slot 1 ][ tile 2 -> slot 0 ][ tile 3 -> slot 1 ]
Epilogue:                     [ drain slot 0   ][ drain slot 1   ][ drain slot 0   ]
```

While the epilogue drains tile 0 from slot 0, the MMA warp is already accumulating tile 1 into slot 1. When tile 1's mainloop finishes, slot 0 has (ideally) been fully read, so tile 2 can go there. The tensor cores never wait for the epilogue. It's like the same idea as double-buffering the A/B tiles in shared memory, applied to the output side of the pipeline instead of the input side.

```python
# MMA warp
for tile_idx, tile in enumerate(my_tiles):
    s = tile_idx % 2
    wait(acc_empty[s]) # slot free?
    for k in range(k_blocks):
        wait(smem_full[k_stage])
        mma(acc[s], A, B, accumulate=(k != 0)) # k=0 overwrites old contents
        commit(smem_empty[k_stage]) # free the A/B smem stage
    commit(acc_full[s]) # tile done -> epilogue

# Epilogue warps
for tile_idx, tile in enumerate(my_tiles):
    s = tile_idx % 2
    wait(acc_full[s])
    regs = tmem_load(acc[s])
    fence / wait for the loads
    arrive(acc_empty[s]) # release the slot ASAP
    convert(regs) -> smem -> TMA store to C

```

# Limits
- Capacity: Two slots have to fit in 512 columns. An fp32 accumulator for a 128×N tile takes N columns, so N ≤ 256 per slot.
- It only fully hides the epilogue if the epilogue is shorter than the mainloop. With small K (few k-blocks), the mainloop can be shorter than the drain. Then the MMA warp still waits on acc_empty, just for less time than before. That's a reason to keep the epilogue lean, for example by subtiling it and releasing TMEM early.
- The first tile's mainloop and the last tile's epilogue can't overlap with anything. That fixed cost gets amortized over more tiles per CTA.
