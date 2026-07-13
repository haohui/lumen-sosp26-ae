# Optimization 05: pipeline batch4

Add a compact software pipeline to the batch4 kernel in `input_model.py`.
Preserve both batch2 kernels exactly as received.

Keep the 224x256x64 tile, raw-buffer operations, contiguous LDS layout,
interleaved-M MFMA mapping, accumulator mapping, and x4 output writeback. This
round changes only batch4 operation timing and buffering.

## Structure and buffering

Retain the existing batch4 factory and its single JIT helper set. The kernel
must call the existing load, LDS-store, LDS-read, matmul, and writeback helpers;
do not paste or duplicate helper bodies in the hot loop or epilogue.

Use exactly:

- one `reg_a/reg_b` global-prefetch buffer;
- one contiguous LDS allocation for A and one for B;
- four reusable K16 operand sets `data0`, `data1`, `data2`, and `data3` for
  each operand, each retaining the input flat `(tile, 2)` `u32` shape.

Store the prefetched registers to LDS before loading another tile into them.
Do not add ping-pong LDS, padding, scheduler hints, stagger, or workgroup
remapping.

## Pipeline sequence

Each K64 tile has four K16 batches. Define `k_total = k // GROUP_K`; selected
workloads have `k_total >= 2`.

Prologue:

1. Zero accumulators.
2. Load global tile 0 into `reg_a/reg_b`.
3. Store tile 0 to LDS and synchronize.
4. Read tile 0 batch 0 into `data0`.
5. Prefetch global tile 1 into `reg_a/reg_b`.

For each steady-state tile:

1. Read batch 1 into `data1`; MFMA batch 0 from `data0`.
2. Read batch 2 into `data2`; MFMA batch 1 from `data1`.
3. Read batch 3 into `data3`; MFMA batch 2 from `data2`.
4. Synchronize before overwriting LDS.
5. Store the prefetched next tile to LDS.
6. Load the following global tile into the reused `reg_a/reg_b`.
7. Synchronize before reading the new LDS tile.
8. Read its batch 0 into `data0`; MFMA batch 3 of the old tile from `data3`.

Use a compact loop equivalent to:

```text
for k_idx in range(0, k_total - 2):
    execute the sequence above
```

Epilogue:

1. Drain batches 1, 2, and 3 of the tile currently in LDS, interleaving each
   read with the preceding MFMA.
2. Synchronize, store the final prefetched tile to LDS, and synchronize.
3. Drain all four batches of the final tile while completing the preceding
   operand-set MFMA.
4. Call the unchanged `_write_results`.

Every K tile must contribute its four K16 batches exactly once.

## Verification

Benchmark 4096 against `input_model.py` first. It must be correct and must not
materially regress. Then run all five sizes with the command in `AGENTS.md`.

Finish only when:

- every record has `"correctness":true`;
- 1024/2048 code and performance remain unchanged;
- batch4 has one non-duplicated helper set;
- exactly four batch4 operand sets per A/B exist;
- one global register buffer and one LDS buffer per operand are used;
- raw-buffer widths, layouts, MFMA count, and writeback remain unchanged.
