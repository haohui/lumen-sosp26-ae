# Optimization 04: pipeline batch2

Add a compact software pipeline to the two batch2 kernels in `input_model.py`.
Leave the batch4 factory and its public dispatch unchanged.

Preserve every tile, raw-buffer operation, contiguous LDS address, MFMA lane
mapping, accumulator mapping, and output store from the input. This round
changes only the timing of existing batch2 operations.

Limit the batch2 source diff to the `kernel` body: its local operand-buffer
allocations and K-loop control flow. Keep every existing batch2 helper
definition byte-for-byte unchanged, including the explicit same-layout views
inside `_load_global_ab`, `_store_shm_ab`, and `_read_shm_ab`. Keep the entire
batch4 factory and public dispatch byte-for-byte unchanged.

## Structure and buffering

Keep the existing batch2 factory and its JIT helpers. The kernel hot loop must
compose `_load_global_ab`, `_store_shm_ab`, `_read_shm_ab`, and `_matmul`.
Never paste their load, address, LDS, or MFMA bodies into the kernel or duplicate
them for an unrolled step.

Do not replace a combined wrapper's calls with copied A/B leaf-helper bodies,
and do not bypass a combined wrapper by calling both leaf helpers from
`kernel`. If a helper-boundary compile error appears, restore the input helper
definitions rather than editing or inlining them; they were already compiled
and correctness-tested by the previous round.

Use exactly:

- one `reg_a/reg_b` global-prefetch buffer;
- one contiguous LDS allocation for A and one for B;
- `data_a0/data_b0` and `data_a1/data_b1`, each with the same flat `(tile, 4)`
  `u32` shape as the input operand registers.

Store `reg_a/reg_b` to LDS before reusing them for another global load. Do not
add ping-pong LDS, padding, scheduler hints, stagger, or workgroup remapping.

## Pipeline sequence

For `GROUP_K=64`, each K tile has two K32 batches. Let
`k_total = k // GROUP_K`; benchmark cases have even `k_total >= 4`.

Prologue:

1. Zero accumulators.
2. Load K tile 0 into `reg_a/reg_b`.
3. Store tile 0 to LDS and synchronize.
4. Read tile 0 batch 0 into `data0`.
5. Prefetch global tile 1 into `reg_a/reg_b`.

One K-tile step is:

1. Read current LDS tile batch 1 into `data1`.
2. MFMA its batch 0 from `data0`.
3. Synchronize before overwriting LDS.
4. Store the prefetched next tile to LDS.
5. Load the following global tile into the reused `reg_a/reg_b`.
6. Synchronize before reading new LDS contents.
7. Read the new LDS tile batch 0 into `data0`.
8. MFMA the old tile batch 1 from `data1`.

Unroll the steady-state loop by two K tiles:

```text
for k_idx in range(0, k_total - 3, 2):
    execute one step for k_idx
    execute one step for k_idx + 1
```

Express both steps with calls to the same existing helpers. At loop exit, one
tile is in LDS and one final tile is prefetched in `reg_a/reg_b`.

Epilogue:

1. Read and compute batch 1 of the current LDS tile while completing its batch
   0 MFMA.
2. Synchronize, store the final prefetched tile to LDS, and synchronize.
3. Read and compute batch 0 and batch 1 of the final tile.
4. Call the unchanged `_write_results`.

Every K tile must contribute exactly two K32 batches once. Do not reload or
recompute a tile in the epilogue.

## Verification

First benchmark 1024 and 2048 against `input_model.py`; both must remain
correct and neither may materially regress. A sharp 2048 regression usually
means code expansion or excessive live ranges: refactor back to compact helper
calls rather than keeping the result.

Then run all five sizes with the command in `AGENTS.md`. Finish only when:

- every record has `"correctness":true`;
- batch4 code and large-size performance are unchanged;
- batch2 contains one helper set with no configuration-based redefinitions;
- only two batch2 operand-register sets exist;
- one global register buffer and one LDS buffer per operand are used;
- raw-buffer widths, layouts, and writeback are unchanged.
