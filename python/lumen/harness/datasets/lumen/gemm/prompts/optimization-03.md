# Optimization 03: vectorize global memory operations

Optimize the MFMA GEMM in `input_model.py` by replacing its scalar global
memory accesses with AMD raw-buffer operations. Preserve the complete GEMM
layout established by the previous round.

This round changes how existing register fragments are transferred between
global memory and registers. It must not change which logical values those
fragments contain or which MFMA lanes consume them.

The public API remains:

```python
gemm_pipeline_transposed_b(A, B, out=None) -> torch.Tensor
```

The result remains `C = A @ B.T`, with BF16 inputs/output and FP32 MFMA
accumulation.

## Input layout and activation contract

Do not change any of the following:

- batch2 uses 64x64x64 for 1024 and 128x128x64 for 2048;
- the existing batch4 family uses a 224x256x64 tile;
- global-fragment ownership formulas;
- global-fragment-to-LDS ownership;
- the current contiguous row-major LDS address formulas;
- LDS-to-MFMA lane ownership;
- batch2's two K32 batches;
- batch4's four K16 batches and interleaved-M mapping;
- MFMA operand order or number of MFMA operations;
- accumulator-to-C mapping;
- the direct-register writeback rule that packs neighboring C columns;
- K-loop ordering or synchronization placement.

In particular, do not transpose or reswizzle LDS, swap MFMA operands, add
`al.shuffle`, or stage C through shared memory.

The input keeps large public sizes on the batch2 fallback because batch4 still
uses scalar global accesses. After converting and validating the batch4 raw
buffer path in this round, dispatch 4096, 8192, and 16384 to batch4. Keep 1024
and 2048 on their existing batch2 configurations. Do not activate batch4 if it
is incorrect or materially slower than the input fallback; fix the raw-buffer
implementation first.

## Full-tensor buffer resources

Create A, B, and C buffer resources once before the K loop and reuse them for
all loads/stores:

```python
a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)
```

Resource ranges are byte counts. Use the full logical tensor ranges so an
out-of-range batch4 M-tile load returns zero and an out-of-range store is
discarded by the buffer operation.

Keep block/tile base offsets in `soffset` and thread/fragment offsets in
`vindex` where possible. Compute all offsets in bytes.

Do not create a new resource inside the K loop or inside a per-fragment loop.

## Batch2 global loads

The existing batch2 ownership is one 8-BF16, 16-byte fragment per logical
load:

```text
elem_base = (load_idx * NUM_THREADS + tid) * VEC_SIZE
tile_row = elem_base // GROUP_K
tile_col = elem_base % GROUP_K
```

Preserve exactly this ownership. For a fixed `load_idx`, adjacent threads must
load adjacent 16-byte fragments.

Replace the loop that loads eight individual BF16 values with one:

```python
al.amdgpu.raw_buffer_load_x4(...)
```

The returned `(4, al.u32)` value must be placed into the same register fragment
that is already written to the same LDS vector.

Retain the exact batch2 representation established in Optimization 01:

```text
reg_a: al.Tensor((LOADS_PER_THREAD_A, 4), al.u32)
reg_b: al.Tensor((LOADS_PER_THREAD_B, 4), al.u32)
shm_a: al.Tensor((SHM_A_VECTORS, 4), al.u32)
shm_b: al.Tensor((SHM_B_VECTORS, 4), al.u32)
```

Assign the result of `raw_buffer_load_x4` directly to
`reg_[load_idx]`. Do not temporarily convert the register tensor to
`(LOADS_PER_THREAD, VEC_SIZE), al.bf16`; that shape cannot be lowered across
the nested helper boundary. Preserve the explicit same-layout `al.view`
adapters in `_load_global_ab`, `_store_shm_ab`, and `_read_shm_ab`, and keep
calling those combined wrappers from `kernel`.

For operand A, derive offsets equivalent to:

```text
tile_base_bytes = (group_m * GROUP_M * k + k_tile * GROUP_K) * BF16_BYTES
thread_bytes = (tile_row * k + tile_col) * BF16_BYTES
```

For operand B, use `group_n * GROUP_N` instead of `group_m * GROUP_M`.

Do not change to thread-major ownership such as:

```text
(tid * LOADS_PER_THREAD + load_idx) * 8
```

That mapping makes each thread locally contiguous but makes a wave access
strided fragments.

## Batch4 global loads

The existing batch4 ownership is one packed `al.u32`, containing two adjacent
BF16 values:

```text
GLOBAL_WORDS_PER_ROW = GROUP_K * BF16_BYTES // 4
GLOBAL_ROWS_PER_ROUND = NUM_THREADS // GLOBAL_WORDS_PER_ROW
tile_row = tid // GLOBAL_WORDS_PER_ROW
           + load_idx * GLOBAL_ROWS_PER_ROUND
tile_word = tid % GLOBAL_WORDS_PER_ROW
```

Preserve this ownership and replace the two scalar BF16 loads with one:

```python
al.amdgpu.raw_buffer_load_x1(...)
```

The thread-level byte offset is equivalent to:

```text
thread_bytes = tile_row * k * BF16_BYTES + tile_word * 4
```

Use the appropriate A or B block base plus `k_tile * GROUP_K * BF16_BYTES` as
the block/tile offset. Store the returned `al.u32` into the same register word
and then the same contiguous LDS word as before.

Use a flat, one-dimensional `al.u32` representation for the complete batch4
staging path. This compiler-visible structure is required by the later
software-pipeline and scheduling rounds:

```text
LOADS_PER_THREAD_A = GROUP_M // GLOBAL_ROWS_PER_ROUND
LOADS_PER_THREAD_B = GROUP_N // GLOBAL_ROWS_PER_ROUND

reg_a: al.Tensor((LOADS_PER_THREAD_A,), al.u32)
reg_b: al.Tensor((LOADS_PER_THREAD_B,), al.u32)
shm_a: al.Tensor((GROUP_M * GLOBAL_WORDS_PER_ROW,), al.u32)
shm_b: al.Tensor((GROUP_N * GLOBAL_WORDS_PER_ROW,), al.u32)
```

Each load writes directly to `reg_a[load_idx]` or `reg_b[load_idx]`. Each LDS
store writes that word directly to `shm_a[physical_word]` or
`shm_b[physical_word]`. The LDS-read helpers read two adjacent flat words into
the existing `(tile, 2)` MFMA operand tensors.

Do not represent these flat words as `(REG_VECTORS, 4)` or
`(SHM_VECTORS, 4)` tensors. Do not index them with `idx // 4, idx % 4`, and do
not retain split `reg_a0/reg_a1` or `reg_b0/reg_b1` buffers. Flattening here
does not change logical ownership or LDS layout; it removes artificial vector
grouping around scalar x1 operations.

Keep exactly one batch4 `_load_global_a`, `_load_global_b`, and
`_load_global_ab`; exactly one `_store_shm_a`, `_store_shm_b`, and
`_store_shm_ab`; and exactly one `_read_shm_a`, `_read_shm_b`, and
`_read_shm_ab`. The combined wrappers must call their A/B helpers rather than
inlining repeated bodies into the kernel. The kernel owns only one flat
`reg_a/reg_b` prefetch buffer and one flat `shm_a/shm_b` allocation.

Do not widen batch4 loads by changing its ownership. A future instruction-level
tuning decision may combine operations, but this round must retain the batch4
layout established in Optimization 02.

## Raw-buffer writeback

Replace ordinary tensor/view stores with raw-buffer stores while preserving
the existing accumulator-to-C mapping.

Continue to pack neighboring `n_tile` accumulator values, with identical
`m_tile` and `acc_idx`, into BF16 words using:

```python
lo = al.bitcast(value0, al.u32)
hi = al.bitcast(value1, al.u32)
packed_bf16x2 = al.amdgpu.perm(hi, lo, 0x07060302)
```

Use the widest naturally contiguous store supported by each fixed layout:

- for the 64x64 batch2 config, store one packed `al.u32` with
  `raw_buffer_store_x1`;
- for the 128x128 batch2 config, combine two packed words and use
  `raw_buffer_store_x2`;
- for batch4, combine four packed words for eight adjacent output columns and
  use `raw_buffer_store_x4`.

These wider stores are instruction grouping only. They must not change the
logical output columns owned by a lane.

Keep exactly one `_write_results` helper in each kernel factory. For batch2,
allocate a fixed two-word local packed buffer and use a captured Python
`STORE_VEC` value to select x1 or x2 inside that single helper, following this
structure:

```python
if STORE_VEC == 1:
    al.amdgpu.raw_buffer_store_x1(packed_words[0], ...)
else:
    al.amdgpu.raw_buffer_store_x2(packed_words, ...)
```

Do not define `_store_packed_results` separately in `if/else` branches. Do not
redefine, shadow, or duplicate any load/store/writeback helper for the 64x64
and 128x128 specializations. Factory invocation supplies specialization; each
factory contains one source implementation of each helper.

Put the per-thread `(row, col)` byte offset in `vindex` and the output
block/warp base in `soffset` where practical. Preserve guards needed to prevent
a partial vector at the end of a logical row from spilling into the next row.
For the five selected benchmark shapes, N is a full tile for every selected
configuration.

## Scope restrictions

Do not add any of the following in this round:

- global prefetching or register double buffering;
- K-loop unrolling or software pipelining;
- changes to `al.syncthreads()` placement;
- instruction scheduling hints;
- K-tile staggering;
- XCC-aware or non-row-major workgroup mapping;
- new tile sizes or new shape configs;
- PyTorch or external-library compute fallbacks.

Do not introduce LDS padding in this round. The K loop must remain the simple
sequence:

1. raw-buffer load the current K tile;
2. store the existing fragments to the frozen contiguous LDS layout;
3. synchronize;
4. execute the existing MFMA batches;
5. synchronize before the next K tile.

## Verification

Run the exact benchmark command documented in `AGENTS.md`. During development,
first use `--matrix-sizes 1024 2048` to validate batch2, then
`--matrix-sizes 4096` to validate batch4. Before finishing, run all five sizes.

Compare performance with the input implementation. The expected improvement
should be especially visible for the batch4 workloads because their input
implementation performs individual BF16 global loads.

Final sanity checks:

- no scalar per-BF16 global-load loop remains in either family;
- batch2 A/B loads use `raw_buffer_load_x4`;
- batch4 A/B loads use `raw_buffer_load_x1`;
- batch4 global registers and LDS are flat one-dimensional `u32` tensors;
- batch4 contains one combined load/store/read wrapper per operation and no
  split prefetch buffers or `idx // 4, idx % 4` staging indices;
- writeback uses raw-buffer stores and packs neighboring columns;
- each factory contains exactly one writeback implementation, with no helper
  definitions inside configuration branches;
- A/B/C resources are created outside the K loop and reused;
- 4096, 8192, and 16384 are activated on batch4 only after its raw-buffer path
  passes correctness and the performance guard;
- both families retain exactly the same contiguous LDS and MFMA layouts as the
  input;
- every MFMA call still passes A first and B second;
- all five benchmark records contain `"correctness":true`.
