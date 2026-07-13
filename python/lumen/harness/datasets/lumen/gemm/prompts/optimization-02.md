# Optimization 02: add the batch4 MFMA family

Extend the batch2 MFMA implementation in `input_model.py` with a batch4 kernel
for the large benchmark sizes. Preserve the existing batch2 factories, helper
structure, layouts, dispatch for 1024/2048, and behavior unchanged.

The public API and result remain:

```python
gemm_pipeline_transposed_b(A, B, out=None) -> torch.Tensor
```

with `C = A @ B.T`, BF16 input/output, and FP32 MFMA accumulation.

## Batch4 configuration

Add `_make_batch4_kernel(...)` with:

```text
GROUP_M = 224
GROUP_N = 256
GROUP_K = 64
WARP_PER_ROW = 2
WARP_PER_COL = 2
```

M uses ceil-div workgroups and the last 224-row tile is partial; global loads
and output stores must guard it. The selected N sizes are divisible by 256.

At the start of the batch4 kernel, use the same wave/lane decomposition as
batch2:

```python
wid = al.amdgpu.readfirstlane(tid // WARP_SIZE)
wtid = tid % WARP_SIZE
```

Instantiate the batch4 kernel once at module scope, but do not make it the
final public dispatch yet. Until its scalar global accesses are replaced by
raw-buffer operations in the next round, keep `gemm_pipeline_transposed_b`
dispatching 1024 to batch2 64x64 and every other selected size to the existing
batch2 128x128 fallback. This prevents an intentionally incomplete batch4
memory path from regressing the public benchmark.

## One batch4 helper set

Follow the batch2 structure. Inside `_make_batch4_kernel`, define exactly one
implementation of:

- `_load_global_a`, `_load_global_b`, `_load_global_ab`;
- `_store_shm_a`, `_store_shm_b`, `_store_shm_ab`;
- `_read_shm_a`, `_read_shm_b`, `_read_shm_ab`;
- `_matmul`, `_write_results`, and `kernel`.

Use factory-captured Python integers directly in tensor annotations and loop
bounds. Do not redefine helpers behind configuration branches, add `_impl`
copies, shadow helpers, or inline their bodies into the main kernel.

## Global ownership and contiguous LDS

Use one packed `al.u32` containing two adjacent BF16 values per global-load
fragment:

```text
GLOBAL_WORDS_PER_ROW = GROUP_K * 2 // 4
GLOBAL_ROWS_PER_ROUND = NUM_THREADS // GLOBAL_WORDS_PER_ROW
tile_row = tid // GLOBAL_WORDS_PER_ROW
           + load_idx * GLOBAL_ROWS_PER_ROUND
tile_word = tid % GLOBAL_WORDS_PER_ROW
```

Load the two BF16 values normally in this round. Raw-buffer operations are
introduced later. Preserve the same `(tile_row, tile_word)` when storing the
packed word to LDS.

Use this exact temporary representation for the scalar batch4 memory path:

```python
reg_a = al.make_local((LOADS_PER_THREAD_A, 2), al.bf16)
reg_b = al.make_local((LOADS_PER_THREAD_B, 2), al.bf16)
shm_a = al.make_shared((GROUP_M, ROW_WORDS), al.u32)
shm_b = al.make_shared((GROUP_N, ROW_WORDS), al.u32)
```

The leaf global-load helpers fill the two BF16 register elements separately.
In `_store_shm_a/b`, pack one register row with this precise AveLang idiom:

```python
packed = al.view(reg_a[store_idx], al.Tensor((1,), al.u32))
shm_a[tile_row, tile_word] = packed
```

Assign `packed` itself. Do not write `packed[0]`: indexing this particular
view is not supported by the current lowerer. Do not begin with a one-element
`u32` register and attempt to construct that word using unsupported scalar
bit operations; the next round replaces this entire scalar path with
raw-buffer loads and flat `u32` staging.

Store each word in a contiguous row-major LDS layout:

```text
ROW_WORDS = GROUP_K * 2 // 4
physical_word = row * ROW_WORDS + word_in_row
```

Do not add LDS padding in this round. Each MFMA operand read consists of two
adjacent `al.u32` words. Padding will be introduced and measured separately.

## Batch4 MFMA mapping

Each K64 tile contains four K16 batches. For the 224x256 tile:

```text
WARP_MAT_M = 112
WARP_MAT_N = 128
M_TILES_PER_WARP = 7
N_TILES_PER_WARP = 8
```

Use the interleaved-M mapping:

```text
A row = warp_row * 16 + (wtid % 16) + m_tile * 32
B row = warp_col * WARP_MAT_N
        + (wtid % 16) * N_TILES_PER_WARP + n_tile
K words = batch_id * 8 + (wtid // 16) * 2
```

Read one `(2, al.u32)` operand for each M/N tile and issue one
`mfma_16x16x16_bf16_f32` per batch. Pass A first and B second.

Allocate `data_a` as `(M_TILES_PER_WARP, 2), al.u32` and `data_b` as
`(N_TILES_PER_WARP, 2), al.u32`. After filling the two words, form the MFMA
operand exactly as:

```python
frag_a = al.view(data_a[m_tile], al.Tensor((1, 2, 1), al.u32))
frag_b = al.view(data_b[n_tile], al.Tensor((1, 2, 1), al.u32))
acc[m_tile, n_tile] = al.amdgpu.mfma_16x16x16_bf16_f32(
    frag_a[0], frag_b[0], acc[m_tile, n_tile]
)
```

The accumulator shape is `(7, 8, 4)`, with output mapping:

```text
row = group_m * 224 + warp_row * 16
      + m_tile * 32 + (wtid // 16) * 4 + acc_idx
col = group_n * 256 + warp_col * 128
      + (wtid % 16) * 8 + n_tile
```

Pack neighboring output columns, never rows. Guard the partial final M tile and
write directly from registers without LDS staging or `al.shuffle`.

## Execution and scope

Keep the batch4 K loop unpipelined:

1. `_load_global_ab`;
2. `_store_shm_ab`;
3. synchronize;
4. call `_read_shm_ab` and `_matmul` for batches 0 through 3;
5. synchronize before the next K tile.

Do not modify batch2. Do not add raw-buffer access, prefetching, pipeline
unrolling, LDS padding, scheduling hints, stagger, or non-row-major workgroup
mapping.

## Verification

Directly validate the new batch4 kernel on 4096 during development. It is fine
to route 4096 to batch4 temporarily for that test, but restore the public
batch2 fallback before finishing. Then run all five sizes with the command in
`AGENTS.md` and compare against `input_model.py`.

Finish only when every final record has `"correctness":true`, public
performance has not materially regressed, and:

- the module contains the instantiated batch4 kernel, while final public large
  sizes still use the batch2 fallback;
- batch4 has one concise helper set;
- every K tile contributes four K16 batches exactly once;
- the interleaved-M mapping and partial-M guards are correct;
- A remains the first MFMA operand;
- batch2 code and performance have not materially changed;
- the trace contains evidence that 4096 batch4 correctness was tested before
  restoring the fallback dispatch.
