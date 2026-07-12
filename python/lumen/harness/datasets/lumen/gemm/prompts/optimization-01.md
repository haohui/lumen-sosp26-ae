# Optimization 01: batch2 MFMA baseline

Optimize `input_model.py` by replacing its scalar dot-product loop with a
batch2 AMD MFMA kernel. This round handles only the batch2 family and establishes
the reusable helper structure needed by later rounds.

Keep the public API:

```python
gemm_pipeline_transposed_b(A, B, out=None) -> torch.Tensor
```

It computes `C = A @ B.T` for row-major BF16 A `(m, k)` and B `(n, k)`, with
FP32 MFMA accumulation and BF16 output.

## Configurations

Create one `_make_batch2_kernel(...)` factory and instantiate it twice:

- 1024: `GROUP_M=64`, `GROUP_N=64`, `GROUP_K=64`,
  using the batch2 family;
- 2048: `GROUP_M=128`, `GROUP_N=128`, `GROUP_K=64`, using the batch2 family.

Use four warps arranged as `WARP_PER_ROW=2`, `WARP_PER_COL=2`. For the other
selected benchmark sizes, use the 128x128 batch2 kernel as the temporary
fallback so all five sizes remain correct. Batch4 is introduced in the next
round.

Define these fixed constants once at module scope and reuse them in the
factories and at launch:

```text
WARP_SIZE = 64
NUM_WARPS = 4
NUM_THREADS = WARP_SIZE * NUM_WARPS = 256
VEC_SIZE = 8          # BF16 values per cooperative load fragment
BF16_BYTES = 2
```

Launch every generated kernel with block size `NUM_THREADS`. Inside a kernel,
`tid = al.thread_id(0)` is therefore in `[0, NUM_THREADS)`. Derive the wave and
lane IDs with:

```python
wid = al.amdgpu.readfirstlane(tid // WARP_SIZE)
wtid = tid % WARP_SIZE
```

## One concise helper set

Inside `_make_batch2_kernel`, compute all tile-dependent sizes as ordinary
Python integers. Define exactly one implementation of each helper:

- `_load_global_a`, `_load_global_b`, and `_load_global_ab`;
- `_store_shm_a`, `_store_shm_b`, and `_store_shm_ab`;
- `_read_shm_a`, `_read_shm_b`, and `_read_shm_ab`;
- `_matmul`;
- `_write_results`;
- `kernel`.

Use the captured Python integers directly in `al.Tensor` annotations, layouts,
and loop bounds. Calling the factory twice provides shape specialization.

Do not:

- branch on `LOADS_A`, tile size, or another captured value to redefine
  helpers;
- define generic `_impl` helpers and then shadow them;
- duplicate helper bodies for the 64x64 and 128x128 configurations;
- inline load, LDS, MFMA, or writeback bodies into `kernel`;
- convert factory-local shape integers to `al.constexpr` objects merely for
  annotations.

Instantiate the two returned kernels once at module scope and select between
them in `gemm_pipeline_transposed_b`.

## Cooperative load and contiguous LDS layout

Use one 8-BF16 fragment per load. Adjacent threads own adjacent fragments:

```text
vector_idx = load_idx * NUM_THREADS + tid
elem_base = vector_idx * VEC_SIZE
tile_row = elem_base // GROUP_K
tile_col = elem_base % GROUP_K
```

`elem_base` is a BF16-element offset within the current logical A or B K tile,
not a byte offset. For a fixed `load_idx`, threads 0 through 255 own 256
consecutive 8-BF16 fragments. The next `load_idx` continues immediately after
those fragments. Consequently:

```text
LOADS_PER_THREAD_A = GROUP_M * GROUP_K // (NUM_THREADS * VEC_SIZE)
LOADS_PER_THREAD_B = GROUP_N * GROUP_K // (NUM_THREADS * VEC_SIZE)
```

This is wave-coalesced ownership. Do not replace it with
`(tid * LOADS_PER_THREAD + load_idx) * VEC_SIZE`, which makes adjacent lanes
access strided fragments.

Load the eight BF16 values normally in this round; raw-buffer operations are a
later optimization. The same `(tile_row, tile_col)` must be used when storing
the fragment to LDS.

Use this exact register and LDS representation in this round:

```python
reg_a = al.make_local((LOADS_PER_THREAD_A, 4), al.u32)
reg_b = al.make_local((LOADS_PER_THREAD_B, 4), al.u32)
shm_a = al.make_shared((SHM_A_VECTORS, 4), al.u32)
shm_b = al.make_shared((SHM_B_VECTORS, 4), al.u32)
```

Inside each leaf global-load helper, view its `u32` register tensor as BF16
with the explicit row-major layout `(LOADS_PER_THREAD, VEC_SIZE)` and strides
`(VEC_SIZE, 1)`, then fill that BF16 view element by element. Keep the public
helper argument and the allocation itself as `(LOADS_PER_THREAD, 4), al.u32`;
do not change them to a BF16 tensor after encountering a helper-boundary type
error.

Store each `(4, al.u32)` fragment in a contiguous row-major LDS layout. One
vector contains 8 BF16 values, so every K64 row contains 8 vectors:

```text
physical_vector = row * 8 + vector_in_row
```

Use separate contiguous LDS allocations and the same formula for A and B. Do
not add padding in this round. A later optimization will change only the
physical LDS spacing while preserving the logical ownership established here.

### Required wrapper pattern

AveLang currently needs an explicit same-shaped view when one nested JIT helper
passes a tensor to another nested JIT helper. In each `*_ab` wrapper, create a
view of every tensor argument before delegating to the A and B leaf helpers.
For example, `_store_shm_ab` must form views equivalent to:

```python
shm_a_view = al.view(
    shm_a, al.u32,
    al.make_layout((SHM_A_VECTORS, 4), (4, 1)),
)
reg_a_view = al.view(
    reg_a, al.u32,
    al.make_layout((LOADS_PER_THREAD_A, 4), (4, 1)),
)
```

and pass those views to `_store_shm_a`; do the analogous operation for B.
Apply the same adapter pattern in `_load_global_ab` for `reg_a/reg_b`, and in
`_read_shm_ab` for `shm_a/shm_b/data_a/data_b`. The views do not change data or
layout; they make the helper boundary explicit to the compiler.

The kernel must call `_load_global_ab`, `_store_shm_ab`, and `_read_shm_ab`.
Do not work around a lowering error by leaving an unused `*_ab` wrapper and
calling its A/B leaf helpers directly from `kernel`.

## Batch2 MFMA mapping

Each 64-wide K tile contains two K32 batches. Define:

```text
WARP_MAT_M = GROUP_M // 2
WARP_MAT_N = GROUP_N // 2
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16
```

### MFMA lane semantics

Use only:

```python
al.amdgpu.mfma_16x16x16_bf16_f32(data_a, data_b, acc)
```

For one logical 16x16x16 MFMA tile, its native lane mapping is:

```text
A(i, kk): wtid = i + (kk // 4) * 16, element = kk % 4
B(kk, j): wtid = j + (kk // 4) * 16, element = kk % 4
```

For each lane and its four FP32 accumulator elements:

```text
C row within the MFMA tile = 4 * (wtid // 16) + acc_idx
C col within the MFMA tile = wtid % 16
```

The physical input B is stored as `(n, k)`, so its logical row is the output
column `j`. Load B with the same K ordering as A and pass A first, B second.
Never swap operands to simplify writeback; fix the writeback mapping instead.

### Exact operand-register representation

Allocate one reusable K32-batch operand set:

```python
data_a = al.make_local((M_TILES_PER_WARP, 4), al.u32)
data_b = al.make_local((N_TILES_PER_WARP, 4), al.u32)
acc = al.make_local(
    (M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32
)
```

For one M or N tile, `data_[tile]` contains exactly one contiguous 16-byte LDS
fragment: four `u32`, or eight BF16 values. Keep this storage flat. Split it
into the two natural K16 operands only inside `_matmul`.

`_read_shm_a/b` take `batch_id` and fill this reusable operand set. For each
batch, read one `(4, al.u32)` fragment per M/N tile with:

```text
A row = warp_row * WARP_MAT_M
        + (wtid % 16) * M_TILES_PER_WARP + m_tile
B row = warp_col * WARP_MAT_N
        + (wtid % 16) * N_TILES_PER_WARP + n_tile
K vector = batch_id * 4 + (wtid // 16)
```

Here a K vector contains 8 BF16 values. Therefore batch 0 addresses K columns
0-31 and batch 1 addresses K columns 32-63. Convert each logical row and
`K vector` through the contiguous LDS formula before reading.

The read is logically equivalent to:

```text
for m_tile:
    data_a[m_tile] = shm_a[A row * 8 + K vector]
for n_tile:
    data_b[n_tile] = shm_b[B row * 8 + K vector]
```

Do not read A using `N_TILES_PER_WARP` or B using `M_TILES_PER_WARP`.

### MFMA calls for one K32 batch

`_matmul` consumes the current `data_a/data_b`. For every `(m_tile, n_tile)`,
issue exactly two calls in natural fragment order:

```python
frag_a = al.view(data_a[m_tile], al.Tensor((2, 2, 1), al.u32))
frag_b = al.view(data_b[n_tile], al.Tensor((2, 2, 1), al.u32))

acc[m_tile, n_tile] = al.amdgpu.mfma_16x16x16_bf16_f32(
    frag_a[0], frag_b[0], acc[m_tile, n_tile]
)
acc[m_tile, n_tile] = al.amdgpu.mfma_16x16x16_bf16_f32(
    frag_a[1], frag_b[1], acc[m_tile, n_tile]
)
```

Together the two calls accumulate the selected K32 batch. Calling `_matmul`
once for batch 0 and once for batch 1 gives exactly four K16 MFMA steps per K64
tile. Do not add, omit, or reorder fragment halves.

### Accumulator to C mapping

The LDS row swizzle assigns consecutive MFMA `i` values to rows separated by
`M_TILES_PER_WARP`, and consecutive MFMA `j` values to columns separated by
`N_TILES_PER_WARP`. Combining that swizzle with the native accumulator mapping
above gives:

```text
row = group_m * GROUP_M + warp_row * WARP_MAT_M
      + (wtid // 16) * (4 * M_TILES_PER_WARP)
      + acc_idx * M_TILES_PER_WARP + m_tile
col = group_n * GROUP_N + warp_col * WARP_MAT_N
      + (wtid % 16) * N_TILES_PER_WARP + n_tile
```

Use this formula directly in `_write_results`. For fixed `m_tile`, `acc_idx`,
and lane, successive `n_tile` values are neighboring output columns. Successive
`acc_idx` values are not neighboring rows because of the M-tile swizzle.

Pack neighboring `n_tile` values into BF16 pairs using
`al.amdgpu.perm(hi, lo, 0x07060302)`. Never pack neighboring rows. Guard output
bounds and write directly from registers; do not stage C through LDS or use
`al.shuffle`.

For the in-bounds pair, view C as a flat `u32` tensor with layout
`((m * n) // 2,), (1,)` and store the packed word at
`(row * n + col) // 2`. Use a scalar BF16 store only for a final unpaired
column. Do not compute `packed` and then replace the pair with two independent
scalar conversions.

## Simple execution order

Keep the K loop unpipelined:

1. call `_load_global_ab`;
2. call `_store_shm_ab`;
3. synchronize;
4. for batch 0 and batch 1, call `_read_shm_ab` then `_matmul`;
5. synchronize before overwriting LDS.

After the loop, call `_write_results`. Do not add raw-buffer operations,
prefetching, loop unrolling, scheduling hints, stagger, or non-row-major
workgroup mapping.

## Verification

Run the benchmark command in `AGENTS.md`, first for 1024 and 2048 and then for
all five sizes. Finish only when every record has `"correctness":true` and:

- both batch2 configurations are exercised;
- the factory contains one non-duplicated helper set;
- the kernel composes the helpers;
- both K32 batches execute exactly once per K tile, with two K16 MFMA calls
  per batch;
- `data_a/data_b` use the exact flat `(tile, 4)` `u32` representation and are
  viewed as two MFMA operands only inside `_matmul`;
- LDS row and K-vector ownership match the formulas above;
- A is the first MFMA operand;
- neighboring output columns are packed together.
