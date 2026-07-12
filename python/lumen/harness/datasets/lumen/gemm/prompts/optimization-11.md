# Optimization 11: specialize stores and align paired operand order

Apply two final code-generation tunings to `input_model.py`:

1. add the shape-specific batch4 writeback width;
2. issue paired B operations before their corresponding A operations.

Preserve every mapping, tensor layout, MFMA operand, pipeline step, scheduling
hint, and dispatch decision except for these two changes.

## Factory mode

Add `STORE_VEC` as an ordinary parameter of the existing batch4 factory:

- 4096 row-major/base-offset specialization: `STORE_VEC=4`;
- 8192 mapping8/base-offset specialization: `STORE_VEC=4`;
- 16384 mapping8/default-load specialization: `STORE_VEC=2`.

`STORE_VEC` counts packed `al.u32` words per raw-buffer store. Keep one
batch4 factory and exactly one `_write_results` helper; branch inside that
helper without duplicating it.

## Writeback

Each lane owns `N_TILES_PER_WARP=8` adjacent BF16 columns. A pair of adjacent
accumulator N tiles is already packed into one `al.u32` with:

```text
perm(hi, lo, 0x07060302)
```

For `STORE_VEC == 4`, retain the current behavior exactly: pack all four words
and issue one `raw_buffer_store_x4` at the lane's existing `col`.

For `STORE_VEC == 2`, issue two `raw_buffer_store_x2` operations. Iterate:

```text
for n_tile_base in al.range(0, N_TILES_PER_WARP, 4):  # 0, then 4
    packed2[0] = pack accumulator tiles n_tile_base + 0 and +1
    packed2[1] = pack accumulator tiles n_tile_base + 2 and +3
    store_col = col + n_tile_base
    vindex = (row * n + store_col) * BF16_BYTES
    raw_buffer_store_x2(packed2, c_rsrc, vindex, block_base_bytes, 0)
```

At the top of `_write_results`, before its loops and compile-time branch,
unconditionally allocate the existing `(4,)` `packed_words` container and one
new `(2,)` `al.u32` `packed2` container. Do not declare a local tensor inside
the `STORE_VEC` branch; that form can crash the AveLang compiler. Do not create
separate helpers or unroll the two stores into repeated source blocks. If
retaining bounds guards, the x4 guard covers `global_col + 7`, while each x2
guard covers `global_col + n_tile_base + 3`. The selected benchmark shapes
have full N tiles, so the guards must not alter their ownership.

Do not replace the x2 path with two x1 stores, and do not change the mapping
from accumulator indices to neighboring output columns.

## Paired operand instruction order

In both the batch2 and batch4 factories, find every paired operation of these
three kinds:

- global loads of A and B;
- register-to-LDS stores of A and B;
- LDS-to-register reads of A and B.

Keep the source structure already present. If a pair is centralized in an
`*_ab` wrapper, reverse the two leaf-helper calls once inside that wrapper. If
the input issues adjacent A/B calls directly, reverse each existing direct
pair at its current call site. Do not add or remove wrappers.

Every pair must issue B first and A second while retaining the original
arguments and ownership:

```text
_load_global_b(... B arguments ...)
_load_global_a(... A arguments ...)

_store_shm_b(... B arguments ...)
_store_shm_a(... A arguments ...)

_read_shm_b(... B arguments ...)
_read_shm_a(... A arguments ...)
```

Reverse only the paired helper calls. Do not move a pair relative to another
pipeline operation, barrier, MFMA, or scheduler hint. In particular, MFMA
must still receive A as its first operand and B as its second operand. Do not
rename, duplicate, inline, or split helpers.

## Verification

Run all five correctness/performance cases and compare with `input_model.py`.
1024 through 8192 should compile to their unchanged stores; only 16384 selects
x2. Finish only when:

- every correctness record is true and performance has not materially
  regressed;
- the batch4 factory has one writeback implementation and the three expected
  `STORE_VEC` specialization arguments;
- every paired global load, LDS store, and LDS read issues B before A;
- no mapping, pipeline, scheduler, MFMA, or dispatch operation changed.
