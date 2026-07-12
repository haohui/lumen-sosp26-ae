# Optimization 12: expose affine memory access patterns

Refactor the existing global-load and LDS helper indexing in `input_model.py`
so the compiler can see lane-dependent base addresses plus compile-time affine
loop offsets. This is an address-generation optimization only: every accessed
logical and physical element must remain identical.

Preserve all factory parameters, shape specializations, resource ranges,
raw-buffer widths, global ownership, padded LDS layouts, MFMA mappings,
pipelines, scheduler hints, writeback, paired B-before-A order, and dispatch.

## General affine form

In each leaf global-load, register-to-LDS-store, and LDS-to-register-read
helper:

1. compute the lane-dependent address for loop iteration zero before the
   static loop;
2. compute the constant address stride between successive iterations once;
3. address iteration `i` as `base + i * stride`.

Prefer this form over recomputing row, column, row group, and the complete
address from `i` inside every loop iteration. Do not use a mutable running
index such as `offset += stride`; keep the loop-index relationship explicit so
gfx942 code generation can fold constant LDS offsets into instructions.

When a quotient is already available, form a remainder as
`value - quotient * divisor` instead of independently recomputing
`value % divisor`. Hoist other loop-invariant integer arithmetic out of static
loops as well. Do not replace an expression if doing so changes signedness,
overflow behavior, or the final address.

## Global-load helpers

Keep the existing raw-buffer resource, `LOAD_MODE`, `vindex/soffset` split,
load width, and wave-coalesced ownership.

For batch2, one loop step advances from the current 16-byte fragment to the
fragment owned by the same lane `NUM_THREADS * VEC_SIZE // GROUP_K` logical
rows later. Compute its byte stride once and express every
`raw_buffer_load_x4` address as a base plus `load_idx * stride`.

For batch4, one loop step advances `GLOBAL_ROWS_PER_ROUND` logical rows. Compute
the corresponding byte stride once and express every `raw_buffer_load_x1`
address as a base plus `load_idx * stride`.

The sum of `vindex` and `soffset` must remain exactly the same as in the input
for every load and every `LOAD_MODE` specialization.

## Register-to-LDS stores

Retain the padded physical layout introduced earlier. Derive the physical LDS
address for the first register fragment from the same global-load ownership,
then derive the constant padded-layout stride between register rows.

The selected layouts align each store loop's logical row step to its
`READ_ROWS` grouping, so successive stores can use:

```text
physical_base + store_idx * physical_stride
```

Keep batch2 stores vectorized as one 16-byte `(4, al.u32)` fragment. Keep
batch4 stores at their existing packed-word width. Do not scalarize a
vectorized LDS operation or change which thread stores a fragment.

## LDS-to-register reads

Keep the MFMA-oriented ownership:

- lane row ownership comes from `wtid % 16`;
- K-vector ownership comes from `wtid // 16`;
- `batch_id` selects the existing K region;
- the tile loop visits the same logical rows in the same order.

Compute `start_row`, the padded physical base, and the physical tile stride
before the tile loop. Then read tile `tile_idx` from
`physical_base + tile_idx * physical_stride`. A structured `al.view` with
compile-time layout strides may be used when it exposes the same relationship
more directly.

Keep each batch2 LDS read as one 16-byte `(4, al.u32)` fragment. Keep each
batch4 MFMA operand read as the same two adjacent `al.u32` words. Do not change
operand-register shapes or MFMA calls.

## Scope and verification

Change only the leaf global-load, LDS-store, and LDS-read helpers, plus local
compile-time constants needed by those helpers. Do not alter combined wrapper
structure or call order.

Run all five correctness/performance cases and compare with `input_model.py`.
Finish only when:

- every correctness record is true;
- no workload materially regresses;
- every refactored static memory loop visibly uses a base plus loop-index
  times stride;
- all raw-buffer and LDS operation widths are unchanged;
- the source diff contains no pipeline, scheduler, mapping, MFMA, writeback,
  or dispatch change.
