# Optimization 10: specialize raw-buffer load offsets

Optimize `input_model.py` by selecting where each raw-buffer global-load base
offset is carried. This round changes only the split between the `vindex` and
`soffset` arguments of existing raw-buffer loads; their sum, accessed address,
load width, and ownership must remain identical.

Preserve all WGM and K-stagger mappings, LDS layouts, MFMA work, pipelines,
scheduler hints, writeback, and grid calculation.

## Load modes

Define module-level constants:

```text
LOAD_MODE_DEFAULT = 0
LOAD_MODE_BASE_OFFSET = 1
```

Add `LOAD_MODE` as an ordinary parameter to both existing kernel factories.
Keep exactly one `_load_global_a` and one `_load_global_b` helper per factory;
do not redefine or duplicate helpers for the two modes.

Each helper already computes:

```text
tile_base_bytes = (output_tile_row_base * k + k_offset) * BF16_BYTES
thread_bytes = per-thread offset within that K tile
```

Immediately before each existing `raw_buffer_load_x4` or
`raw_buffer_load_x1`, derive fresh call arguments for that loop iteration:

```text
load_vindex = thread_bytes
load_soffset = tile_base_bytes
if LOAD_MODE == LOAD_MODE_BASE_OFFSET:
    load_vindex = load_vindex + load_soffset
    load_soffset = al.convert(0, the_existing_integer_type)
```

Pass `load_vindex` as `vindex` and `load_soffset` as `soffset`. Do not mutate
`tile_base_bytes` itself inside the load loop: every `load_idx` needs the same
tile base. In default mode, these aliases reproduce the current arguments.
Do not move an offset into the resource descriptor and do not alter the load
loop, its stride, or register layout.

## Shape specializations

Use these modes:

- 1024 batch2 64x64: `LOAD_MODE_DEFAULT`;
- 2048 batch2 128x128: `LOAD_MODE_BASE_OFFSET`;
- 4096 batch4 row-major: `LOAD_MODE_BASE_OFFSET`;
- 8192 batch4 mapping8: `LOAD_MODE_BASE_OFFSET`;
- 16384 batch4 mapping8: `LOAD_MODE_DEFAULT`.

The existing batch4 mapping8 specialization can no longer be shared by 8192
and 16384. Instantiate the same batch4 factory twice with different
`LOAD_MODE` values and dispatch the two shapes separately. Do not duplicate
the factory or any helper bodies.

## Verification

Run all five correctness/performance cases and compare with `input_model.py`.
The only kernel-body changes should be the mode branch in the four existing A/B
load helpers. The remaining diff should contain factory parameters,
specializations, and dispatch selection. Do not add store-width variants or
change any raw-buffer instruction width in this round.
