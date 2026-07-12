# Optimization 09: stagger batch2 K-tile order

Optimize `input_model.py` by changing only the order in which batch2 output
workgroups visit K64 tiles. The reduction still visits every K tile exactly
once, so the mathematical result and every per-tile operation remain the same.

Do not change batch4. Preserve WGM mapping, global-load ownership and widths,
LDS layouts, MFMA mapping, the software pipeline, scheduler hints, writeback,
and shape dispatch.

## One batch2 K-index helper

Add `STAGGER_MASK` and `STAGGER_STRIDE` as ordinary parameters of the existing
`_make_batch2_kernel` factory. Both selected batch2 specializations use:

```text
STAGGER_MASK = 31
STAGGER_STRIDE = 2
```

Inside that factory define exactly one `@avelang.jit` helper:

```text
_k_tile(group_n: al.u32, k_total: al.u32, offset: al.u32) -> al.u32
```

Its exact behavior is:

```text
k_start = (group_n & STAGGER_MASK) * STAGGER_STRIDE
k_start = al.convert(0, al.u32) if k_start >= k_total else k_start
return (k_start + offset) % k_total
```

Stagger by `group_n`, not `group_m`. The reset to zero is required when the
chosen start lies outside the current reduction. Do not duplicate the helper
for the two batch2 tile sizes.

## Apply it to every batch2 global load

The existing `_load_global_ab` takes a BF16 K-element offset, whereas `_k_tile`
returns a K64 tile index. Therefore every batch2 global-load call must pass:

```text
_k_tile(group_n, k_total, logical_offset) * GROUP_K
```

Replace all four logical forms used by the pipeline:

- prologue tile `0`;
- prologue tile `1`;
- steady-state tile `k_idx + 2`;
- steady-state tile `k_idx + 3`.

Do not change loop bounds or reorder any load, LDS operation, barrier, MFMA, or
scheduler call. The epilogue needs no new load; it consumes the final values
already fetched by the unchanged pipeline.

## Verification

Benchmark 1024 and 2048 first, then all five sizes. All correctness results
must be true, and batch4 performance should remain unchanged. Finish only when
the diff consists of the two factory parameters, one batch2 `_k_tile` helper,
the two batch2 instantiation arguments, and replacement of the four K-offset
expressions. Do not add load-offset modes or store-width variants in this round.
