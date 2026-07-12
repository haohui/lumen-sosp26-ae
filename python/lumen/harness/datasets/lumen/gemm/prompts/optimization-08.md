# Optimization 08: add shape-specific WGM mapping

Optimize `input_model.py` by changing only the mapping from `al.block_id(0)`
to the output-tile coordinates `(group_m, group_n)`. Preserve all per-tile
loads, LDS layouts, MFMA operations, pipelines, scheduling hints, and
writeback exactly.

## One shared mapping helper

Define these module-level constants:

```text
WGM_ROW_MAJOR = 0
WGM_XCC = 1
WGM_XCC_MAPPING8 = 2
MI300_CU_COUNT = 38 * 8
WGM_XCC_WIDTH = 8
```

Define exactly one module-level `@avelang.jit` helper named `_wgm_mapping`.
Do not duplicate it inside either kernel factory. Its inputs are `m`, `n`,
`group_size_m`, `group_size_n`, `wgm_mode`, and `ceil_groups`, all represented
as `al.u32`; it returns `(group_m, group_n)` as two `al.u32` values.

Implement the following algorithm exactly. The arithmetic below is kernel
arithmetic, not host-side Python arithmetic:

```text
linear_group_id = al.block_id(0)
m_groups = m // group_size_m
n_groups = n // group_size_n
if ceil_groups != 0:
    m_groups = (m + group_size_m - 1) // group_size_m
    n_groups = (n + group_size_n - 1) // group_size_n

if wgm_mode != WGM_ROW_MAJOR:
    total_groups = m_groups * n_groups
    cu_count = al.convert(MI300_CU_COUNT, al.u32)
    wgm_xcc = al.convert(WGM_XCC_WIDTH, al.u32)
    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc
    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (total_groups % cu_count
                 if linear_group_id >= cu_tail_limit
                 else cu_count)
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    mapped = cu_base + cu_xcc_stride
    linear_group_id = (mapped
                       if linear_group_id < linear_group_limit
                       else linear_group_id)

group_m = linear_group_id // n_groups
group_n = linear_group_id - group_m * n_groups

if wgm_mode == WGM_XCC_MAPPING8:
    workgroup_mapping = al.convert(8, al.u32)
    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = (workgroup_mapping if mapping_tail == 0 else mapping_tail)
    mapping_span = (mapping_tail
                    if mapping_block >= mapping_groups
                    else workgroup_mapping)
    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

return group_m, group_n
```

This is only a permutation of valid output tiles. Do not add bounds checks or
change grid-size calculation.

## Factory parameters and dispatch

Add `WGM_MODE` as an ordinary Python parameter to both existing factories.
Each kernel calls the shared helper once, where it currently converts
`block_id(0)` directly to row-major coordinates:

- batch2 passes `ceil_groups=0` because both M and N are exact multiples;
- batch4 passes `ceil_groups=1` because its 224-row tile uses ceil-div M.

Instantiate these specializations from the existing factory bodies:

- 1024 batch2 64x64: `WGM_XCC`;
- 2048 batch2 128x128: `WGM_XCC`;
- 4096 batch4 224x256: `WGM_ROW_MAJOR`;
- 8192 and 16384 batch4 224x256: `WGM_XCC_MAPPING8`.

The 8192 and 16384 dispatch entries may share the same mapping8 specialization.
Do not copy the batch4 factory or any helper bodies to create the two batch4
specializations.

## Verification

Run correctness and performance for all five sizes and compare with
`input_model.py`. Finish only when the diff contains the shared mapping helper,
factory parameters, specializations, dispatch selection, and replacement of
the two old row-major mapping sites. Do not add K stagger or change global-load
offset mode, store vector width, scheduling sequences, or any pipeline code.
