# Optimization 07: schedule the hot loops

Add gfx942 instruction-scheduling hints to the existing pipelined GEMM in
`input_model.py`. Preserve every semantic operation and all existing code
structure, layouts, buffers, mappings, and dispatch.

## Scheduling helper

In each kernel factory, add exactly one `@avelang.jit`
`_hot_loop_scheduler()` helper. It must contain only
`al.amdgpu.sched_group_barrier(mask, count, 0)` calls; it must not load, store,
synchronize, compute, or mutate data.

Use:

```text
SCHED_MASK_MFMA        = 0x008
SCHED_MASK_BUFFER_LOAD = 0x020
SCHED_MASK_DS_READ     = 0x100
SCHED_MASK_DS_WRITE    = 0x200
```

Capture `LOOP_SCHEDULER` as an ordinary Python factory parameter:

- batch2 64x64: mode 1;
- batch2 128x128: mode 0;
- batch4 224x256: mode 1.

Do not redefine the helper in a configuration branch. Define it once and use
one compile-time `if LOOP_SCHEDULER == 0` inside it.

## Mode 0 sequence

Emit this exact sequence:

```python
al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
for _ in al.range(8):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
al.amdgpu.sched_group_barrier(0x0800, 1, 0)
al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)

for _ in al.range(2):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

for _ in al.range(3):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

al.amdgpu.sched_group_barrier(0x0800, 1, 0)
for _ in al.range(8):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
```

## Mode 1 sequence

Emit this exact sequence:

```python
for _ in al.range(45):
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 8, 0)
al.amdgpu.sched_group_barrier(0x0800, 1, 0)
al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)

for _ in al.range(30):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

al.amdgpu.sched_group_barrier(0x0800, 1, 0)
for _ in al.range(15):
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 5, 0)
```

## Call sites

- batch2: call `_hot_loop_scheduler()` twice at the very end of every
  two-K-tile steady-state loop iteration;
- batch4: call it once at the very end of every steady-state loop iteration.

Do not call it in prologues or epilogues. Do not move any existing read, MFMA,
barrier, store, or load relative to another semantic operation.

## Verification

Benchmark 1024/2048 and 4096 first, then all five sizes. Compare each shape
with `input_model.py`; correctness must remain true and no size may materially
regress.

Finish only when the diff contains factory parameters, scheduler constants,
one semantic-free helper per family, and the required call sites—nothing else.
Do not add WGM mapping, K stagger, load-mode changes, or store-width changes.
