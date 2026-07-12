# Optimization 06: explicit hot-loop instruction scheduling

Optimize the existing AveLang FlashAttention kernel by adding the final AMD
instruction-scheduling directives around the QK and P@V MFMA groups and by
applying the final online-softmax rescaling shortcut consistently. Preserve the
entire padded, pipelined stage-05 implementation; this round should be a small,
targeted change.

## Required transformation: scheduler groups

The input already defines scheduler masks, instruction-count constants,
`_mfma_per_issue`, and the three hot-loop scheduler helpers. Populate those
helpers as follows, keeping the final `al.amdgpu.sched_barrier(0)` in each one:

- In `_hot_loop_scheduler_qk_major`, repeat `SCHED_INST_ALU_LIGHT` times:
  issue `sched_group_barrier(SCHED_MASK_VALU, 1, 0)`, then
  `sched_group_barrier(SCHED_MASK_MFMA, QK_MAJOR_MFMA_PER_ISSUE, 0)`.
- Do the same in `_hot_loop_scheduler_qk_minor`, using
  `QK_MINOR_MFMA_PER_ISSUE` for the MFMA quota.
- In `_hot_loop_scheduler_gemm_o`, first repeat `SCHED_INST_ALU_MEDIUM` times
  with a one-instruction VALU group followed by an MFMA group of
  `GEMM_O_MFMA_PER_ISSUE`. Then repeat `SCHED_INST_TRANS_HEAVY` times with a
  one-instruction transcendental group (`SCHED_MASK_TRANS`) followed by the
  same MFMA group.

Do not move the existing calls to these helpers. They already mark the intended
boundaries after the major/minor QK MFMA groups and the P@V MFMA group.

## Required transformation: softmax rescaling shortcut

Use a threshold of `40.0` in both normal output-update helpers and both final
prepare/drain softmax helpers.

For every partition after the first:

- Compute the existing delta `(block_max - mi) * scale_log2`.
- If the delta is at most 40, keep the previous maximum (`mi_new = mi`) and,
  after computing probabilities with that maximum, update the denominator as
  `l = l + row_sum`. Do not evaluate `exp2` and do not rescale `out_acc` in this
  path.
- Otherwise preserve the existing max selection and exponential-rescaling
  path: compute `exp2((mi - mi_new) * scale_log2)`, update `l` with the scaled
  previous sum, and rescale `out_acc` before the P@V accumulation.
- Preserve the all-masked/`NEG_INF` behavior and the first-partition behavior.

Do not carry a `keep_previous` flag between max selection and denominator
update. Match the intended VALU shape by testing the delta directly in both
places: once while selecting `mi_new`, and again while choosing `l += row_sum`
versus exponential rescaling. Remove any such flag already present in the input
prepare helpers, change their threshold from 8 to 40, and add the same direct
branch structure to the steady-state batch-0 and batch-1 update helpers.

## Correctness and scope

- Preserve causal grouped-query attention, traversal order, online-softmax
  state, BF16/FP32 types, LDS layouts, K/V pipeline, barriers, MFMA operations,
  mirrored query tiles, grid, workgroup size, and public API.
- Do not add or remove memory-pipeline barriers, change LDS allocation, alter
  tile sizes, introduce fallback compute, or perform unrelated refactoring.
- Do not add scheduler directives outside the three existing scheduler helpers.

The final implementation must pass correctness for sequence lengths 1024,
2048, 4096, 8192, and 16384. On an otherwise idle MI300X with repository
benchmark defaults, the stage-06 target is approximately 0.164, 0.545, 1.91,
6.62, and 26.8 ms. Treat results more than 3% slower than these targets as
unfinished: inspect whether every scheduler group and every threshold shortcut
was applied exactly, revise, and rerun correctness and performance. Keep the
specified MFMA quotas unchanged; multiplying them or tuning them away from the
derived `*_MFMA_PER_ISSUE` values does not implement this stage.
