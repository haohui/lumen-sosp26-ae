# Optimization 03: pipeline the second MoE GEMM stage

Optimize the existing AveLang fused FP8 MoE kernel by software-pipelining only
Stage 2, the W2 projection and BF16 output path. The input already contains the
two-buffer Stage 1 activation pipeline; preserve it exactly.

The current Stage 2 processes one 128-column output tile at a time: load W2,
run MFMA, convert and route-weight the result, transpose it through one LDS
buffer, then read and atomically write it before proceeding. Replace this with
a two-stage ping-pong pipeline that overlaps the next W2 load and the current
MFMA with LDS transpose/readback of the previous result.

## Required transformation

- Allocate two complete Stage 2 result buffers in LDS. Keep the existing
  `RET_DWORDS` layout for each buffer and increase total shared memory to hold
  both. Construct `shm_ret0` and `shm_ret1` as non-overlapping subviews.
- Separate the existing LDS transpose/writeback operation into three helpers:
  1. `_stage2_write_shm` writes an `(8, 2)` packed-u32 result fragment into the
     existing transposed LDS layout;
  2. `_stage2_read_shm` reads that layout back into an `(8, 2)` packed fragment
     using the consumer lane mapping;
  3. `_stage2_write_back` consumes the packed fragment and performs the same
     bounds-checked BF16 atomic output writes as the input implementation.
- Change `_stage2` to accept both LDS result buffers. Maintain two W2 register
  tile pairs (`curr` and `next`) and two W2 scales.
- Prologue: load the first W2 tile and scale into the current registers, advance
  offsets, and initialize the second LDS result buffer with zero fragments so
  the first pipeline read has a valid predecessor.
- Process two 128-column output tiles per outer iteration. For each tile:
  - prefetch the following W2 tile/scale into the register set not currently
    consumed, when another tile exists;
  - read the preceding packed result from the opposite LDS buffer;
  - run the existing two MFMA helpers with the current W2 registers;
  - preserve route-weight multiplication and BF16 rounding;
  - write the newly packed result to the current LDS buffer;
  - after the first tile, write the preceding fragment back to output;
  - synchronize before either LDS result buffer is reused.
- Alternate current/next W2 registers and `shm_ret0`/`shm_ret1` without copying
  whole fragments between them. Advance W2 value and scale offsets
  incrementally.
- Epilogue: after the loop, synchronize, select the LDS buffer containing the
  last result from the output-tile parity, read it, and write the final
  128-column tile back exactly once.

The generated IR must expose overlap among W2 prefetch, MFMA, and the delayed
LDS/output path. Merely duplicating buffers around the old serialized loop is
not sufficient.

## Correctness invariants

- Preserve routed fused-MoE semantics, FP8 block scaling, Stage 1 output and
  quantization, route weights, BF16 rounding, and atomic accumulation.
- Preserve the public `fused_moe_fp8_blockscale_g1u1` API, route-group mapping,
  workgroup size, and MFMA fragment mapping.
- Every W2 output tile must be accumulated exactly once. Keep all LDS and W2
  accesses in bounds, including the pipeline prologue and epilogue.
- Preserve the Stage 1 ping-pong pipeline already present in the input.

## Scope boundary for this round

This round is only the Stage 2 W2/result pipeline. Do not introduce later
optimizations:

- no `v_setvskip` invalid-route masking;
- no packed/vectorized SiLU, FP8 quantization, or route-weight arithmetic
  changes;
- no persistent thread-group scheduling or `num_persistent_tgs` argument;
- no explicit instruction-scheduling barriers.

The final implementation must pass correctness for token counts 1024, 2048,
4096, 8192, and 16384. Benchmark all five sizes and retain the optimized
implementation only if it does not materially regress the input kernel; do not
add fallback compute paths.
