# Optimization 04: mask invalid routes with v_setvskip

Optimize the existing AveLang fused FP8 MoE kernel by replacing the divergent
per-token bounds branch in the Stage 2 output hot path with an AMD
`v_setvskip` mask. The input already contains the Stage 1 and Stage 2
ping-pong pipelines; preserve both.

Padded route groups contain token slots whose decoded token id is greater than
or equal to `num_tokens`. Those slots must not perform output atomics. The
current `_stage2_write_back` checks `tokens[i] < num_tokens` around every pair
of atomic adds. Move that validity decision out of the repeatedly executed
output-tile path.

## Required transformation

- After the kernel decodes its eight local token ids, construct one `u32`
  `invalid_token_mask`. Bit `i` must be one exactly when `tokens[i] >=
  num_tokens`.
- Thread this mask through `_stage2` into every `_stage2_write_back` call,
  including both delayed pipeline writes and the final epilogue write.
- In `_stage2_write_back`, remove the per-token `if tokens[i] < num_tokens`
  branch. For each of the eight token slots:
  - compute the existing output byte offset and unpack the same two packed BF16
    pairs;
  - call `S.amdgpu.v_setvskip(invalid_token_mask, i)` immediately before the
    two atomic adds so an invalid slot skips them;
  - perform the same two BF16 atomic adds at offsets separated by 256 bytes;
  - immediately restore execution with `S.amdgpu.v_setvskip(0, 0)` before the
    next slot.
- Keep the mask bit ordering identical to the `tokens[0:8]` loop. Do not use a
  wave ballot or change route decoding.

The optimization is complete only if the output hot loop has no token-validity
control-flow branch and the two atomics are guarded by `v_setvskip`, while the
skip state is reset after each slot.

## Correctness invariants

- Invalid padded routes must perform no global output writes. Valid routes must
  produce exactly the same two atomic additions as before.
- Preserve routed fused-MoE semantics, FP8 scaling and quantization, route
  weights, BF16 rounding, and atomic accumulation.
- Preserve the public API, route-group mapping, workgroup size, Stage 1
  pipeline, Stage 2 ping-pong buffers, MFMA operations, and pipeline barriers.

## Scope boundary for this round

This round is only invalid-route masking with `v_setvskip`. Do not introduce
later optimizations:

- no packed/vectorized SiLU, FP8 quantization, or route-weight arithmetic
  changes;
- no persistent thread-group scheduling or `num_persistent_tgs` argument;
- no explicit instruction-scheduling barriers.

The final implementation must pass correctness for token counts 1024, 2048,
4096, 8192, and 16384. Benchmark all five sizes and retain the optimized
implementation only if it does not materially regress the input kernel; do not
add fallback compute paths.
