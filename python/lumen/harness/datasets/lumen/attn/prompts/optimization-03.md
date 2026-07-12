# Optimization 03: direct global-to-LDS Q/K loads and mirrored query tiles

Optimize the existing AveLang FlashAttention kernel in two narrowly scoped
ways: use AMD direct global-to-LDS buffer loads for Q and K staging, and let
each physical workgroup process a pair of logical query tiles mirrored around
the sequence midpoint. The input already contains the transposed/repacked V
LDS layout from the previous round; preserve it.

## Required transformation: asynchronous Q/K staging

- In the Q staging helper, replace the scalar global buffer load followed by a
  normal LDS tensor store with `al.amdgpu.raw_buffer_load_x1_lds`.
- Keep the current global vector/soffset calculation and the current swizzled
  destination row and column. Convert the destination LDS word index to its
  byte offset and issue a four-byte direct-to-LDS load.
- Apply the same transformation to each K-page staging load. Q and K must keep
  their existing LDS layouts and register-consumption layouts.
- Preserve the Q load wait and synchronization sequence: wait for the
  direct-to-LDS operations before the workgroup reads Q, and retain the
  barriers that protect reuse of the shared allocation. Preserve the existing
  K load/consume synchronization performed by the attention loop.
- Do not convert the V load path in this round; V must retain the global load
  and transposed LDS store produced by the previous optimization.

## Required transformation: mirrored query-tile execution

- Let one physical workgroup process the logical tile at `head_idx_q` and its
  mirror `num_q_tiles - 1 - head_idx_q`. Process only one pass for the center
  tile when the two indices are equal; otherwise process two passes by calling
  the existing per-tile helper sequentially.
- Halve the physical query-tile grid with ceiling division in both the public
  launcher and the device-side bounds check.
- Redistribute the physical `(query head, tile)` grid before deriving the KV
  head so all eight query heads still cover every logical tile. Treat the
  physical head/tile pair as a merged linear index over the physical tiles,
  map its low three bits back to the query head within each group of eight,
  and use the remaining high bits as the first logical query tile.
- Preserve the existing `reverse_pass` behavior: the first tile uses the
  forward pass and the mirrored tile uses the reverse pass. Do not duplicate
  the attention body; reuse the existing per-tile processing helper and shared
  allocation.

## Correctness invariants

- Every `(sequence, query head, logical query tile)` must execute exactly once,
  including odd tile counts and the midpoint tile.
- Preserve causal grouped-query attention semantics, output locations, and the
  mapping from query heads to KV heads.
- Preserve the packed Q/K/V interface, `seq_ptr`, `max_seq_len`, BF16 output,
  FP32 softmax state, and FP32 output accumulation.
- Preserve tile sizes, workgroup dimensions, MFMA instructions, online-softmax
  math, and the transposed V LDS representation from the input.

## Scope boundary for this round

This round contains only direct global-to-LDS Q/K loads and mirrored query-tile
execution. Do not introduce optimizations from later rounds:

- no new LDS padding, XOR swizzle, or bank-conflict-specific layout;
- no double-buffered K/V software pipeline;
- no producer/consumer wave specialization;
- no hand-written instruction scheduling or scheduler-group barriers;
- no changes to softmax rescaling thresholds or arithmetic shortcuts.

The final implementation must pass correctness for sequence lengths 1024,
2048, 4096, 8192, and 16384 and should match the performance level of the
repository's third attention optimization stage without fallback compute paths.
