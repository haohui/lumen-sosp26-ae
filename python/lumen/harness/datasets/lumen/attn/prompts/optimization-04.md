# Optimization 04: padded LDS layouts and a paired K/V software pipeline

Optimize the existing AveLang FlashAttention kernel by eliminating the dominant
LDS bank conflicts and reorganizing the serial K-slice loop into a two-page
software pipeline. The input already has direct global-to-LDS Q/K loads, the
transposed V representation, and mirrored query-tile execution; preserve all
three.

## Required transformation: LDS padding

- Pad each Q stage and each K page by one 128-bit unit for every group of data
  distributed across a wave. Define the Q-stage and K-page allocation padding
  from their row counts, `HEAD_DIM`, `WARP_SIZE`, and `U128_BYTES / U32_BYTES`
  rather than using magic total sizes.
- Add one 128-bit (`U128_BYTES / U32_BYTES`) padding unit to the logical Q/K
  row stride used by register fetches and direct-to-LDS destinations. Recompute
  all dependent tile/page strides and shared-allocation sizes from that padded
  stride.
- Add one packed-`u32` padding position per V row-pair group in each parity
  half. Keep the even/odd transposed V representation, but make every V store
  and register-fetch view use the padded half sizes and strides consistently.
- Audit every LDS subview, layout, direct-to-LDS byte offset, V parity base, and
  shared-memory size. The padding must remove bank aliasing without changing
  the register fragments seen by QK or P@V MFMA instructions.

The intended symbolic values are equivalent to:

- `QK_ROW_PADDING_WORDS = U128_BYTES // U32_BYTES`;
- Q/K stage padding proportional to
  `(stage_rows * HEAD_DIM) // (WARP_SIZE * 2) * (U128_BYTES // U32_BYTES)`;
- `V_PADDING_U32 = 1`, incorporated through the existing V-half and V-tile
  size expressions.

## Required transformation: paired K/V software pipeline

Replace the fully serial per-slice loop with a prologue, paired steady-state
loop, and drain epilogue using the two existing K LDS pages, two global V
register buffers, and two V LDS slices.

- Preserve `_actual_k_slice_from_ordinal` so forward and reverse mirrored-tile
  passes visit the same logical slices as before.
- In the prologue, compute QK for the first slice, start its V global load, and
  fill both K LDS pages for the first even/odd pair before entering the paired
  loop.
- Iterate over pairs of slice ordinals. Alternate K pages and V buffers so QK
  computation/global V fetch for the current or next slice overlaps the LDS
  store, V-fragment fetch, softmax update, and P@V MFMA for the prior result.
  Preload the next even K page before it is consumed and reload the odd page on
  subsequent pairs.
- Add a QK helper for an already-loaded K page. It must only fetch the K
  register fragment, run the correct batch-0 or batch-1 QK MFMA, and apply the
  existing causal mask; it must not issue another global K load.
- Keep two steady-state helper orderings selected by the lower versus upper
  four waves. Both wave groups still perform the complete algorithm and access
  the same data; only order independent QK and V global-load operations
  differently to improve issue overlap. This is not producer/consumer wave
  specialization.
- Split the last softmax/output update into a prepare step and a P@V step. The
  prepare helper computes row max, online-softmax rescaling, probabilities, and
  row sum while the final V load is outstanding. The drain epilogue waits for
  the appropriate V buffer, stores/fetches its parity slice, and performs the
  final batch-0 or batch-1 P@V MFMA.
- Preserve all necessary `s_waitcnt` and workgroup barriers around K page reuse,
  V LDS reuse, and the final output-LDS reuse. Do not add barriers blindly:
  retain overlap while ensuring no wave reads an incomplete or overwritten
  page.

## Required source shape for downstream scheduling

The pipeline's source structure affects the generated instructions and the next
round's scheduler directives. Implement this stage using exactly these four new
JIT helpers (besides the existing helpers):

- `_flash_attn_compute_qk_page_loaded`;
- `_flash_attn_packed_pair_step_wg0`;
- `_flash_attn_packed_pair_step_wg1`;
- `_flash_attn_packed_drain_epilogue`.

Do not introduce a generic `_paired_k_loop`, separate prologue helper,
QK-and-V wrapper, dynamic `has_next` flags, or separate batch-0/batch-1 drain
helpers. Keep the prologue directly in `_flash_attn_packed_process_tile`:
compute the first QK page, issue first V, load K page 1 and then page 0, wait,
barrier, and let the lower four waves issue the second V load.

After the prologue, branch once on `wid < NUM_WARPS // 2`. Put a complete
`for pair_idx in al.range(pair_count)` loop in each branch. Each loop reloads
later odd K pages into page 1, preloads the next even page into page 0, and calls
only its corresponding `wg0` or `wg1` helper. The two helpers take `pair_idx`
and `max_k_slice` and derive `odd_slice` and `next_even_slice` internally. After
the outer branch, call the single parity-selecting drain helper.

This intentional duplication is required. Do not refactor it into a shorter
equivalent implementation even if that version passes correctness or has close
stage-04 timing; the following instruction-scheduling stage relies on this
fixed compiler-visible shape.

For the drain-only softmax prepare helpers, retain the numerical shortcut used
by this stage: when `(block_max - mi) * scale_log2 <= 8`, keep the previous max
and add the new row sum directly instead of evaluating an exponential rescale.
The normal steady-state update helpers keep their existing equations.

## Correctness invariants

- Handle zero-based `max_k_slice` values of either parity. Every causal K slice
  must contribute exactly once, and the drain must use the buffer/page matching
  the final slice.
- Preserve the existing online-softmax equations and the input kernel's
  numerical behavior. The prepare/drain split may move operations but must not
  omit rescaling of the accumulated output.
- Preserve causal grouped-query attention, the packed Q/K/V API, BF16 output,
  FP32 softmax state and accumulation, tile sizes, workgroup size, MFMA shapes,
  mirrored query tiles, and physical grid.

## Scope boundary for this round

This round contains LDS conflict padding and the paired K/V pipeline described
above. Do not introduce later-stage changes:

- no producer-only and consumer-only wave roles;
- no cross-wave handoff protocol or dedicated loader waves;
- no hand-written `sched_group_barrier` instruction scheduling;
- no scheduler barrier policy changes;
- no later softmax rescaling-threshold changes.

The final implementation must pass correctness for sequence lengths 1024,
2048, 4096, 8192, and 16384 and should match the performance level of the
repository's fourth attention optimization stage without fallback compute
paths.

On an otherwise idle MI300X with the repository benchmark defaults, the stage
target is approximately 0.17, 0.58, 2.04, 7.08, and 28.2 ms for those sequence
lengths. Treat results more than 3% slower than these targets as unfinished:
inspect the hot-loop ordering and unnecessary waits/barriers, revise the kernel,
and rerun correctness and performance before stopping.
