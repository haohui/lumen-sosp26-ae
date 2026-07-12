# Optimization 05: preserve the stage-04 kernel and verify the stage boundary

Carry the existing AveLang FlashAttention implementation forward unchanged as
the fifth stage. Write the complete input implementation to
`output_model_new.py`, then validate it.

This repository's fifth reference file does not introduce an effective runtime
kernel transformation relative to stage 04. It contains duplicate definitions
of the loaded-page QK helper, the two wave-ordered pair-step helpers, and the
drain epilogue; the later identical definitions shadow the earlier ones. Those
duplicates do not change the active functions or measured performance and
should not be recreated.

## Required action

- Preserve the padded Q/K/V LDS layouts, direct global-to-LDS Q/K loads,
  transposed V representation, paired K/V pipeline, wave-dependent issue
  ordering, mirrored query-tile execution, and drain softmax shortcut exactly.
- Copy the complete working kernel into `output_model_new.py`; do not leave the
  output empty and do not replace it with an import or wrapper.
- Run correctness and performance validation on the copied implementation.

## Scope boundary

- Do not add duplicate or dead helper definitions merely to mimic source-file
  text that has no runtime effect.
- Do not add producer/consumer roles, loader-only waves, new cross-wave
  synchronization, scheduler-group barriers, instruction scheduling, or
  softmax-threshold changes.
- Do not change public APIs, tile/grid shapes, MFMA operations, LDS allocation,
  numerical behavior, or fallback paths.

The final implementation must pass correctness for sequence lengths 1024,
2048, 4096, 8192, and 16384. Its performance should remain at the stage-04/05
level: approximately 0.17, 0.58, 2.04, 7.07, and 28.1 ms on an otherwise idle
MI300X with repository benchmark defaults. If copying or formatting changes the
performance materially, restore the input implementation exactly and retest.
