# Optimization 02: transpose V while staging it in LDS

Optimize the existing AveLang FlashAttention kernel by changing only the LDS
layout used for V. The current kernel loads V coalescently from global memory,
stores it in a row-major LDS matrix, and later gathers individual BF16 values
from LDS to assemble the register fragments consumed by the P @ V MFMA.

Keep the coalesced global loads, but transpose and repack each V tile as it is
written to LDS so that every wave can load its MFMA operand fragments directly
from the consumption layout. The goal is to remove the scalar LDS gather and
temporary BF16 assembly from the V register-fetch path.

## Required transformation

- Preserve the global V layout and the existing four packed `u32` values loaded
  by each participating thread.
- During the LDS store, separate the even and odd BF16 lanes from those packed
  values. AMD `perm` operations are appropriate for forming packed even-row and
  odd-row values without scalar unpacking.
- Store the even and odd values into separate logical halves of the existing V
  LDS allocation. Index the layout by row-pair group and V column so that the
  later P @ V MFMA lane mapping can address packed fragments directly.
- Replace the row-major BF16 gather in the V register-fetch helper with a packed
  `u32` view of the transposed LDS layout. Derive indices from the output batch,
  V chunk, lane pair, lane half, and lane parity, then copy the packed values
  directly into the existing `v_regs` fragment shape.
- Preserve the existing `v_regs` meaning expected by both P @ V MFMA helpers.
  Do not change the MFMA operations or their accumulator mapping.
- Keep all LDS accesses in bounds and preserve the existing barriers between V
  stores and V loads.

## Correctness invariants

- Preserve causal grouped-query attention semantics exactly.
- Preserve the packed Q/K/V interface, `seq_ptr`, `max_seq_len`, BF16 output,
  FP32 softmax state, and FP32 output accumulation.
- Preserve the numerical ordering and online-softmax rescaling logic.
- Preserve the Q and K LDS layouts and the Q @ K MFMA path.
- Preserve tile sizes, workgroup dimensions, grid mapping, and shared-memory
  allocation size unless a small layout-only correction is strictly required.

## Scope boundary for this round

This round is only the V-transpose optimization. Do not introduce optimizations
from later rounds:

- no asynchronous global-to-LDS copy;
- no new LDS padding or bank-conflict swizzle;
- no software pipeline or double buffering;
- no producer/consumer wave specialization;
- no hand-written instruction scheduling or scheduler-barrier changes;
- no changes to query-tile mirroring or workgroup scheduling.

The final implementation must pass correctness for sequence lengths 1024,
2048, 4096, 8192, and 16384 and should improve the measured performance over
the input implementation without using fallback compute paths.
