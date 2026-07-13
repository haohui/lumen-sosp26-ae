# Optimization 06: add LDS padding

Optimize `input_model.py` by adding padding to the existing LDS layouts to
reduce bank conflicts. Preserve the complete global ownership, MFMA lane
mapping, accumulator mapping, raw-buffer operations, and software pipeline.

This round changes only the physical LDS address of each logical A/B value.
The public API and result remain unchanged.

## Frozen dataflow

Do not change:

- shape dispatch or tile sizes;
- global-load ownership or raw-buffer widths;
- register-fragment ownership;
- LDS-to-MFMA logical row and K ownership;
- MFMA operand order, fragment order, or count;
- accumulator-to-C mapping and raw-buffer writeback;
- pipeline prologue, steady-state ordering, epilogue, or barriers.

Do not add scheduling hints, stagger, workgroup remapping, new buffering, or
fallback compute paths.

## Batch2 padded layout

Batch2 stores one 16-byte vector `(4, al.u32)` containing 8 BF16 values. A K64
row therefore contains 8 vectors. Add two padding vectors, or 32 bytes, after
each logical row group.

Use:

- 64x64 batch2: `READ_ROWS_A=2`, `READ_ROWS_B=2`;
- 128x128 batch2: `READ_ROWS_A=4`, `READ_ROWS_B=4`.

For A and B independently:

```text
VECTORS_PER_ROW = GROUP_K // 8
PADDING_VECTORS = 32 // 16
VECTORS_PER_GROUP = READ_ROWS * VECTORS_PER_ROW + PADDING_VECTORS

row_group = row // READ_ROWS
row_in_group = row % READ_ROWS
physical_vector = row_group * VECTORS_PER_GROUP
                  + row_in_group * VECTORS_PER_ROW
                  + vector_in_row
```

Resize each LDS allocation to include the padding. Apply this same physical
address function in both the global-register-to-LDS store helper and the
LDS-to-MFMA read helper. The logical `(row, K vector)` values must not change.

## Batch4 padded layout

Batch4 addresses LDS in packed `al.u32` words. One K64 row contains:

```text
ROW_WORDS = GROUP_K * 2 // 4
```

Use:

- A: `READ_ROWS_A=1`;
- B: `READ_ROWS_B=8`;
- both operands: `PAD_WORDS=8 // 4`, or 8 padding bytes per row group.

For A and B independently:

```text
GROUP_WORDS = READ_ROWS * ROW_WORDS + PAD_WORDS
row_group = row // READ_ROWS
row_in_group = row % READ_ROWS
physical_word = row_group * GROUP_WORDS
                + row_in_group * ROW_WORDS
                + word_in_row
```

Resize LDS and use this formula consistently in stores and reads. Each MFMA
operand still reads the same two adjacent logical words.

## Verification

Run the command in `AGENTS.md`, first for 1024/2048 and then for 4096 before
all five sizes. Compare performance with `input_model.py`; this round is useful
only if padding reduces LDS conflicts without harming another family.

Finish only when:

- every record has `"correctness":true`;
- each logical value has the same owner and MFMA consumer as before;
- LDS stores and reads use identical padded-address formulas;
- no pipeline operation or barrier moved;
- only LDS sizes and physical address calculations changed.
