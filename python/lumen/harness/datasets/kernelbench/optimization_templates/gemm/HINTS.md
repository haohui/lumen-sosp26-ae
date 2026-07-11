## 1: Prompt 1

Implement the matrix multiplication path with AveLang kernels.

Use `al.amdgpu.mfma_32x32x8_bf16_f32` for matrix multiplication and
`al.amdgpu.raw_buffer_load_x4` for vectorized global-memory loads. The MFMA
instruction computes a 32x32x8 matrix multiplication cooperatively in a wave.
Build larger tiles by issuing multiple MFMA instructions across K and output
subtiles.

MFMA swizzle invariants:
  For A, `i in [0,32)`, `j in [0,8)`: `A(i, j) -> (lane_id = i + (j / 4) * 32, element = j % 4)`
  For B, `j in [0,8)`, `i in [0,32)`: `B(j, i) -> (lane_id = j + (i / 4) * 32, element = i % 4)`

Accumulator invariant for C: for each lane in [0, 64) and acc_idx in [0, 16):
  - col = tile_col_base + (lane % 32)
  - row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

Do not invent a different unpacking. Do not use a naive acc_idx -> contiguous row mapping. Treat this mapping as fixed when writing results.

Stage A and B through LDS. Each thread loads operand fragments in 16-byte chunks. Treat each 16-byte fragment as `(4, al.u32)` and reinterpret it as `2 x (4, al.bf16)`. Feed both `(4, al.bf16)` halves into MFMA in natural order. The intended effect is a cooperative `32x32x16` accumulation from two natural MFMA steps. Do not add lane-dependent or K-dependent control flow to select halves.
  - The two `(4, al.bf16)` halves from one 16-byte LDS load collectively represent a swizzled `32x16` operand contribution with 4-column interleaving.
  - Consuming them in natural order must produce the same final C as a naive conceptual layout because operand pairings remain consistent under MFMA swizzle.

Scale the kernel from one wave to four waves without changing the MFMA per-wave invariant. Interpret the 4 warps as a 2 x 2 warp grid. Keep MFMA math identical per warp. Only add warp ownership offsets at operand fetch and output writeback.

## 2: Prompt 2

Implement software pipelining to overlap MFMA, LDS access and global memory access. Use double buffering. Unroll the K-loop by 2 to minimize branching. Split the LDS access and overlap with MFMA in a fine-grain way to reduce the size of the working sets of the shared memroy A/B.

## 3: Prompt 3

Utilize the range encoded by the buffer resource descriptors used with
`al.amdgpu.raw_buffer_load_x4` and `al.amdgpu.raw_buffer_store_*` to remove
explicit branches guarding OOB access. The range is in bytes. With the proper
range, OOB loads return zero and OOB stores are discarded. Computation and LDS
access therefore remain safe with zero values. Removing branches in the loop is
more beneficial than reducing the extra computation and LDS access.
