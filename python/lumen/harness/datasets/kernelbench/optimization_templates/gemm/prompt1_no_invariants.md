## 1: Prompt 1

Implement the matrix multiplication path with AveLang kernels.

Use `al.amdgpu.mfma_32x32x8_bf16_f32` for matrix multiplication and
`al.amdgpu.raw_buffer_load_x4` for vectorized global-memory loads. The MFMA
instruction computes a 32x32x8 matrix multiplication cooperatively in a wave.
Build larger tiles by issuing multiple MFMA instructions across K and output
subtiles.

Stage A and B through LDS. Each thread loads operand fragments in 16-byte chunks. Treat each 16-byte fragment as `(4, al.u32)` and reinterpret it as `2 x (4, al.bf16)`. Feed both `(4, al.bf16)` halves into MFMA in natural order. The intended effect is a cooperative `32x32x16` accumulation from two natural MFMA steps. Do not add lane-dependent or K-dependent control flow to select halves.
  - The two `(4, al.bf16)` halves from one 16-byte LDS load collectively represent a swizzled `32x16` operand contribution with 4-column interleaving.
  - Consuming them in natural order must produce the same final C as a naive conceptual layout because operand pairings remain consistent under MFMA swizzle.

Scale the kernel from one wave to four waves without changing the MFMA per-wave invariant. Interpret the 4 warps as a 2 x 2 warp grid. Keep MFMA math identical per warp. Only add warp ownership offsets at operand fetch and output writeback.
