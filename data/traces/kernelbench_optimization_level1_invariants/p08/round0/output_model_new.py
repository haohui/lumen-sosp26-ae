import torch
import avelang
import avelang.language as al


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K_dim: al.i32,
    N: al.i32,
):
    # --- tensor views ---
    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K_dim), (K_dim, 1)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((K_dim, N), (N, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    # --- thread / warp decomposition ---
    tid = al.thread_id(0)
    lane = tid % 64
    lane_group = lane // 32

    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    block_m = al.block_id(1) * 64
    block_n = al.block_id(0) * 64

    # --- LDS: one row per lane for both A and B ---
    a_smem = al.make_shared((256, 4), al.i32)
    b_smem = al.make_shared((256, 4), al.i32)

    # K-interleaving offset pattern for A: [0,1,2,3, 8,9,10,11]
    a_k_offs = al.make_local((8,), al.i32)
    a_k_offs[0] = 0
    a_k_offs[1] = 1
    a_k_offs[2] = 2
    a_k_offs[3] = 3
    a_k_offs[4] = 8
    a_k_offs[5] = 9
    a_k_offs[6] = 10
    a_k_offs[7] = 11

    acc = al.full((16,), 0.0, al.f32)

    # K loop
    for k_block in al.range(0, K_dim, 16):
        # -- load A: 8 bf16 per thread, K-interleaved, pack into 4 i32 --
        a_row = block_m + warp_row * 32 + lane % 32
        # lane_group 0: K=[0,1,2,3,8,9,10,11], lane_group 1: K=[4,5,6,7,12,13,14,15]
        a_k_base = k_block + lane_group * 4

        for p in al.range(4):
            lo_u16 = al.convert(0, al.u16)
            hi_u16 = al.convert(0, al.u16)
            if a_row < M:
                k0 = a_k_base + a_k_offs[p * 2]
                k1 = a_k_base + a_k_offs[p * 2 + 1]
                if k0 < K_dim:
                    lo_u16 = al.bitcast(A[a_row, k0], al.u16)
                if k1 < K_dim:
                    hi_u16 = al.bitcast(A[a_row, k1], al.u16)
            lo_i32 = al.convert(lo_u16, al.i32)
            hi_i32 = al.convert(hi_u16, al.i32)
            a_smem[tid][p] = lo_i32 | (hi_i32 << 16)

        # -- load B along K: per-lane data, pack into 4 i32 --
        # lane L: N column n = warp_col*32 + L%32
        # lane_group 0: K=[0..3, 8..11], lane_group 1: K=[4..7, 12..15]
        b_n_col = warp_col * 32 + lane % 32
        b_k_offset = lane_group * 4

        for p in al.range(4):
            lo_u16 = al.convert(0, al.u16)
            hi_u16 = al.convert(0, al.u16)
            # p=0,1: step 1 K (b_k_offset..b_k_offset+3), p=2,3: step 2 K (b_k_offset+8..b_k_offset+11)
            bk = k_block + b_k_offset + (p // 2) * 8 + (p % 2) * 2
            bn = block_n + b_n_col
            if bk < K_dim:
                if bn < N:
                    lo_u16 = al.bitcast(B[bk, bn], al.u16)
                # second element from next K row at same N column
                bk2 = bk + 1
                if bk2 < K_dim:
                    hi_u16 = al.bitcast(B[bk2, bn], al.u16)
            lo_i32 = al.convert(lo_u16, al.i32)
            hi_i32 = al.convert(hi_u16, al.i32)
            b_smem[tid][p] = lo_i32 | (hi_i32 << 16)

        al.syncthreads()

        # -- MFMA: two calls per K tile --
        a_words = a_smem[tid]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))

        b_words = b_smem[tid]
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        al.syncthreads()

    # --- writeback ---
    warp_m = block_m + warp_row * 32
    warp_n = block_n + warp_col * 32
    for a_idx in al.range(16):
        col = warp_n + lane % 32
        row = warp_m + 8 * (a_idx // 4) + 4 * (lane // 32) + a_idx % 4
        if row < M:
            if col < N:
                C[row, col] = al.convert(acc[a_idx], al.bf16)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8205, 2949) or tuple(B.shape) != (2949, 5921):
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8205, 5921), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((93, 129, 1), (256, 1, 1))](
            A_ptr=A, B_ptr=B, C_ptr=C,
            M=8205, K_dim=2949, N=5921,
        )
        return C
