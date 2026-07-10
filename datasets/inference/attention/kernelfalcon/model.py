import torch
import triton
import triton.language as tl


@triton.jit
def _fused_fa2_causal_bshd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, S, HQ, HK, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused FA2 main path (causal) for external [B, S, H, D] layout:
      1) tiled QK^T
      2) online softmax
      3) tiled P @ V
      4) store O
    """
    pid_m = tl.program_id(axis=0)   # query block index along S
    pid_bh = tl.program_id(axis=1)  # flattened (B, HQ)

    b = pid_bh // HQ
    hq = pid_bh % HQ

    # MQA/GQA mapping equivalent to repeat_interleave on KV heads
    group_size = HQ // HK
    kv_h = hq // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile once: [BLOCK_M, BLOCK_D]
    q_ptrs = (
        q_ptr
        + b * stride_qb
        + offs_m[:, None] * stride_qs
        + hq * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # bf16 expected by test

    # Online softmax state per query row
    valid_m = offs_m < S
    m_i = tl.where(valid_m, -float("inf"), 0.0).to(tl.float32)
    l_i = tl.where(valid_m, 0.0, 1.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # scale = 1/sqrt(D), implemented in log2 domain for exp2 softmax
    d_f32 = tl.full((), D, tl.float32)
    qk_scale_log2 = 1.4426950408889634 / tl.sqrt(d_f32)  # log2(e) / sqrt(D)

    # Causal optimization: keys beyond (pid_m+1)*BLOCK_M are never needed
    n_end = tl.minimum(S, (pid_m + 1) * BLOCK_M)

    for start_n in tl.range(0, n_end, BLOCK_N):
        offs_n = start_n + offs_n_base

        # Load K: [BLOCK_N, BLOCK_D]
        k_ptrs = (
            k_ptr
            + b * stride_kb
            + offs_n[:, None] * stride_ks
            + kv_h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale_log2

        # causal + bounds mask
        attn_mask = (
            (offs_m[:, None] < S)
            & (offs_n[None, :] < S)
            & (offs_n[None, :] <= offs_m[:, None])
        )
        qk = tl.where(attn_mask, qk, -float("inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V: [BLOCK_N, BLOCK_D]
        v_ptrs = (
            v_ptr
            + b * stride_vb
            + offs_n[:, None] * stride_vs
            + kv_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # P @ V accumulate (tensor-core friendly path)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        m_i = m_ij

    # Normalize and store
    out = acc / l_i[:, None]
    o_ptrs = (
        o_ptr
        + b * stride_ob
        + offs_m[:, None] * stride_os
        + hq * stride_oh
        + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=o_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton causal attention forward for [B, S, H, D].

    Fused stages inside one kernel launch:
      - QK^T tile matmul
      - causal masking
      - online softmax
      - P@V tile matmul
      - final cast/store

    Wrapper does only validation/allocation/launch (no PyTorch compute path).
    """
    if not (isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)):
        raise TypeError("Q, K, V must be torch.Tensor")

    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Expected Q, K, V with shape [B, S, H, D]")

    B, S, HQ, D = Q.shape
    Bk, Sk, HK, Dk = K.shape
    Bv, Sv, HV, Dv = V.shape

    if not (B == Bk == Bv):
        raise ValueError("Batch dimension mismatch")
    if not (S == Sk == Sv):
        raise ValueError("Sequence length mismatch")
    if not (D == Dk == Dv):
        raise ValueError("Head dim mismatch")
    if HK != HV:
        raise ValueError("K/V head count mismatch")
    if HQ % HK != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError("Q, K, V must be on same device")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("Q, K, V dtype mismatch")
    if Q.dtype != torch.bfloat16:
        raise ValueError("This kernel expects bfloat16 tensors for this test")
    if D > 128:
        raise ValueError("This implementation supports D <= 128")

    O = torch.empty_like(Q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    grid = (triton.cdiv(S, BLOCK_M), B * HQ)

    _fused_fa2_causal_bshd_kernel[grid](
        Q, K, V, O,
        B, S, HQ, HK, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return O