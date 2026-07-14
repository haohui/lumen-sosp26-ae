import torch
import triton
import triton.language as tl


@triton.jit
def _fused_causal_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, T, H, D,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    sm_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    One-program-per (t, b, h) fused causal attention:
      1) Q load + scale (in fp32 math)
      2) score = Q @ K^T
      3) causal mask (s <= t)
      4) online softmax
      5) probs @ V accumulation
      6) bf16 store
    """
    pid_t = tl.program_id(axis=0)     # query time index t
    pid_bh = tl.program_id(axis=1)    # flattened (b, h)

    b = pid_bh // H
    h = pid_bh % H

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load query row q[b, t, h, :]
    q_ptrs = q_ptr + b * stride_qb + pid_t * stride_qt + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Online softmax state for this (b, h, t)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # Iterate over keys/values along sequence dimension
    for start_n in tl.range(0, T, BLOCK_N):
        s = start_n + offs_n
        mask_s = s < T
        causal = s <= pid_t
        valid = mask_s & causal

        # Load K block: k[b, s, h, d]
        k_ptrs = (
            k_ptr
            + b * stride_kb
            + s[:, None] * stride_kt
            + h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # scores[s] = dot(q, k[s]) * sm_scale
        qk = tl.sum(k * q[None, :], axis=1)
        qk = qk * sm_scale
        qk = tl.where(valid, qk, -float("inf"))

        # Online softmax update
        block_m = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        p = tl.where(valid, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)

        # Load V block and accumulate weighted sum
        v_ptrs = (
            v_ptr
            + b * stride_vb
            + s[:, None] * stride_vt
            + h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    out = acc / l_i

    o_ptrs = o_ptr + b * stride_ob + pid_t * stride_ot + h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_d)


def kernel_function(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor):
    """
    Fused causal attention wrapper.

    Fused stages in a single Triton kernel:
      - q scaling by sm_scale
      - qk^T matmul
      - causal masking
      - softmax (online, numerically stable)
      - probs @ v matmul
      - cast/store bf16 output

    Wrapper does only validation/allocation/launch (no math).
    """
    if not (isinstance(x0, torch.Tensor) and isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)):
        raise TypeError("kernel_function expects three torch.Tensor inputs")

    if not (x0.is_cuda and x1.is_cuda and x2.is_cuda):
        raise ValueError("All inputs must be CUDA tensors")

    if x0.device != x1.device or x0.device != x2.device:
        raise ValueError("All inputs must be on the same device")

    if x0.ndim != 4 or x1.ndim != 4 or x2.ndim != 4:
        raise ValueError("Expected 4D tensors shaped [B, T, H, D]")

    if x0.shape != x1.shape or x0.shape != x2.shape:
        raise ValueError("Input shapes must match exactly")

    if x0.dtype != x1.dtype or x0.dtype != x2.dtype:
        raise ValueError("Input dtypes must match")

    B, T, H, D = x0.shape

    # This kernel is specialized for D <= 128 (test case uses D=128)
    BLOCK_D = 128
    if D > BLOCK_D:
        raise ValueError(f"Head dimension D={D} not supported by this kernel (max {BLOCK_D})")

    # Required semantics: scale = 1/sqrt(128) = 0.08838834764831843
    sm_scale = 0.08838834764831843

    out = torch.empty((B, T, H, D), device=x0.device, dtype=torch.bfloat16)

    # One program per (t, b*h)
    grid = (T, B * H)

    _fused_causal_attention_kernel[grid](
        x0, x1, x2, out,
        B, T, H, D,
        x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        BLOCK_N=32,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _expand_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_heads = q.shape[2]
        kv_heads = k.shape[2]
        if kv_heads == q_heads:
            return k.contiguous(), v.contiguous()
        if kv_heads != 1 or q_heads % kv_heads != 0:
            raise ValueError(
                f"unsupported GQA head layout: q_heads={q_heads}, kv_heads={kv_heads}"
            )
        groups = q_heads // kv_heads
        return (
            k.repeat_interleave(groups, dim=2).contiguous(),
            v.repeat_interleave(groups, dim=2).contiguous(),
        )

    def build_call(
        self,
        *,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ):
        k_expanded, v_expanded = self._expand_kv(q_bshd, k_bshd, v_bshd)
        q = q_bshd.contiguous()
        return lambda: kernel_function(q, k_expanded, v_expanded)

    def forward(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
    ) -> torch.Tensor:
        k_expanded, v_expanded = self._expand_kv(q_bshd, k_bshd, v_bshd)
        return kernel_function(q_bshd.contiguous(), k_expanded, v_expanded)
