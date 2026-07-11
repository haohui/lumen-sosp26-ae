import torch
import torch.nn as nn

batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = 1024 
head_dim = 128


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # Keep the computation flow explicit for prompt consistency.
        assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
        bq, sq, hq, dq = Q.shape
        bk, sk, hk, dk = K.shape
        bv, sv, hv, dv = V.shape

        assert (bq, sq, dq) == (bk, sk, dk) == (bv, sv, dv)
        assert hq == num_q_heads
        assert hk == num_kv_heads 
        assert hv == hk
        assert dq == head_dim 

        if hk != hq:
            groups = hq // hk
            K = K.repeat_interleave(groups, dim=2)
            V = V.repeat_interleave(groups, dim=2)

        qf = Q.to(torch.float32)
        kf = K.to(torch.float32)
        vf = V.to(torch.float32)
        sm_scale = 0.08838834764831843

        scores = torch.einsum("bthd,bshd->bhts", qf * sm_scale, kf)
        causal_mask = torch.triu(
            torch.ones((sq, sk), device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", probs, vf)
        return out.to(torch.bfloat16)

def get_inputs():
    Q = torch.randn(
        batch_size, sequence_length, num_q_heads, head_dim, dtype=torch.bfloat16
    )
    K = torch.randn(
        batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    V = torch.randn(
        batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    return [Q, K, V]


def get_init_inputs():
    return []
