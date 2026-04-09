import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if K.shape[1] == 1:
            K = K.repeat_interleave(8, dim=1)
            V = V.repeat_interleave(8, dim=1)

        return F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )


batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
head_dim = 128
supported_sequence_lengths = (1024, 2048, 4096, 8192, 16384)


def get_inputs():
    Q = torch.randn(batch_size, num_q_heads, sequence_length, head_dim, dtype=torch.bfloat16)
    K = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16)
    return [Q, K, V]


def get_init_inputs():
    return []
