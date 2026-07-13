import struct
import torch
import torch.nn as nn

from gemm_kernel import gemm_kernel
from gn_kernel import groupnorm_leakyrelu_double_kernel


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)

    def forward(self, x):
        x = x.contiguous()
        B = x.shape[0]
        K_val = x.shape[1]
        N_val = self.hidden_size
        G_val = self.num_groups

        eps_bits = struct.unpack('<i', struct.pack('<f', float(self.eps)))[0]
        neg_bits = struct.unpack('<i', struct.pack('<f', float(self.negative_slope)))[0]

        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_bf16 = self.fc.weight.data.to(dtype=torch.bfloat16).contiguous()
        bias_bf16 = self.fc.bias.data.to(dtype=torch.bfloat16).contiguous()
        gn_w_bf16 = self.gn.weight.data.to(dtype=torch.bfloat16).contiguous()
        gn_b_bf16 = self.gn.bias.data.to(dtype=torch.bfloat16).contiguous()

        c_bf16 = torch.empty(B, N_val, dtype=torch.bfloat16, device=x.device)

        threads_per_block = 128
        cols_per_thread = 8
        cols_per_block = threads_per_block * cols_per_thread
        grid_m = B
        grid_n = (N_val + cols_per_block - 1) // cols_per_block

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (threads_per_block, 1, 1))](
            x_bf16, w_bf16, bias_bf16, c_bf16,
            B, N_val, K_val,
            cols_per_thread, 8, 8,
        )

        out_bf16 = torch.empty(B, N_val, dtype=torch.bfloat16, device=x.device)
        total_pairs = B * G_val
        grid_gn = (total_pairs + 255) // 256

        groupnorm_leakyrelu_double_kernel[lambda: ((grid_gn, 1, 1), (256, 1, 1))](
            c_bf16, gn_w_bf16, gn_b_bf16, out_bf16,
            B, N_val, G_val,
            eps_bits, neg_bits,
        )

        return out_bf16
