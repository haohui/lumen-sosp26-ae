import torch
import sys
sys.path.insert(0, '.')
import _test_all_kernels as tk

B, IC, OC, H, W, K, S, P = 1, 4, 8, 8, 8, 3, 2, 1
OH = (H - 1) * S - 2 * P + K + 1
OW = OH
TILE_SIZE = 256
spatial_size = OH * OW
spatial_tiles = (spatial_size + TILE_SIZE - 1) // TILE_SIZE
bc_extent = B * OC

x = torch.randn(B, IC, H, W, dtype=torch.bfloat16, device='cuda')
weight = torch.randn(IC, OC, K, K, dtype=torch.bfloat16, device='cuda')
conv_bias = torch.randn(OC, dtype=torch.bfloat16, device='cuda')
extra_bias = torch.randn(OC, 1, 1, dtype=torch.bfloat16, device='cuda')
scale = 2.0

# Step 1: Conv
conv_out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')
tk.conv_transpose2d_kernel[lambda: ((spatial_tiles, bc_extent, 1), (TILE_SIZE, 1, 1))](
    x, weight, conv_bias, conv_out,
    B, IC, OC, H, W, OH, OW, K, S, P, spatial_size, TILE_SIZE,
)
torch.cuda.synchronize()
print('Conv done')

# Step 2: Softmax
softmax_out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')
tk.softmax_channel_kernel[lambda: ((spatial_size, B, 1), (OC, 1, 1))](
    conv_out, softmax_out, B, OC, H, W,
)
torch.cuda.synchronize()
print('Softmax done')

# Step 3: Bias + Scale + Sigmoid
final_out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')
tk.bias_scale_sigmoid_kernel[lambda: ((spatial_tiles, bc_extent, 1), (TILE_SIZE, 1, 1))](
    softmax_out, extra_bias, final_out,
    B, OC, OH, OW, float(scale), spatial_size, TILE_SIZE,
)
torch.cuda.synchronize()
print('Final done')
print('Output sample:', final_out[0, 0, 0, :4])
