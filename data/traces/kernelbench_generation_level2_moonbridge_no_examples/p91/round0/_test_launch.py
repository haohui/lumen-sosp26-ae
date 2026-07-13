import torch
import sys
sys.path.insert(0, '.')
import _test_all_kernels as tk

# Small test: B=1, IC=4, OC=8, H=8, W=8, K=3, S=2, P=1
B, IC, OC, H, W, K, S, P = 1, 4, 8, 8, 8, 3, 2, 1
OH = (H - 1) * S - 2 * P + K + 1  # (8-1)*2 - 2 + 3 + 1 = 14 - 2 + 3 + 1 = 16
OW = OH

x = torch.randn(B, IC, H, W, dtype=torch.bfloat16, device='cuda')
weight = torch.randn(IC, OC, K, K, dtype=torch.bfloat16, device='cuda')
bias = torch.randn(OC, dtype=torch.bfloat16, device='cuda')
out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')

TILE_SIZE = 256
spatial_size = OH * OW
spatial_tiles = (spatial_size + TILE_SIZE - 1) // TILE_SIZE
bc_extent = B * OC

print(f'Launching conv kernel: grid=({spatial_tiles},{bc_extent},1), block=({TILE_SIZE},1,1)')
print(f'OH={OH}, OW={OW}, spatial_size={spatial_size}')

tk.conv_transpose2d_kernel[lambda: (
    (spatial_tiles, bc_extent, 1),
    (TILE_SIZE, 1, 1),
)](
    x, weight, bias, out,
    B, IC, OC, H, W, OH, OW,
    K, S, P, spatial_size, TILE_SIZE,
)

torch.cuda.synchronize()
print('Conv kernel launched OK')
print('Output sample:', out[0, 0, 0, :4])
