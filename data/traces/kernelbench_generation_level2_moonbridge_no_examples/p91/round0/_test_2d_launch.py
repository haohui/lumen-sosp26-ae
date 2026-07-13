import torch
import sys
sys.path.insert(0, '.')
import _test_2d_conv as tk

B, IC, OC, H, W, K, S, P = 1, 4, 8, 8, 8, 3, 2, 1
OH = (H - 1) * S - 2 * P + K + 1
OW = OH
TILE_H, TILE_W = 4, 4

x = torch.randn(B, IC, H, W, dtype=torch.bfloat16, device='cuda')
weight = torch.randn(IC, OC, K, K, dtype=torch.bfloat16, device='cuda')
bias = torch.randn(OC, dtype=torch.bfloat16, device='cuda')
out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')

tiles_h = (OH + TILE_H - 1) // TILE_H
tiles_w = (OW + TILE_W - 1) // TILE_W
bc_extent = B * OC

print(f'grid=({tiles_h}, {tiles_w}, {bc_extent}), block=({TILE_H}, {TILE_W}, 1)')

tk.conv_transpose2d_kernel_2d[lambda: (
    (tiles_h, tiles_w, bc_extent),
    (TILE_H, TILE_W, 1),
)](
    x, weight, bias, out,
    B, IC, OC, H, W, OH, OW, K, S, P,
    TILE_H, TILE_W,
)
torch.cuda.synchronize()
print('Launch OK')
