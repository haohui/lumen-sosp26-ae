Triton Kernel Dump for 03_kernelfalcon (5 datapoints, winner-only)

Environment used:
- container: kernel-benchmark-rocm-traffic
- TRITON_KERNEL_DUMP=1
- dtype: torch.bfloat16

Cleanup status:
- removed raw dump dirs:
  - triton_dump_20260323_155721
  - triton_dump_20260323_155857
- kept this winner-only package: triton_kernel_dump_5points

Datapoints (S):
- 1024
- 2048
- 4096
- 8192
- 16384

Fixed shape params:
- batch_size=16
- num_q_heads=8
- num_kv_heads=1
- head_dim=128
- causal=True

Each shape directory keeps only the winner kernel artifacts:
- *.ttir
- *.ttgir
- *.llir
- *.amdgcn
- *.hsaco

Winner config per shape:
- S=1024: BLOCK_M=128, BLOCK_N=64, BLOCK_D=128, num_warps=8, num_stages=1, median_ms=0.48957416534423825
- S=2048: BLOCK_M=128, BLOCK_N=64, BLOCK_D=128, num_warps=8, num_stages=1, median_ms=1.678370361328125
- S=4096: BLOCK_M=128, BLOCK_N=64, BLOCK_D=128, num_warps=8, num_stages=1, median_ms=6.77339859008789
- S=8192: BLOCK_M=128, BLOCK_N=64, BLOCK_D=128, num_warps=8, num_stages=1, median_ms=26.62459716796875
- S=16384: BLOCK_M=128, BLOCK_N=64, BLOCK_D=128, num_warps=8, num_stages=1, median_ms=104.65070343017578

Winner hash mapping:
- S=1024: 43GS56HUQDQP2VUXURVKCZTH5Y4KKNT5UCEY2EHYUQC7YPVTYPJQ
- S=2048: 43GS56HUQDQP2VUXURVKCZTH5Y4KKNT5UCEY2EHYUQC7YPVTYPJQ
- S=4096: 43GS56HUQDQP2VUXURVKCZTH5Y4KKNT5UCEY2EHYUQC7YPVTYPJQ
- S=8192: 43GS56HUQDQP2VUXURVKCZTH5Y4KKNT5UCEY2EHYUQC7YPVTYPJQ
- S=16384: 43GS56HUQDQP2VUXURVKCZTH5Y4KKNT5UCEY2EHYUQC7YPVTYPJQ
