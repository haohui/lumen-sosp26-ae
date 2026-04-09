Triton Kernel Dump for 03_kernelfalcon MoE (5 datapoints, winner-only per variant)

Environment used:
- container: kernel-benchmark-rocm-traffic
- TRITON_KERNEL_DUMP=1
- TRITON_PRINT_AUTOTUNING=1
- TRITON_ALWAYS_COMPILE=1
- HIP_VISIBLE_DEVICES=5
- CPU_AFFINITY=112-127
- device=cuda:0

Datapoints (seq): 1024, 2048, 4096, 8192, 16384
Fixed params: dim=7168, inter_dim=2048, experts=32, topk=4, input_dtype=fp8

S=1024:
- fused median_ms=16.581751
- nofused median_ms=25.527199
- faster_variant=fused
- fused selected kernels (6):
  - _count_routes_kernel | hash=FBW4UTKQWXCQGP45HIWQCXKYSMWXNU62ESJBQAPPB6N2IOCGI5ZA | meta={"num_warps": 1}
  - _prefix_prepare_kernel | hash=V3OKPK3G435DLW3DMNSA2UFFV6PVJ7MQPSEHMIKFGLVRDV3NMCHQ | meta={"num_warps": 1}
  - _scatter_routes_kernel | hash=26WTMJA6HMCLYB4EGZ6JSNW5TMLQMJX3WSVPI23RZ5GMFJA72PAA | meta={"num_warps": 1}
  - _fill_group_expert_kernel | hash=NR3HA7WR25OXWX4AHDW6RINCXQCUA4B2TZWHDL47A57MSLUEMS5A | meta={"num_warps": 1}
  - _fc1_silu_grouped_kernel | hash=U6QDCKRLG3R6WREWKTWIV5WPJLB26YU5Z3A5V7FQR55RZBXN6GJA | meta={"num_warps": 8}
  - _fc2_reduce_tiled_kernel | hash=R3OPQ3BGAQL4WIDW2OZJSIRYLBDHWMQKJQXH26LJAYL6TF7LPZGQ | meta={"num_warps": 4, "DB_TILE": 1}
- fused autotune winners:
  - _fc1_silu_grouped_kernel: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _fc2_reduce_tiled_kernel: DB_TILE: 1, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
- nofused selected kernels (2):
  - _moe_fc1_topk4_kernel | hash=YC3V2GL2KOBR54QKTFN7AV5CEHUXMFPNMO6VLY5DFVXYFPWWQDNQ | meta={"num_warps": 4, "BI": 32}
  - _moe_fc2_topk4_kernel | hash=RDPPSDUGCBL2CYJLWQ2LW5HRWLQPE5DFUB4435WOCWPQXQDSD7AQ | meta={"num_warps": 8, "BO": 128}
- nofused autotune winners:
  - _moe_fc1_topk4_kernel: BI: 32, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
  - _moe_fc2_topk4_kernel: BO: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

S=2048:
- fused median_ms=31.638515
- nofused median_ms=51.398109
- faster_variant=fused
- fused selected kernels (6):
  - _count_routes_kernel | hash=FBW4UTKQWXCQGP45HIWQCXKYSMWXNU62ESJBQAPPB6N2IOCGI5ZA | meta={"num_warps": 1}
  - _prefix_prepare_kernel | hash=V3OKPK3G435DLW3DMNSA2UFFV6PVJ7MQPSEHMIKFGLVRDV3NMCHQ | meta={"num_warps": 1}
  - _scatter_routes_kernel | hash=26WTMJA6HMCLYB4EGZ6JSNW5TMLQMJX3WSVPI23RZ5GMFJA72PAA | meta={"num_warps": 1}
  - _fill_group_expert_kernel | hash=NR3HA7WR25OXWX4AHDW6RINCXQCUA4B2TZWHDL47A57MSLUEMS5A | meta={"num_warps": 1}
  - _fc1_silu_grouped_kernel | hash=U6QDCKRLG3R6WREWKTWIV5WPJLB26YU5Z3A5V7FQR55RZBXN6GJA | meta={"num_warps": 8}
  - _fc2_reduce_tiled_kernel | hash=R3OPQ3BGAQL4WIDW2OZJSIRYLBDHWMQKJQXH26LJAYL6TF7LPZGQ | meta={"num_warps": 4, "DB_TILE": 1}
- fused autotune winners:
  - _fc1_silu_grouped_kernel: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _fc2_reduce_tiled_kernel: DB_TILE: 1, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
- nofused selected kernels (2):
  - _moe_fc1_topk4_kernel | hash=YC3V2GL2KOBR54QKTFN7AV5CEHUXMFPNMO6VLY5DFVXYFPWWQDNQ | meta={"num_warps": 4, "BI": 32}
  - _moe_fc2_topk4_kernel | hash=RDPPSDUGCBL2CYJLWQ2LW5HRWLQPE5DFUB4435WOCWPQXQDSD7AQ | meta={"num_warps": 8, "BO": 128}
- nofused autotune winners:
  - _moe_fc1_topk4_kernel: BI: 32, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
  - _moe_fc2_topk4_kernel: BO: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

S=4096:
- fused median_ms=62.675941
- nofused median_ms=103.867485
- faster_variant=fused
- fused selected kernels (6):
  - _count_routes_kernel | hash=FBW4UTKQWXCQGP45HIWQCXKYSMWXNU62ESJBQAPPB6N2IOCGI5ZA | meta={"num_warps": 1}
  - _prefix_prepare_kernel | hash=V3OKPK3G435DLW3DMNSA2UFFV6PVJ7MQPSEHMIKFGLVRDV3NMCHQ | meta={"num_warps": 1}
  - _scatter_routes_kernel | hash=26WTMJA6HMCLYB4EGZ6JSNW5TMLQMJX3WSVPI23RZ5GMFJA72PAA | meta={"num_warps": 1}
  - _fill_group_expert_kernel | hash=NR3HA7WR25OXWX4AHDW6RINCXQCUA4B2TZWHDL47A57MSLUEMS5A | meta={"num_warps": 1}
  - _fc1_silu_grouped_kernel | hash=U6QDCKRLG3R6WREWKTWIV5WPJLB26YU5Z3A5V7FQR55RZBXN6GJA | meta={"num_warps": 8}
  - _fc2_reduce_tiled_kernel | hash=R3OPQ3BGAQL4WIDW2OZJSIRYLBDHWMQKJQXH26LJAYL6TF7LPZGQ | meta={"num_warps": 4, "DB_TILE": 1}
- fused autotune winners:
  - _fc1_silu_grouped_kernel: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _fc2_reduce_tiled_kernel: DB_TILE: 1, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
- nofused selected kernels (2):
  - _moe_fc1_topk4_kernel | hash=YC3V2GL2KOBR54QKTFN7AV5CEHUXMFPNMO6VLY5DFVXYFPWWQDNQ | meta={"num_warps": 4, "BI": 32}
  - _moe_fc2_topk4_kernel | hash=RDPPSDUGCBL2CYJLWQ2LW5HRWLQPE5DFUB4435WOCWPQXQDSD7AQ | meta={"num_warps": 8, "BO": 128}
- nofused autotune winners:
  - _moe_fc1_topk4_kernel: BI: 32, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _moe_fc2_topk4_kernel: BO: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

S=8192:
- fused median_ms=124.573997
- nofused median_ms=210.806808
- faster_variant=fused
- fused selected kernels (6):
  - _count_routes_kernel | hash=FBW4UTKQWXCQGP45HIWQCXKYSMWXNU62ESJBQAPPB6N2IOCGI5ZA | meta={"num_warps": 1}
  - _prefix_prepare_kernel | hash=V3OKPK3G435DLW3DMNSA2UFFV6PVJ7MQPSEHMIKFGLVRDV3NMCHQ | meta={"num_warps": 1}
  - _scatter_routes_kernel | hash=26WTMJA6HMCLYB4EGZ6JSNW5TMLQMJX3WSVPI23RZ5GMFJA72PAA | meta={"num_warps": 1}
  - _fill_group_expert_kernel | hash=NR3HA7WR25OXWX4AHDW6RINCXQCUA4B2TZWHDL47A57MSLUEMS5A | meta={"num_warps": 1}
  - _fc1_silu_grouped_kernel | hash=U6QDCKRLG3R6WREWKTWIV5WPJLB26YU5Z3A5V7FQR55RZBXN6GJA | meta={"num_warps": 8}
  - _fc2_reduce_tiled_kernel | hash=R3OPQ3BGAQL4WIDW2OZJSIRYLBDHWMQKJQXH26LJAYL6TF7LPZGQ | meta={"num_warps": 4, "DB_TILE": 1}
- fused autotune winners:
  - _fc1_silu_grouped_kernel: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _fc2_reduce_tiled_kernel: DB_TILE: 1, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
- nofused selected kernels (2):
  - _moe_fc1_topk4_kernel | hash=YC3V2GL2KOBR54QKTFN7AV5CEHUXMFPNMO6VLY5DFVXYFPWWQDNQ | meta={"num_warps": 4, "BI": 32}
  - _moe_fc2_topk4_kernel | hash=RDPPSDUGCBL2CYJLWQ2LW5HRWLQPE5DFUB4435WOCWPQXQDSD7AQ | meta={"num_warps": 8, "BO": 128}
- nofused autotune winners:
  - _moe_fc1_topk4_kernel: BI: 32, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
  - _moe_fc2_topk4_kernel: BO: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

S=16384:
- fused median_ms=251.521027
- nofused median_ms=424.832977
- faster_variant=fused
- fused selected kernels (6):
  - _count_routes_kernel | hash=FBW4UTKQWXCQGP45HIWQCXKYSMWXNU62ESJBQAPPB6N2IOCGI5ZA | meta={"num_warps": 1}
  - _prefix_prepare_kernel | hash=V3OKPK3G435DLW3DMNSA2UFFV6PVJ7MQPSEHMIKFGLVRDV3NMCHQ | meta={"num_warps": 1}
  - _scatter_routes_kernel | hash=26WTMJA6HMCLYB4EGZ6JSNW5TMLQMJX3WSVPI23RZ5GMFJA72PAA | meta={"num_warps": 1}
  - _fill_group_expert_kernel | hash=NR3HA7WR25OXWX4AHDW6RINCXQCUA4B2TZWHDL47A57MSLUEMS5A | meta={"num_warps": 1}
  - _fc1_silu_grouped_kernel | hash=U6QDCKRLG3R6WREWKTWIV5WPJLB26YU5Z3A5V7FQR55RZBXN6GJA | meta={"num_warps": 8}
  - _fc2_reduce_tiled_kernel | hash=R3OPQ3BGAQL4WIDW2OZJSIRYLBDHWMQKJQXH26LJAYL6TF7LPZGQ | meta={"num_warps": 4, "DB_TILE": 1}
- fused autotune winners:
  - _fc1_silu_grouped_kernel: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
  - _fc2_reduce_tiled_kernel: DB_TILE: 1, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
- nofused selected kernels (2):
  - _moe_fc1_topk4_kernel | hash=YC3V2GL2KOBR54QKTFN7AV5CEHUXMFPNMO6VLY5DFVXYFPWWQDNQ | meta={"num_warps": 4, "BI": 32}
  - _moe_fc2_topk4_kernel | hash=RDPPSDUGCBL2CYJLWQ2LW5HRWLQPE5DFUB4435WOCWPQXQDSD7AQ | meta={"num_warps": 8, "BO": 128}
- nofused autotune winners:
  - _moe_fc1_topk4_kernel: BI: 32, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
  - _moe_fc2_topk4_kernel: BO: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

