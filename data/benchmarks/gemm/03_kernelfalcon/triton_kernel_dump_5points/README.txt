Triton Kernel Dump for 03_kernelfalcon (5 datapoints, winner-only)

Environment used:
- HIP_VISIBLE_DEVICES=6
- TRITON_KERNEL_DUMP=1
- TRITON_DUMP_DIR=<per-shape>/dump
- TRITON_CACHE_DIR=<per-shape>/cache (removed after dump)
- dtype: torch.bfloat16

Datapoints (M,N,K):
- (1024,1024,1024)
- (2048,2048,2048)
- (4096,4096,4096)
- (8192,8192,8192)
- (16384,16384,16384)

Each shape directory now keeps only the winner kernel's Triton dump artifacts:
- *.ttir
- *.ttgir
- *.llir
- *.amdgcn
- *.hsaco

Convenience layout:
- Each shape also has `amdgcn_flat/` with one flattened winner assembly file.

Winner mapping (M=N=K):
- 1024: LM6HN7RVC5Y76GAQO3XFUHJ5V3IGMBG5AI5DGCBUPPKYA5NKCO4A
- 2048: GXKVKY5WVKYR5NNRBGUWCGQZNXK3PBCXPXDNHB6FYGV3YX3OHA2A
- 4096: WFVZ7LDE6S4RV4JQYPB3PBKH6LV33ML34OK7F53IYA3HKNCQM55A
- 8192: ZAAYXU6NDWOGHQH4KVEJIUT32KBQI5CYFDSPYZYTEVC3GF7EWKSA
- 16384: ZAAYXU6NDWOGHQH4KVEJIUT32KBQI5CYFDSPYZYTEVC3GF7EWKSA
