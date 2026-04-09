# 05_HipKittens

Attention benchmark results for the HipKittens repository path (Triton attention, not ksearch).

Legacy artifacts have been moved to: `log/legacy_20260324_keep_actual_only/`.

## Benchmark entry

- Script: `benchmark_hipkittens_attention_triton_v02.py`
- Baseline source: `third_party/HipKittens/analysis/baselines/attn/triton_baseline_v02.py`
- Workload: `dtype=bf16, batch_size=16, num_q_heads=8, num_kv_heads=1, head_dim=128, causal=True, seq_lens=[1024, 2048, 4096, 8192, 16384]`
- Isolation used in run: `HIP_VISIBLE_DEVICES=7`, `CPU affinity=120-127`

## Average latency (mean_ms)

Source: `meta/mean_latency_20260324.tsv`

- `1024 -> 0.1947933455631024 ms`
- `2048 -> 0.5974703222136457 ms`
- `4096 -> 1.9455306224333933 ms`
- `8192 -> 7.030844907407408 ms`
- `16384 -> 25.900626627604165 ms`

## Runtime-selected fallback kernel (actual executed)

Source (runtime probe): `meta/triton_runtime_pick_probe_20260324.log`

- Selected config for all seq_lens:
  - `BLOCK_M=128, BLOCK_N=64, waves_per_eu=2, PRE_LOAD_V=False, GRID_CU_MULTIP=2, num_warps=4, num_stages=1`

Source (seq_len -> actual hsaco mapping):
- `meta/triton_fallback_actual_kernel_map_hostpath_20260324.tsv`

Actual dump directory (only selected kernels, not full cache):
- `asm/triton_fallback_actual_20260324/`
- Each seq_len directory contains:
  - `attn_fwd.hsaco`
  - `attn_fwd.amdgcn`
  - `attn_fwd.source`
  - `attn_fwd.ttir`
  - `attn_fwd.ttgir`
  - `attn_fwd.llir`
  - `attn_fwd.disasm.s`
  - `attn_fwd.readelf.txt`

Selected kernel hashes:

| seq_len | hash |
|---|---|
| 1024 | `5871e7d4fc7e26abaaced1b869ba27585344cb78c12c411085b59ee6049993dd` |
| 2048 | `4246a058f7ce0fbaf2b1df217f1a69ce011c862fc88d7b81a7530f811fc1d70f` |
| 4096 | `07446caf4282f3edad2226777596fa38be71eb6e975201e924c90eedb7582513` |
| 8192 | `0fe20d1056501fdae293710dfbfd699288d18a23063dd0e387b472ad334a6d83` |
| 16384 | `c934aa128d7b6edc7599e2a2ad74cf17944e63744c76ad214902d07e9a5919c8` |

## Kernel artifacts

- Cleanup policy applied: keep only actual executed kernels per workload.
- For Triton fallback workload, only the 5 selected kernels (seq_len 1024/2048/4096/8192/16384) are retained.
- Actual-selected file inventory: `meta/triton_fallback_actual_files_manifest_20260324.tsv`

## Assembly dumps

- Actual-selected fallback disassembly + ELF metadata:
  - `asm/triton_fallback_actual_20260324/seq_*/attn_fwd.disasm.s`
  - `asm/triton_fallback_actual_20260324/seq_*/attn_fwd.readelf.txt`
  - `meta/triton_fallback_actual_kernel_map_hostpath_20260324.tsv`

## Kernel identity and versions

Source: `meta/hipkittens_kernel_identity_20260324.txt`

- Kernel symbol: `attn_fwd`
- HSACO target: `amdgcn-amd-amdhsa--gfx942`
- Container: `kernel-benchmark-rocm-traffic`
- `torch=2.9.1+rocm7.1.1.git351ff442`
- `torch_hip=7.1.52802-26aae437f6`
- `triton=3.5.1+rocm7.1.1.gita272dfa8`
- Baseline source sha256:
  - `1eb8dc419e58848db15dfcb0f0cfcfb7aa604cb79d68b1523023590789bb57f3`
