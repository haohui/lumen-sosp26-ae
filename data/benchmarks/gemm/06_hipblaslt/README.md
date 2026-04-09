# 06_hipblaslt

HipBlasLt baseline assets for GEMM (ABt), organized inside `opt_kernel`.

## What Is In This Folder

- [src/hipblaslt_internal_ext.cpp](./src/hipblaslt_internal_ext.cpp)
  - PyTorch C++ extension source used by unified timing script.
- [runtime_libs/](./runtime_libs)
  - Vendored ROCm runtime libs for HipBlasLt path:
    - `libhipblaslt.so*`
    - `libhipblas.so*`
    - `librocblas.so*`
    - `libamdhip64.so*`
- [manifests/runtime_libs.sha256](./manifests/runtime_libs.sha256)
  - Checksums for vendored runtime libraries.
- `20260313_hipblaslt_internal_algo_query.log` (archived outside this package)
  - Historical algorithm selection log.

## Workload To Kernel Mapping

Source of truth: `20260313_hipblaslt_internal_algo_query.log` (archived outside this package)

Canonical 5 GEMM workloads (M=N=K):

| Workload | algo_index | Kernel family (full string in log) | Kernel record path | Assembly result path |
|---|---:|---|---|---|
| 1024 | 165229 | `...BBS_BH_UserArgs_MT128x128x64_MI32x32x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 1 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |
| 2048 | 165229 | `...BBS_BH_UserArgs_MT128x128x64_MI32x32x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 2 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |
| 4096 | 165231 | `...BBS_BH_UserArgs_MT256x224x64_MI16x16x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 3 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |
| 8192 | 165231 | `...BBS_BH_UserArgs_MT256x224x64_MI16x16x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 4 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |
| 16384 | 152215 | `...BBS_BH_Bias_HA_S_SAV_UserArgs_MT512x160x32_MI16x16x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 7 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |

Additional workloads recorded in the same log:

| Workload | algo_index | Kernel family (full string in log) | Kernel record path | Assembly result path |
|---|---:|---|---|---|
| 9216 | 151992 | `...BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 5 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |
| 14592 | 152035 | `...BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x304x32_MI16x16x1...WG64_4_1` | `20260313_hipblaslt_internal_algo_query.log` line 6 | Not dumped as standalone `.amdgcn` in this folder (runtime kernel is internal in `runtime_libs/`) |

Runtime library paths that contain the executable code objects:

- [runtime_libs/libhipblaslt.so.0.12.60403](./runtime_libs/libhipblaslt.so.0.12.60403)
- [runtime_libs/librocblas.so.4.4.60403](./runtime_libs/librocblas.so.4.4.60403)

## Runtime Preference

`opt_kernel/perf_script/benchmark_unified_graph.py` now prefers:

1. `gemm-openai/06_hipblaslt/src/hipblaslt_internal_ext.cpp`
2. `gemm-openai/06_hipblaslt/runtime_libs` for link/runtime search
3. fallback ROCm system dirs (`/opt/rocm/lib`, `/opt/rocm-6.4.3/lib`)
