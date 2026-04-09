# ROCprof + ASM Archive (CUDAForge MoE)

This folder stores the full ROCprof outputs and extracted GPU code object for the CUDAForge MoE baseline row:
- baseline: `opt_kernel/moe-openai/02_cudaforge/output/fused/best_kernel.py`
- run config: `seq_len=1024`, `HIP_VISIBLE_DEVICES=7`, `cpu_cores=120-127`

## Key files

- `rocprof_moe_cudaforge_1024_20260408_024717.stats.csv`: kernel time summary (calls/total/avg/%)
- `rocprof_moe_cudaforge_1024_20260408_024717.csv`: dispatch-level trace CSV
- `rocprof_moe_cudaforge_1024_20260408_024717.json`: ROCprof JSON export
- `rocprof_moe_cudaforge_1024_metrics_20260408_024943.csv`: PMCs (`OccupancyPercent`, `VALU*`, `MemUnitBusy`, `FetchSize`, `WriteSize`, ...)

## ASM / code object

- `hip_fatbin.bin`: `.hip_fatbin` section extracted from the runtime extension `.so`
- `TARGET_MAP.txt`: mapping of fatbin blobs to `amdhsa.target`
- `gfx942_blob2_10.elf`: extracted gfx942 code object (MI300X target)
- `gfx942_blob2_10.rocmllvm.objdump.S`: disassembly produced with `/opt/rocm-6.4.3/llvm/bin/llvm-objdump -d`

The symbol corresponding to the profiled kernel name is:
- `_ZN12_GLOBAL__N_121fused_moe_kernel_implIiEEvPKhS2_S2_PKfPKT_S4_S4_S4_Ptiiiii`

and appears in:
- `gfx942_blob2_10.rocmllvm.objdump.S`
