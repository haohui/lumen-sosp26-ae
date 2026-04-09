# 05_aiter Artifact Pack

This directory stores extracted AITER backend artifacts for DeepSeek blockscale MoE verification.

## Scope

- ASM backend artifacts
- CK 2-stage backend artifacts
- AITER Triton backend artifacts

## Number -> Kernel Mapping (AITER Rows)

Source report:

- [../../../logs/opt_kernel_timing/moe-openai/reports/bench_aiter_backends_cudagraph_20260330_020806.csv](../../../logs/opt_kernel_timing/moe-openai/reports/bench_aiter_backends_cudagraph_20260330_020806.csv)
- [../../../logs/opt_kernel_timing/moe-openai/reports/bench_aiter_backends_cudagraph_20260330_020806.json](../../../logs/opt_kernel_timing/moe-openai/reports/bench_aiter_backends_cudagraph_20260330_020806.json)

| Row label | 1024 | 2048 | 4096 | 8192 | 16384 | Runtime kernel binary |
|---|---:|---:|---:|---:|---:|---|
| AITER-ck | 1.074240 | 1.819072 | 3.332276 | 6.388307 | 12.591383 | [CK/bin/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so](./CK/bin/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so), [CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so](./CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so) |
| AITER-triton | 1.600775 | 2.711695 | 4.832568 | 8.784945 | 16.886300 | [Triton/bin/_moe_fc1_topk4_kernel.hsaco](./Triton/bin/_moe_fc1_topk4_kernel.hsaco), [Triton/bin/_moe_fc2_topk4_kernel.hsaco](./Triton/bin/_moe_fc2_topk4_kernel.hsaco) |
| AITER-asm | 0.866280 | 1.549805 | 2.732016 | 5.267972 | 10.169046 | [ASM/bin/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co](./ASM/bin/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co), [ASM/bin/module_moe_asm.so](./ASM/bin/module_moe_asm.so) |

## Layout

- `ASM/`
  - `src/`: source files used by `module_moe_asm`
  - `bin/`: runtime binaries (`module_moe_asm.so`, selected `.co`)
  - `disasm/`: symbol tables and disassembly outputs
- `CK/`
  - `src/`: CK pybind + generated instance sources (`blob/instances/*.cu`)
  - `bin/`: selected CK module `.so` files
  - `disasm/`: exports/readelf/objdump outputs
- `Triton/`
  - `src/`: AITER Triton Python kernel sources
  - `cache/`: copied Triton cache artifacts (`.source/.ttir/.ttgir/.llir/.amdgcn/.hsaco/.json`)
  - `bin/`: selected topk4 `.hsaco` binaries
  - `disasm/`: symbol tables and disassembly outputs

## Selected Triton Cache Dirs

- fc1 topk4: `/data01/home/daifeng/.triton/cache/BMQRTS4HHO5MWWHGYTGWNZ4EVIACRZOXMDQLJ7IHIMX6PFXXMN3Q`
- fc2 topk4: `/data01/home/daifeng/.triton/cache/KEBQRGDZ7MFTNL23HGJ4JEMICRG5EUMW7HBXHR3XOCJ5K3GLEI4Q`

## Triton Source Mapping (Why 4 Files)

The 4 Triton files are split into two layers:

- API/launch layer
  - `Triton/src/moe_op.py`
  - `Triton/src/moe_op_silu_fused.py`
- JIT kernel definition layer
  - `Triton/src/moe_op_kernel.py`
  - `Triton/src/moe_op_silu_fused_kernel.py`

Current `test_moe_blockscale.py` Triton path uses both stages:

- Stage1 (FC1 + SiLU): `moe_op_silu_fused.py` -> `moe_op_silu_fused_kernel.py`
- Stage2 (FC2/reduce): `moe_op.py` -> `moe_op_kernel.py`

So all 4 files are part of one runtime path; GPU execution runs kernels from the `*_kernel.py` layer.

## Runtime Binary / Disassembly Mapping

This section answers: "which binary is actually run, and which file is its disassembly?"

### ASM (AITER)

- Python entry (test path): `aiter.fmoe_fp8_blockscale_g1u1(...)`
- Runtime module (`.so`):
  - [ASM/bin/module_moe_asm.so](./ASM/bin/module_moe_asm.so)
- Kernel code object (`.co`) used for this DeepSeek blockscale path:
  - [ASM/bin/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co](./ASM/bin/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co)
- Disassembly / symbols:
  - [ASM/disasm/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co.objdump.S](./ASM/disasm/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co.objdump.S)
  - [ASM/disasm/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co.symbols.txt](./ASM/disasm/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co.symbols.txt)
  - [ASM/disasm/module_moe_asm.so.exports.txt](./ASM/disasm/module_moe_asm.so.exports.txt)
  - [ASM/disasm/module_moe_asm.so.readelf.txt](./ASM/disasm/module_moe_asm.so.readelf.txt)

### CK (AITER 2-stage)

- Python entries (test path):
  - `aiter.ck_moe_stage1_fwd(...)`
  - `aiter.ck_moe_stage2_fwd(...)`
- Selected runtime module family (`.so`):
  - [CK/bin/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so](./CK/bin/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so)
  - [CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so](./CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so)
  - [CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so](./CK/bin/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so)
- How module is chosen:
  - `preshuffle_on/off` depends on `getattr(w1, "is_shuffled", False)`.
  - In `test_moe_blockscale.py`, weights come from `shuffle_weight(...)`, which sets `is_shuffled=True`, so the typical path is `...preshuffle_on...`.
- Disassembly / symbols:
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.objdump.S](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.objdump.S)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.exports.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.exports.txt)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.readelf.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so.readelf.txt)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.objdump.S](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.objdump.S)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.exports.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.exports.txt)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.readelf.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage2.so.readelf.txt)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.objdump.S](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.objdump.S)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.exports.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.exports.txt)
  - [CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.readelf.txt](./CK/disasm/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_1x128_mulWeightStage1.so.readelf.txt)

### Triton (AITER)

- Python entries (test path):
  - `triton_moe_silu(...)` from `Triton/src/moe_op_silu_fused.py`
  - `triton_moe(...)` from `Triton/src/moe_op.py`
- Runtime binary form:
  - Triton does not use a fixed `.so` for these kernels; it JIT-compiles to `.hsaco` cache objects.
- Cached binaries included in this pack:
  - [Triton/bin/_moe_fc1_topk4_kernel.hsaco](./Triton/bin/_moe_fc1_topk4_kernel.hsaco)
  - [Triton/bin/_moe_fc2_topk4_kernel.hsaco](./Triton/bin/_moe_fc2_topk4_kernel.hsaco)
- Corresponding disassembly / symbols:
  - [Triton/disasm/_moe_fc1_topk4_kernel.hsaco.objdump.S](./Triton/disasm/_moe_fc1_topk4_kernel.hsaco.objdump.S)
  - [Triton/disasm/_moe_fc1_topk4_kernel.hsaco.symbols.txt](./Triton/disasm/_moe_fc1_topk4_kernel.hsaco.symbols.txt)
  - [Triton/disasm/_moe_fc2_topk4_kernel.hsaco.objdump.S](./Triton/disasm/_moe_fc2_topk4_kernel.hsaco.objdump.S)
  - [Triton/disasm/_moe_fc2_topk4_kernel.hsaco.symbols.txt](./Triton/disasm/_moe_fc2_topk4_kernel.hsaco.symbols.txt)
- Source metadata of these two cached kernels points to:
  - `trace_runs/.../03_kernelfalcon.../best_kernel.py`
  - See:
    - [Triton/cache/BMQRTS4HHO5MWWHGYTGWNZ4EVIACRZOXMDQLJ7IHIMX6PFXXMN3Q/_moe_fc1_topk4_kernel.source](./Triton/cache/BMQRTS4HHO5MWWHGYTGWNZ4EVIACRZOXMDQLJ7IHIMX6PFXXMN3Q/_moe_fc1_topk4_kernel.source)
    - [Triton/cache/KEBQRGDZ7MFTNL23HGJ4JEMICRG5EUMW7HBXHR3XOCJ5K3GLEI4Q/_moe_fc2_topk4_kernel.source](./Triton/cache/KEBQRGDZ7MFTNL23HGJ4JEMICRG5EUMW7HBXHR3XOCJ5K3GLEI4Q/_moe_fc2_topk4_kernel.source)

## Notes

- This pack is for inspection/replay/documentation, not source-of-truth build logic.
- Per-backend provenance is also recorded in:
  - `ASM/MANIFEST.txt`
  - `CK/MANIFEST.txt`
  - `Triton/MANIFEST.txt`
