# Attention d128 KSearch (Organized)

This directory is organized to mirror `opt_kernel/gemm-openai/04_ksearch` and exposes the nohack r8 result as canonical best.

## Canonical best (nohack)

- `best_solution.json`
- `best_kernel.cu`
- `best_kernel.h`
- `best_binding.cpp`
- `best_kernel.py`
- `best_eval_report.json`

The selected best round is `round8` (`round_008`) and corresponds to:

`KSearch 8.312745 41.183276 160.986890 645.655625 3760.498828`

with settings: `bf16, GQA, hdim=128, qheads=8, bs=16, kvheads=1`.

## Full output / rounds

- `trace_input_output/round1..round10`: prompt input/output + kernel files per round.
- `output/rounds_20260323_061108_per_workload/`: full per-workload K-Search artifacts.
- `output/rounds_20260323_061108_per_workload/MANIFEST.tsv`: index of workload -> rounds path -> round count.

## Full traffic

- `traffic/per_workload/wl*_seq*_bs16_kv1/`: per-workload MITM traffic captures.
- `traffic/legacy/`: legacy capture files from historical subtree.
