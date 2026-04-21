# Harness Tools

## KernelBench Batch Runner

Use `run_kernelbench_case.py` to evaluate imported KernelBench cases in batch.

All paths below are relative to the repository root `lumen-sosp26-ae/`.

The input file must contain one case directory per line. Empty lines and lines starting with `#` are ignored.

Example input file:

```text
data/invariant/kb_eval_pipeline/runs/kb1/p01/round0
data/invariant/kb_eval_pipeline/runs/kb1/p05/round0
```

Run:

```bash
python3 data/invariant/kb_eval_pipeline/harness/tools/run_kernelbench_case.py \
  --input-file /tmp/kernelbench_cases.txt \
  --output /tmp/kernelbench_results.jsonl \
  --num-correct-trials 1 \
  --num-perf-trials 10 \
  --measure-performance
```

Notes:

- Output is JSONL, one result object per input case.
- The runner continues after per-case failures and records exception details in the corresponding JSON object.
- By default, performance timing uses `cudagraph`.
- Evaluator logs are written to stderr; the JSONL file contains only structured results.
