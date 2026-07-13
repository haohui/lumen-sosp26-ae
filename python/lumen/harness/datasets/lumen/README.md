# Lumen kernel optimization

Run an optimization sequence from the repository root with a ROCm GPU and a
configured `codex` executable available in `PATH`:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_optimize DOMAIN --gpu-id GPU_ID
```

`DOMAIN` is `gemm`, `attn`, or `moe`. The command exposes the selected physical
GPU through `HIP_VISIBLE_DEVICES`, runs Codex once per prompt, and checks every
candidate for correctness before starting the next round.

The built-in sequences are:

| Domain | Starting kernel | Prompts |
| --- | --- | --- |
| `gemm` | `datasets/inference/gemm/lumen/gemm-naive.py` | optimization-01 through optimization-12 |
| `attn` | `datasets/inference/attention/lumen/attn_01_naive.py` | optimization-02 through optimization-06 |
| `moe` | `datasets/inference/moe/lumen/moe_01_baseline.py` | optimization-02 through optimization-07 |

To resume an interrupted run from its latest successful round:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_optimize DOMAIN \
  --resume-run runs/lumen_DOMAIN_codex_YYYYMMDD_HHMMSS_ffffff \
  --gpu-id GPU_ID
```

The resume command reads the original prompt sequence from `run_config.json`;
no prompt needs to be repeated on the command line. A failed or incomplete
round and any later round directories are replaced.

Use `--kernel` to replace the starting kernel or repeat `--prompt-file` to run
a custom prompt sequence. `--prepare-only` creates a round workspace without
starting Codex.

Runs are written to `runs/lumen_<domain>_codex_<timestamp>/`. Each `roundN/`
contains `input_model.py`, `output_model_new.py`, `prompt.txt`,
`round_status.json`, `eval_result.json`, `codex_result.json`, and
`trace.jsonl` when a Codex session trace is available.
