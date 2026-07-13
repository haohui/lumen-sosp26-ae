# Lumen Attention optimization

Run the complete Attention optimization sequence from the repository root:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_attn_optimize \
  --kernel datasets/inference/attention/lumen/attn_01_naive.py \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-02.md \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-03.md \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-04.md \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-05.md \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-06.md \
  --gpu-id 6
```

Resume a completed run by appending more rounds:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_attn_optimize \
  --resume-run runs/lumen_attn_codex_YYYYMMDD_HHMMSS_ffffff \
  --prompt-file python/lumen/harness/datasets/lumen/attn/prompts/optimization-06.md \
  --gpu-id 6
```
