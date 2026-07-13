# Lumen MoE optimization

Run the complete MoE optimization sequence from the repository root:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_moe_optimize \
  --kernel datasets/inference/moe/lumen/moe_01_baseline.py \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-02.md \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-03.md \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-04.md \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-05.md \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-06.md \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-07.md \
  --gpu-id 6
```

Resume a completed run by appending more rounds:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_moe_optimize \
  --resume-run runs/lumen_moe_codex_YYYYMMDD_HHMMSS_ffffff \
  --prompt-file python/lumen/harness/datasets/lumen/moe/prompts/optimization-07.md \
  --gpu-id 6
```
