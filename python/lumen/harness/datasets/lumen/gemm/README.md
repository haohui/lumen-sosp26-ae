# Lumen GEMM optimization

Run one or more GEMM optimization rounds from the repository root:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_gemm_optimize \
  --kernel datasets/inference/gemm/lumen/gemm-naive.py \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-01.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-02.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-03.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-04.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-05.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-06.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-07.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-08.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-09.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-10.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-11.md \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-12.md \
  --gpu-id 6
```

Repeat `--prompt-file` to run multiple prompts sequentially.

Resume a completed run by appending another round:

```bash
PYTHONPATH=python python -m lumen.tools.cli.lumen_gemm_optimize \
  --resume-run runs/lumen_gemm_codex_YYYYMMDD_HHMMSS_ffffff \
  --prompt-file python/lumen/harness/datasets/lumen/gemm/prompts/optimization-02.md \
  --gpu-id 6
```
