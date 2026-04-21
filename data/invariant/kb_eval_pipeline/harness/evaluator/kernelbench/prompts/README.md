# KernelBench prompt examples

This folder contains PyTorch modules paired with CUDA kernels. The evaluator
uses them as in-context examples when constructing KernelBench prompts.

## Acknowledgements

- Fused GeLU and tiled matmul examples are adapted from Christian Mills,
  GPU MODE Lecture 04.
- The minimal flash attention example is adapted from Peter Kim's Minimal
  Flash Attention implementation.
