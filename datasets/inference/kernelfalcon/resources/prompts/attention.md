[KERNELFALCON ATTENTION HARD CONSTRAINTS]
- In Triton kernels, task-static dimensions and head counts must be compile-time
  constants (`tl.constexpr`) when used for shape/head mapping, including
  num_q_heads, num_kv_heads, and head_dim.
