[HARD CONSTRAINTS: ATTENTION-NOHACK, MUST SATISFY]
1) Keep exact semantics and layout [B, S, H, D] for inputs Q/K/V and output.
1.1) Forbidden external interface: do NOT define external I/O layout as [B, H, S, D].
2) Runtime main compute path MUST be custom HIP/Triton kernel compute.
3) Forbidden in runtime main path (Python + C++):
   torch.nn.functional.scaled_dot_product_attention, torch.scaled_dot_product_attention,
   torch.einsum/matmul/mm/bmm, torch.softmax/logsumexp,
   at::scaled_dot_product_attention, at::matmul/mm/bmm/einsum,
   flash_attn*, xformers*.
4) No try/except fallback and no branch that routes to eager/reference torch compute.
5) BF16 task: output tensor dtype must remain BF16 (no final output cast to fp32).
6) If constraints cannot be satisfied, output exactly: UNSAT_ATTN_NOHACK
7) Append ANTI_HACK_MANIFEST with:
   - forbidden_api_used: []
   - main_compute_kernels: [kernel names]
   - fallback_path: false
8) The output code MUST include at least one explicit HIP kernel declaration:
   "__global__ void ...", and must invoke it through a load_inline-exported op
   on the forward main path (no placeholder/pseudocode).
9) Do NOT include CUDA-only headers/APIs in generated C++/HIP source
   (forbidden examples: ATen/cuda/*, cuda_runtime_api.h, cudaStream_t).
   Use HIP headers/APIs only (e.g., hip/hip_runtime.h) plus ATen common headers.
10) Performance hard constraints for this task (S in {1024, 2048, 4096, 8192, 16384}):
   - Do NOT generate a degenerate serial kernel (e.g., one thread computes one full [SxS] row/column).
   - Main attention kernels must use cooperative parallelism (multiple warps / wavefronts per block).
   - Do not use launch configs equivalent to single-thread execution on main compute kernels
     (forbidden patterns include threads=1, blockDim.x=1 for the main path).
11) Kernel launches must be graph-capture friendly: launch on the current PyTorch/HIP stream
   used by the caller, not on an unrelated default/new stream, and do not add device-wide
   synchronization on the main path.
12) Append PERF_MANIFEST with:
   - launch_config: {grid, block, num_warps}
   - tile_sizes: {BLOCK_M, BLOCK_N, BLOCK_K or equivalent}
   - expected_parallelism: short explanation
