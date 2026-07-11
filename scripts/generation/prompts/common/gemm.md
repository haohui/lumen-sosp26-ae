[HARD CONSTRAINTS: GEMM_ABT-NOHACK, MUST SATISFY]
1) Required semantics are EXACTLY C = A @ B^T, with A shape (M, K), B shape (N, K), C shape (M, N).
2) In custom kernel indexing, B must be accessed as B[n, k] (or an equivalent linearized form).
   Do NOT compute A @ B and do NOT reinterpret B as shape (K, N).
3) Forbidden in runtime main path (Python + C++/HIP):
   torch.matmul/mm/bmm/einsum/addmm, F.linear, at::matmul/mm/bmm/einsum/addmm/linear.
4) No try/except fallback and no conditional branch that routes to eager torch reference compute.
5) Keep BF16 task semantics: for BF16 inputs, output tensor dtype must remain BF16.
6) Do not call external GEMM libraries as the main compute path (e.g., hipBLAS/rocBLAS GEMM wrappers).
7) Kernel launches must be graph-capture friendly: launch on the current PyTorch/HIP stream
   used by the caller, not on an unrelated default/new stream, and do not add device-wide
   synchronization on the main path.
8) Append ANTI_HACK_MANIFEST (as comments) with:
   - target_semantics: A@B^T
   - forbidden_api_used: []
   - fallback_path: false
