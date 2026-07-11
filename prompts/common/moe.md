[HARD CONSTRAINTS: FUSED-NOHACK, MUST SATISFY]
1) Runtime main path MUST be custom HIP/Triton kernels for dispatch+expert_compute+combine.
2) Forbidden in runtime main path (Python + C++):
   torch.matmul/mm/bmm/einsum/addmm, F.linear, index_add_/scatter_add_,
   at::matmul/mm/bmm/einsum/addmm/linear, at::silu.
3) load_inline is allowed ONLY as compilation wrapper; compute must be inside custom kernels.
4) No per-expert eager matmul loop. Expert loop is allowed only for kernel launch bookkeeping, not ATen compute.
5) If any forbidden API is required, output exactly: UNSAT_FUSED_NOHACK
6) Kernel launches must be graph-capture friendly: launch on the current PyTorch/HIP stream
   used by the caller, not on an unrelated default/new stream, and do not add device-wide
   synchronization on the main path.
7) Output must include ANTI_HACK_MANIFEST:
   - forbidden_api_used: []
   - main_compute_kernels: [kernel names]
   - single_runtime_op: true
