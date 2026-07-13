# Optimization 06: add persistent route-group execution

Optimize the existing AveLang fused FP8 MoE kernel by adding an optional
persistent thread-group mode. The input already contains both software
pipelines, invalid-route `v_setvskip` masking, and packed arithmetic; preserve
all of them.

Normally grid dimension 1 launches one thread group per route group. Add a
`num_persistent_tgs` launcher option that can cap the number of resident thread
groups. In persistent mode, each launched group must process a strided sequence
of route groups until all valid route groups have been covered.

## Required transformation

### Launcher and public API

- Add `num_persistent_tgs: int = 0` immediately before `out` in both the public
  `fused_moe_fp8_blockscale_g1u1` function and its implementation helper.
  Forward it through unchanged. The default value must preserve non-persistent
  behavior and existing callers.
- Compute `split_k = ceil_div(inter_dim, GROUP_DIM)`.
- Begin with the existing route-group grid size. When
  `num_persistent_tgs > 0`, compute the allowed route-group count as
  `ceil_div(num_persistent_tgs, split_k)`, clamp the launched route-group grid
  to at least one and at most the normal route-group count, and pass that
  launched count to the device kernel as `persistent_route_step`.
- When persistence is disabled, pass a zero step and retain the original grid.

### Device route loop

- Add `persistent_route_step` to the device kernel arguments.
- Treat `block_id(1)` as `route_group_begin`. In normal mode, execute exactly
  that one route group. In persistent mode, iterate from `route_group_begin`
  to `num_valid_m_blocks` with stride `persistent_route_step`.
- Derive a finite `route_group_iters` count before the loop. Keep guards for
  both `route_group < num_valid_m_blocks` and
  `route_base < num_valid_ids`; speculative grid entries must remain safe.
- Use a single lexical `for route_group_iter in S.range(route_group_iters)`
  around the route body. Do not extract the body into a separately JIT-compiled
  helper, use a `while`, or hard-code a maximum iteration count; those forms
  change private/LDS lifetime and are not equivalent to this optimization.
- Move every route-specific operation inside this loop: expert/token metadata
  loads, invalid-token mask, sorted route weights, expert-relative W1/W3/W2
  offsets, Stage 1, quantization, and Stage 2.
- Reuse the workgroup's shared-memory allocation across iterations. Add a full
  workgroup barrier at the end of each route iteration so no iteration reuses
  LDS while another wave is still finishing the previous route.

### Repeated metadata access

- Create reusable AMD buffer resources for `sorted_token_ids` and
  `sorted_expert_ids` outside the persistent loop, with a sufficiently large
  resource range just like the weight resources.
- Inside the loop, load `expert_id`, the two token selectors, and all eight
  local token ids using `S.amdgpu.raw_buffer_load_x1` with byte offsets derived
  from the current `route_base`. Mask token ids with `0x00FFFFFF` exactly as in
  the input.
- The load operand order is critical. Create `zero = S.convert(0, S.u32)` and
  call each metadata load as
  `S.amdgpu.raw_buffer_load_x1(resource, zero, byte_offset, 0)`. The second
  operand is the zero vector index and the third operand is the varying byte
  offset. Do not pass `byte_offset` as the second operand.
- Keep sorted-weight loading and the invalid-token-mask bit ordering unchanged.

The optimization is complete only if one launched workgroup can execute more
than one route group when persistence is enabled. Merely reducing the launch
grid without adding the strided device loop is incorrect.

## Correctness invariants

- `num_persistent_tgs=0` must preserve the input behavior exactly.
- Persistent and non-persistent modes must cover every valid route group
  exactly once for arbitrary valid route counts and must produce equivalent
  output.
- Preserve routed fused-MoE semantics, FP8 scaling/quantization, route weights,
  BF16 atomics, both pipelines, packed arithmetic, `v_setvskip`, MFMA mapping,
  and workgroup size.
- Keep all metadata loads and route-specific weight offsets in bounds.

## Scope boundary for this round

This round is only persistent route-group execution. Do not add explicit
instruction scheduling or `sched_group_barrier`/`sched_barrier` calls.

The final implementation must pass the normal correctness benchmark for token
counts 1024, 2048, 4096, 8192, and 16384. Also exercise at least one nonzero
`num_persistent_tgs` value and confirm it matches non-persistent output. Do not
add fallback compute paths.
