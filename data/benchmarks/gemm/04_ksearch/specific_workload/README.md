# specific_workload (KSearch GEMM)

- Source run: `ksearch/K-Search/.ksearch-output-gemm-per-workload/20260322_094056`
- Contains copied traffic and per-workload best kernel artifacts.

## Layout

- `traffic/`: copied from `04_ksearch/traffic`
- `workloads/wlXX_m{M}_n{N}_k{K}_{uuid8}/`: per-workload best kernel and metadata

## Workloads

- wl01: M=N=K=1024, uuid=6d90d39b-0569-40fb-9c3d-a8596165eedd
- wl02: M=N=K=2048, uuid=5fcbad3d-10eb-46bf-8b5b-fd4f5856f8d5
- wl03: M=N=K=4096, uuid=c53ad9b2-f93f-494d-83d4-eceda3e9d776
- wl04: M=N=K=8192, uuid=65716dfd-3fab-42c2-b0a0-15b36b89c44c
- wl05: M=N=K=9216, uuid=f68a6e4a-7f65-4197-897a-a330e66f498d
