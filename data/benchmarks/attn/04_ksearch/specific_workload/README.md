# Attention KSearch Specific Workloads

- Params: bf16, KV=1, head_dim=128, num_q_heads=8, batch_size=16, causal=True
- Seq lens: 1024, 2048, 4096, 8192, 16384
- Timing: warmup 200ms, repeat >= 1s, GPU7, CPU 120-127
