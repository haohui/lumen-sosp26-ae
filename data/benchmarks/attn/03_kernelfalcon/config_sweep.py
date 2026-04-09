import json
import os
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_causal_attention_cfg(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, HQ, HKV, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b = pid_bh // HQ
    hq = pid_bh % HQ
    hkv = tl.where(HKV == 1, 0, hq)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + b * stride_qb + hq * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    log2e = 1.4426950408889634
    qk_scale = sm_scale * log2e

    n_start = 0
    while n_start < S:
        offs_n = n_start + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + b * stride_kb + hkv * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + b * stride_vb + hkv * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k)) * qk_scale

        causal = offs_m[:, None] >= offs_n[None, :]
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        qk = tl.where(causal & valid, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        n_start += BLOCK_N

    out = acc / l_i[:, None]

    o_ptrs = o_ptr + b * stride_ob + hq * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=o_mask)


def bench_one_config(q, k, v, cfg, warmup, repeat, trials):
    B, HQ, S, D = q.shape
    HKV = k.shape[1]
    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    grid = (triton.cdiv(S, cfg["BLOCK_M"]), B * HQ)

    def launch():
        _fused_causal_attention_cfg[grid](
            q, k, v, o,
            B, HQ, HKV, S, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            sm_scale,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_D=cfg["BLOCK_D"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    launch()
    torch.cuda.synchronize()

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(repeat):
            launch()
        ed.record()
        torch.cuda.synchronize()
        samples.append(st.elapsed_time(ed) / repeat)

    return {
        **cfg,
        "mean_ms": float(statistics.mean(samples)),
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "samples_ms": [float(x) for x in samples],
    }


def schedule_for_seq(seq_len):
    if seq_len <= 2048:
        return {"warmup": 20, "repeat": 100, "trials": 5}
    if seq_len <= 4096:
        return {"warmup": 10, "repeat": 40, "trials": 4}
    if seq_len <= 8192:
        return {"warmup": 6, "repeat": 20, "trials": 4}
    return {"warmup": 3, "repeat": 8, "trials": 3}


if __name__ == "__main__":
    assert torch.cuda.is_available(), "ROCm device not available"
    device = torch.device("cuda")
    torch.manual_seed(0)

    B = 16
    HQ = 8
    HKV = 1
    D = 128
    seq_lens = [1024, 2048, 4096, 8192, 16384]

    configs = [
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128, "num_warps": 4, "num_stages": 1},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128, "num_warps": 4, "num_stages": 1},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 128, "num_warps": 8, "num_stages": 1},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 128, "num_warps": 8, "num_stages": 1},
    ]

    all_rows = []
    best_by_seq = {}

    for s in seq_lens:
        q = torch.randn((B, HQ, s, D), device=device, dtype=torch.bfloat16)
        k = torch.randn((B, HKV, s, D), device=device, dtype=torch.bfloat16)
        v = torch.randn((B, HKV, s, D), device=device, dtype=torch.bfloat16)

        sch = schedule_for_seq(s)
        print(f"\n=== S={s} warmup={sch['warmup']} repeat={sch['repeat']} trials={sch['trials']} ===", flush=True)
        seq_rows = []

        for cfg in configs:
            row = bench_one_config(q, k, v, cfg, sch["warmup"], sch["repeat"], sch["trials"])
            row["S"] = s
            row["B"] = B
            row["HQ"] = HQ
            row["HKV"] = HKV
            row["D"] = D
            seq_rows.append(row)
            all_rows.append(row)
            print(row, flush=True)

        best = min(seq_rows, key=lambda x: x["median_ms"])
        best_by_seq[str(s)] = best
        print(f"BEST_S={s}: {best}", flush=True)

    out = {
        "shape": {"B": B, "HQ": HQ, "HKV": HKV, "D": D, "causal": True},
        "seq_lens": seq_lens,
        "triton_kernel_dump": os.environ.get("TRITON_KERNEL_DUMP"),
        "triton_dump_dir": os.environ.get("TRITON_DUMP_DIR"),
        "rows": all_rows,
        "best_by_seq": best_by_seq,
    }

    out_path = "config_sweep_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {out_path}")
