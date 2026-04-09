#ifndef DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H
#define DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H

#include <torch/extension.h>

void launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float sm_scale,
    at::Tensor& out);

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale);

#endif  // DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H