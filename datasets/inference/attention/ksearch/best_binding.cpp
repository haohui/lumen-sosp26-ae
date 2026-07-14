#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "kernel.h"

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA/HIP tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA/HIP tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA/HIP tensor");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    const int64_t B = q_c.size(0);
    const int64_t S = q_c.size(1);
    const int64_t HQ = q_c.size(2);
    const int64_t D = q_c.size(3);

    TORCH_CHECK(k_c.size(0) == B && v_c.size(0) == B, "batch mismatch");
    TORCH_CHECK(k_c.size(1) == S && v_c.size(1) == S, "seq_len mismatch");
    TORCH_CHECK(k_c.size(3) == D && v_c.size(3) == D, "head_dim mismatch");

    const int64_t HK = k_c.size(2);
    const int64_t HV = v_c.size(2);

    TORCH_CHECK(HQ == 8, "num_q_heads must be 8");
    TORCH_CHECK(HK == 1 || HK == 8, "num_kv_heads must be 1 or 8");
    TORCH_CHECK(HV == HK, "v num_kv_heads must equal k num_kv_heads");
    TORCH_CHECK(D == 128, "head_dim must be 128");

    auto out = torch::empty_like(q_c);

    dim3 block(64, 1, 1);
    dim3 grid(static_cast<unsigned int>(S),
              static_cast<unsigned int>(8),
              static_cast<unsigned int>(B));

    auto stream = at::cuda::getCurrentCUDAStream();

    const hip_bfloat16* q_ptr = reinterpret_cast<const hip_bfloat16*>(q_c.data_ptr<at::BFloat16>());
    const hip_bfloat16* k_ptr = reinterpret_cast<const hip_bfloat16*>(k_c.data_ptr<at::BFloat16>());
    const hip_bfloat16* v_ptr = reinterpret_cast<const hip_bfloat16*>(v_c.data_ptr<at::BFloat16>());
    hip_bfloat16* o_ptr = reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>());

    hipError_t err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
        grid,
        block,
        0,
        stream.stream(),
        q_ptr,
        k_ptr,
        v_ptr,
        static_cast<float>(sm_scale),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(HK),
        o_ptr);

    TORCH_CHECK(err == hipSuccess, "HIP launch failed: ", hipGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "dense_qkv_prefill_causal_h8_kv1or8_d128 (HIP)");
}
