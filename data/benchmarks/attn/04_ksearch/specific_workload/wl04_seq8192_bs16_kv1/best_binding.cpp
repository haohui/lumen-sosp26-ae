#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <limits>
#include <string>

#include "kernel.h"
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

static inline torch::Tensor to_cuda_bf16_contig(const torch::Tensor& t) {
    torch::Tensor x = t;
    if (!x.is_cuda()) {
        x = x.to(torch::kCUDA);
    }
    if (x.scalar_type() != torch::kBFloat16) {
        x = x.to(torch::kBFloat16);
    }
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x;
}

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(q.size(0) == kFixedBatchSize, "batch_size must be 16");
    TORCH_CHECK(q.size(1) == kNumQHeads, "num_q_heads must be 8");
    TORCH_CHECK(k.size(1) == kNumKVHeads && v.size(1) == kNumKVHeads, "num_kv_heads must be 1");
    TORCH_CHECK(q.size(3) == kHeadDim && k.size(3) == kHeadDim && v.size(3) == kHeadDim, "head_dim must be 128");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "seq_len mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v head mismatch");

    const int64_t seq_len64 = q.size(2);
    TORCH_CHECK(seq_len64 > 0, "seq_len must be > 0");
    TORCH_CHECK(seq_len64 <= std::numeric_limits<int>::max(), "seq_len too large");
    const int seq_len = static_cast<int>(seq_len64);

    const bool return_cpu = !q.is_cuda();

    auto q_dev = to_cuda_bf16_contig(q);
    auto k_dev = to_cuda_bf16_contig(k);
    auto v_dev = to_cuda_bf16_contig(v);

    auto out_dev = torch::empty({kFixedBatchSize, kNumQHeads, seq_len64, kHeadDim},
                                q_dev.options().dtype(torch::kBFloat16));

    const hip_bfloat16* q_ptr =
        reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<c10::BFloat16>());
    const hip_bfloat16* k_ptr =
        reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<c10::BFloat16>());
    const hip_bfloat16* v_ptr =
        reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<c10::BFloat16>());
    hip_bfloat16* out_ptr =
        reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<c10::BFloat16>());

    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    hipError_t err = launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        seq_len,
        static_cast<float>(sm_scale),
        stream);
    TORCH_CHECK(err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(err));

    if (return_cpu) {
        return out_dev.to(torch::kCPU);
    }
    return out_dev;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          &run,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("sm_scale"),
          "dense_qkv_prefill_causal_h8_kv1_d128_bs16 forward (HIP)");
}