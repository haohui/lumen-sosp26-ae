#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "kernel.h"
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define CHECK_HIP_OK(err, msg) TORCH_CHECK((err) == hipSuccess, msg, ": ", hipGetErrorString(err))

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.defined() && k.defined() && v.defined(), "q, k, v must be defined tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");

    const bool input_was_cuda = q.is_cuda();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    const int64_t bq = q_c.size(0);
    const int64_t hq = q_c.size(1);
    const int64_t sq = q_c.size(2);
    const int64_t dq = q_c.size(3);

    const int64_t bk = k_c.size(0);
    const int64_t hk = k_c.size(1);
    const int64_t sk = k_c.size(2);
    const int64_t dk = k_c.size(3);

    const int64_t bv = v_c.size(0);
    const int64_t hv = v_c.size(1);
    const int64_t sv = v_c.size(2);
    const int64_t dv = v_c.size(3);

    TORCH_CHECK(bq == 16, "batch_size must be 16");
    TORCH_CHECK(hq == 8, "num_q_heads must be 8");
    TORCH_CHECK(hk == 1 && hv == 1, "num_kv_heads must be 1");
    TORCH_CHECK(dq == 128 && dk == 128 && dv == 128, "head_dim must be 128");
    TORCH_CHECK((bq == bk) && (bk == bv), "batch mismatch");
    TORCH_CHECK((sq == sk) && (sk == sv), "seq_len mismatch");
    TORCH_CHECK((dq == dk) && (dk == dv), "head_dim mismatch");
    TORCH_CHECK(hv == hk, "k and v head mismatch");

    if (sq == 0) {
        auto empty_out = torch::empty({bq, hq, sq, dq}, q_c.options().dtype(torch::kBFloat16));
        return input_was_cuda ? empty_out : empty_out.to(torch::kCPU);
    }

    auto q_dev = q_c.is_cuda() ? q_c : q_c.to(torch::kCUDA);
    auto device = q_dev.device();

    auto k_dev = (k_c.is_cuda() && k_c.device() == device) ? k_c : k_c.to(device);
    auto v_dev = (v_c.is_cuda() && v_c.device() == device) ? v_c : v_c.to(device);

    if (device.has_index()) {
        hipError_t set_dev_err = hipSetDevice(device.index());
        CHECK_HIP_OK(set_dev_err, "hipSetDevice failed");
    }

    auto out_dev = torch::empty({bq, hq, sq, dq}, q_dev.options().dtype(torch::kBFloat16));

    const auto* q_ptr = reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<at::BFloat16>());
    const auto* k_ptr = reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<at::BFloat16>());
    const auto* v_ptr = reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<at::BFloat16>());
    auto* out_ptr = reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<at::BFloat16>());

    dim3 grid(static_cast<unsigned int>(sq), static_cast<unsigned int>(bq), 2);
    dim3 block(256, 1, 1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    hipError_t launch_err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_shared_kv_kernel(
        grid,
        block,
        0,
        stream,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        static_cast<int>(sq),
        static_cast<float>(sm_scale));
    CHECK_HIP_OK(launch_err, "Kernel launch failed");

    if (input_was_cuda) {
        return out_dev;
    }
    return out_dev.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          &run,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("sm_scale"),
          "dense_qkv_prefill_causal_h8_kv1_d128_bs16 (shared-KV across heads, HIP)");
}