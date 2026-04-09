#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "kernel.h"

#include <cstdint>
#include <string>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define HIP_CHECK(cmd)                                                                 \
    do {                                                                               \
        hipError_t e = (cmd);                                                          \
        TORCH_CHECK(e == hipSuccess, "HIP error: ", hipGetErrorString(e),             \
                    " (", static_cast<int>(e), ") at ", __FILE__, ":", __LINE__);     \
    } while (0)

static float parse_sm_scale(const py::object& sm_scale_obj) {
    if (py::isinstance<py::float_>(sm_scale_obj) || py::isinstance<py::int_>(sm_scale_obj)) {
        return sm_scale_obj.cast<float>();
    }
    if (py::hasattr(sm_scale_obj, "item")) {
        return sm_scale_obj.attr("item")().cast<float>();
    }
    return sm_scale_obj.cast<float>();
}

static torch::Tensor to_cuda_bf16_contiguous(const torch::Tensor& t, const torch::Device& device) {
    torch::Tensor x = t;
    if (x.scalar_type() != torch::kBFloat16) {
        x = x.to(torch::kBFloat16);
    }
    if (!x.is_cuda() || x.device() != device) {
        x = x.to(device);
    }
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x;
}

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, py::object sm_scale_obj) {
    TORCH_CHECK(q.defined() && k.defined() && v.defined(), "q, k, v must be defined tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D tensors");

    const int64_t bq = q.size(0), hq = q.size(1), sq = q.size(2), dq = q.size(3);
    const int64_t bk = k.size(0), hk = k.size(1), sk = k.size(2), dk = k.size(3);
    const int64_t bv = v.size(0), hv = v.size(1), sv = v.size(2), dv = v.size(3);

    TORCH_CHECK(bq == 16, "batch_size must be 16");
    TORCH_CHECK(hq == 8, "num_q_heads must be 8");
    TORCH_CHECK(hk == 1 && hv == 1, "num_kv_heads for k and v must be 1");
    TORCH_CHECK(dq == 128 && dk == 128 && dv == 128, "head_dim must be 128");
    TORCH_CHECK(bq == bk && bk == bv, "batch dimensions must match");
    TORCH_CHECK(sq == sk && sk == sv, "sequence lengths must match");
    TORCH_CHECK(dq == dk && dk == dv, "head dimensions must match");
    TORCH_CHECK(sq > 0, "seq_len must be > 0");

    const float sm_scale = parse_sm_scale(sm_scale_obj);

    torch::Device device(torch::kCUDA, 0);
    if (q.is_cuda()) {
        device = q.device();
    } else if (k.is_cuda()) {
        device = k.device();
    } else if (v.is_cuda()) {
        device = v.device();
    }

    const int device_index = device.has_index() ? device.index() : 0;
    HIP_CHECK(hipSetDevice(device_index));

    torch::Tensor q_dev = to_cuda_bf16_contiguous(q, device);
    torch::Tensor k_dev = to_cuda_bf16_contiguous(k, device);
    torch::Tensor v_dev = to_cuda_bf16_contiguous(v, device);

    auto out_dev = torch::empty(
        {bq, hq, sq, dq},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    dim3 block(64, 8, 1);
    dim3 grid(static_cast<unsigned int>(sq), 16u, 1u);
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const auto* q_ptr = reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<c10::BFloat16>());
    const auto* k_ptr = reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<c10::BFloat16>());
    const auto* v_ptr = reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<c10::BFloat16>());
    auto* out_ptr = reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<c10::BFloat16>());

    hipError_t launch_err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass(
        grid, block, 0, stream, q_ptr, k_ptr, v_ptr, out_ptr, static_cast<int>(sq), sm_scale);
    TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

    return out_dev;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sm_scale"),
          "Dense causal prefill attention (B=16, Hq=8, Hkv=1, D=128) single-pass streaming kernel");
}