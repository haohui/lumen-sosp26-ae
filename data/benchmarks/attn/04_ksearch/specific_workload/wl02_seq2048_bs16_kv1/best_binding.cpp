#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <limits>
#include <string>
#include <vector>

#include "kernel.h"
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define CHECK_HIP(call)                                                                 \
    do {                                                                                \
        hipError_t _err = (call);                                                       \
        TORCH_CHECK(_err == hipSuccess, "HIP error: ", hipGetErrorString(_err));       \
    } while (0)

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.defined() && k.defined() && v.defined(), "q, k, v must be defined");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D tensors");

    const int64_t bq = q.size(0), hq = q.size(1), sq = q.size(2), dq = q.size(3);
    const int64_t bk = k.size(0), hk = k.size(1), sk = k.size(2), dk = k.size(3);
    const int64_t bv = v.size(0), hv = v.size(1), sv = v.size(2), dv = v.size(3);

    TORCH_CHECK((bq == bk) && (bk == bv), "Batch size mismatch");
    TORCH_CHECK((sq == sk) && (sk == sv), "Sequence length mismatch");
    TORCH_CHECK((dq == dk) && (dk == dv), "Head dim mismatch");

    TORCH_CHECK(bq == 16, "Expected batch_size == 16, got ", bq);
    TORCH_CHECK(hq == 8, "Expected num_q_heads == 8, got ", hq);
    TORCH_CHECK(hk == 1, "Expected num_kv_heads == 1 for k, got ", hk);
    TORCH_CHECK(hv == 1, "Expected num_kv_heads == 1 for v, got ", hv);
    TORCH_CHECK(dq == 128, "Expected head_dim == 128, got ", dq);
    TORCH_CHECK(sq > 0, "seq_len must be > 0");
    TORCH_CHECK(sq <= std::numeric_limits<int>::max(), "seq_len too large for kernel int parameter");

    const bool q_cuda = q.is_cuda();
    const bool k_cuda = k.is_cuda();
    const bool v_cuda = v.is_cuda();
    const bool any_cuda = q_cuda || k_cuda || v_cuda;
    const bool all_cuda = q_cuda && k_cuda && v_cuda;

    TORCH_CHECK(!any_cuda || all_cuda, "q, k, v must all be HIP/ROCm tensors or all be CPU tensors");

    int target_device = 0;
    if (all_cuda) {
        target_device = q.get_device();
        TORCH_CHECK(k.get_device() == target_device && v.get_device() == target_device,
                    "All HIP/ROCm tensors must be on the same device");
    } else {
        int device_count = 0;
        CHECK_HIP(hipGetDeviceCount(&device_count));
        TORCH_CHECK(device_count > 0, "No HIP device available");
    }

    CHECK_HIP(hipSetDevice(target_device));
    const torch::Device cuda_dev(torch::kCUDA, target_device);

    auto to_bf16_cuda_contig = [&](const torch::Tensor& t) -> torch::Tensor {
        torch::Tensor x = t;
        if (!x.is_contiguous()) {
            x = x.contiguous();
        }
        if (!x.is_cuda() || x.get_device() != target_device || x.scalar_type() != torch::kBFloat16) {
            x = x.to(cuda_dev, torch::kBFloat16, false, false);
        }
        if (!x.is_contiguous()) {
            x = x.contiguous();
        }
        return x;
    };

    auto q_dev = to_bf16_cuda_contig(q);
    auto k_dev = to_bf16_cuda_contig(k);
    auto v_dev = to_bf16_cuda_contig(v);

    auto out_dev = torch::empty({16, 8, sq, 128}, q_dev.options().dtype(torch::kBFloat16));

    const hip_bfloat16* q_ptr = reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<c10::BFloat16>());
    const hip_bfloat16* k_ptr = reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<c10::BFloat16>());
    const hip_bfloat16* v_ptr = reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<c10::BFloat16>());
    hip_bfloat16* out_ptr = reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<c10::BFloat16>());

    dim3 block(512, 1, 1);
    dim3 grid(static_cast<unsigned int>(sq), 1, 16);

    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    hipError_t launch_err = ksearch_launch_online_row_streaming_causal_attention(
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
    TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

    if (all_cuda) {
        return out_dev;
    }

    return out_dev.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run",
        &run,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("sm_scale"),
        "Dense causal prefill attention (B=16, Hq=8, Hkv=1, D=128) using online row-streaming kernel");
}