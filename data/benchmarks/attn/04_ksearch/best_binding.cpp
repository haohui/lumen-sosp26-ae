#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <vector>
#include <limits>

#include "kernel.h"
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>

namespace py = pybind11;

#define HIP_CHECK(cmd)                                                                 \
    do {                                                                               \
        hipError_t e__ = (cmd);                                                        \
        TORCH_CHECK(e__ == hipSuccess, "HIP error: ", hipGetErrorString(e__));        \
    } while (0)

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");

    const int64_t B = q.size(0);
    const int64_t S = q.size(1);
    const int64_t HQ = q.size(2);
    const int64_t DQ = q.size(3);

    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "batch_size mismatch");
    TORCH_CHECK(k.size(1) == S && v.size(1) == S, "seq_len mismatch");
    TORCH_CHECK(k.size(3) == DQ && v.size(3) == DQ, "head_dim mismatch");
    TORCH_CHECK(HQ == 8, "num_q_heads must be 8");
    TORCH_CHECK(DQ == 128, "head_dim must be 128");

    const int64_t HK = k.size(2);
    TORCH_CHECK(HK == 1 || HK == 8, "num_kv_heads must be 1 or 8");
    TORCH_CHECK(v.size(2) == HK, "k/v num_kv_heads mismatch");

    TORCH_CHECK(B > 0 && S > 0, "batch_size and seq_len must be > 0");
    TORCH_CHECK(B <= std::numeric_limits<int>::max() && S <= std::numeric_limits<int>::max(),
                "B and S must fit int32");

    const bool return_cpu = !q.is_cuda();

    int device_index = 0;
    if (q.is_cuda()) device_index = q.get_device();
    else if (k.is_cuda()) device_index = k.get_device();
    else if (v.is_cuda()) device_index = v.get_device();

    HIP_CHECK(hipSetDevice(device_index));

    auto q_dev = q.contiguous();
    auto k_dev = k.contiguous();
    auto v_dev = v.contiguous();

    if (!q_dev.is_cuda()) q_dev = q_dev.to(torch::Device(torch::kCUDA, device_index));
    if (!k_dev.is_cuda() || k_dev.get_device() != device_index) {
        k_dev = k_dev.to(torch::Device(torch::kCUDA, device_index));
    }
    if (!v_dev.is_cuda() || v_dev.get_device() != device_index) {
        v_dev = v_dev.to(torch::Device(torch::kCUDA, device_index));
    }

    auto out_dev = torch::empty({B, S, HQ, DQ}, q_dev.options().dtype(torch::kBFloat16));

    const auto* q_ptr = reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<c10::BFloat16>());
    const auto* k_ptr = reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<c10::BFloat16>());
    const auto* v_ptr = reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<c10::BFloat16>());
    auto* o_ptr = reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<c10::BFloat16>());

    dim3 grid(static_cast<unsigned int>(S),
              static_cast<unsigned int>(HQ),
              static_cast<unsigned int>(B));
    dim3 block(64, 1, 1);
    size_t shared_mem = 0;
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    HIP_CHECK(ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128_online(
        grid,
        block,
        shared_mem,
        stream,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(HQ),
        static_cast<int>(HK),
        static_cast<float>(sm_scale)));

    if (return_cpu) {
        return out_dev.to(torch::kCPU);
    }
    return out_dev;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sm_scale"));
}
