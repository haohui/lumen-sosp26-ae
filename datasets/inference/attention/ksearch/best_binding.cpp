#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

#include <limits>
#include <string>

#include "kernel.h"

namespace py = pybind11;

static void validate_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D tensors.");
    TORCH_CHECK(
        q.device() == k.device() && q.device() == v.device(),
        "q, k, v must be on the same device.");

    const auto B = q.size(0);
    const auto S = q.size(1);
    const auto Hq = q.size(2);
    const auto D = q.size(3);

    TORCH_CHECK(
        k.size(0) == B && k.size(1) == S && k.size(3) == D,
        "k shape must match [B, S, Hkv, D] with B,S,D from q.");
    TORCH_CHECK(
        v.size(0) == B && v.size(1) == S && v.size(3) == D,
        "v shape must match [B, S, Hkv, D] with B,S,D from q.");
    TORCH_CHECK(v.size(2) == k.size(2), "v num_kv_heads must equal k num_kv_heads.");

    TORCH_CHECK(Hq == 8, "num_q_heads must be 8.");
    TORCH_CHECK(k.size(2) == 1 || k.size(2) == 8, "num_kv_heads must be 1 or 8.");
    TORCH_CHECK(D == 128, "head_dim must be 128.");

    TORCH_CHECK(B <= std::numeric_limits<int>::max(), "B too large for int kernel args.");
    TORCH_CHECK(S <= std::numeric_limits<int>::max(), "S too large for int kernel args.");
}

static torch::Tensor to_bf16_contig_on_device(const torch::Tensor& t, const torch::Device& dev)
{
    if(t.device() == dev && t.scalar_type() == torch::kBFloat16 && t.is_contiguous())
    {
        return t;
    }

    auto out = t;
    if(out.device() != dev || out.scalar_type() != torch::kBFloat16)
    {
        out = out.to(dev, torch::kBFloat16, /*non_blocking=*/false, /*copy=*/false);
    }
    if(!out.is_contiguous())
    {
        out = out.contiguous();
    }
    return out;
}

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale)
{
    validate_inputs(q, k, v);

    const bool input_was_cuda = q.is_cuda();

    int device_index = 0;
    if(input_was_cuda)
    {
        device_index = q.get_device();
    }
    else
    {
        hipError_t st = hipGetDevice(&device_index);
        TORCH_CHECK(st == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(st));
    }

    hipError_t hip_st = hipSetDevice(device_index);
    TORCH_CHECK(hip_st == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(hip_st));

    const torch::Device dev(torch::kCUDA, device_index);

    auto q_dev = to_bf16_contig_on_device(q, dev);
    auto k_dev = to_bf16_contig_on_device(k, dev);
    auto v_dev = to_bf16_contig_on_device(v, dev);

    const int B = static_cast<int>(q_dev.size(0));
    const int S = static_cast<int>(q_dev.size(1));
    const int Hq = static_cast<int>(q_dev.size(2));
    const int D = static_cast<int>(q_dev.size(3));
    const int Hkv = static_cast<int>(k_dev.size(2));

    auto out_dev = torch::empty({B, S, Hq, D}, q_dev.options().dtype(torch::kBFloat16).device(dev));

    constexpr int Q_BLOCK = 16;
    const dim3 block(64, 1, 1);
    const dim3 grid(
        static_cast<unsigned int>((S + Q_BLOCK - 1) / Q_BLOCK),
        static_cast<unsigned int>(Hq),
        static_cast<unsigned int>(B));

    const float scale = static_cast<float>(sm_scale);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device_index).stream();
    hipError_t launch_err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
        grid,
        block,
        0,
        stream,
        reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr()),
        B,
        S,
        Hq,
        Hkv,
        D,
        scale);

    TORCH_CHECK(
        launch_err != hipErrorNotSupported,
        "MFMA main-path kernel is not supported in this build/runtime (rocWMMA unavailable or non-gfx942 device).");
    TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

    if(input_was_cuda)
    {
        return out_dev;
    }
    return out_dev.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "run",
        &run,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("sm_scale"),
        "dense_qkv_prefill_causal_h8_kv1or8_d128");
}
