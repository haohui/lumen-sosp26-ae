#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <c10/hip/HIPStream.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "kernel.h"

namespace py = pybind11;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d) TORCH_CHECK((x).dim() == (d), #x " must have " #d " dims")
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be on HIP device")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static inline void check_shapes(
    const torch::Tensor& input_q,
    const torch::Tensor& w1_q,
    const torch::Tensor& w2_q,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const torch::Tensor& input_scale,
    const torch::Tensor& fc1_scale,
    const torch::Tensor& fc2_scale) {
    CHECK_DIM(input_q, 2);
    CHECK_DIM(w1_q, 3);
    CHECK_DIM(w2_q, 3);
    CHECK_DIM(topk_weights, 2);
    CHECK_DIM(topk_ids, 2);
    CHECK_DIM(input_scale, 2);
    CHECK_DIM(fc1_scale, 2);
    CHECK_DIM(fc2_scale, 2);

    const int64_t seq_len = input_q.size(0);

    TORCH_CHECK(input_q.size(1) == MOE_HIDDEN_SIZE, "input_q.shape[1] must be 7168");
    TORCH_CHECK(
        w1_q.size(0) == MOE_NUM_EXPERTS && w1_q.size(1) == MOE_INTERMEDIATE2_SIZE && w1_q.size(2) == MOE_HIDDEN_SIZE,
        "w1_q shape must be [32, 4096, 7168]");
    TORCH_CHECK(
        w2_q.size(0) == MOE_NUM_EXPERTS && w2_q.size(1) == MOE_HIDDEN_SIZE && w2_q.size(2) == MOE_INTERMEDIATE_SIZE,
        "w2_q shape must be [32, 7168, 2048]");
    TORCH_CHECK(topk_weights.size(0) == seq_len && topk_weights.size(1) == MOE_TOPK,
                "topk_weights shape must be [seq_len, 4]");
    TORCH_CHECK(topk_ids.size(0) == seq_len && topk_ids.size(1) == MOE_TOPK, "topk_ids shape must be [seq_len, 4]");
    TORCH_CHECK(input_scale.size(0) == seq_len && input_scale.size(1) == MOE_HIDDEN_BLOCKS,
                "input_scale shape must be [seq_len, 56]");
    TORCH_CHECK(fc1_scale.size(0) == MOE_NUM_EXPERTS && fc1_scale.size(1) == MOE_FC1_SCALES_PER_EXPERT,
                "fc1_scale shape must be [32, 1792]");
    TORCH_CHECK(fc2_scale.size(0) == MOE_NUM_EXPERTS && fc2_scale.size(1) == MOE_FC2_SCALES_PER_EXPERT,
                "fc2_scale shape must be [32, 896]");

    TORCH_CHECK(seq_len > 0, "seq_len must be > 0");
}

torch::Tensor run(
    torch::Tensor input_q,
    torch::Tensor w1_q,
    torch::Tensor w2_q,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor input_scale,
    torch::Tensor fc1_scale,
    torch::Tensor fc2_scale) {
    bool return_cpu = false;

    if (!input_q.is_cuda()) {
        return_cpu = true;
        const auto dev = torch::Device(torch::kCUDA, 0);
        input_q = input_q.to(dev);
        w1_q = w1_q.to(dev);
        w2_q = w2_q.to(dev);
        topk_weights = topk_weights.to(dev);
        topk_ids = topk_ids.to(dev);
        input_scale = input_scale.to(dev);
        fc1_scale = fc1_scale.to(dev);
        fc2_scale = fc2_scale.to(dev);
    }

    CHECK_CUDA(input_q);
    CHECK_CUDA(w1_q);
    CHECK_CUDA(w2_q);
    CHECK_CUDA(topk_weights);
    CHECK_CUDA(topk_ids);
    CHECK_CUDA(input_scale);
    CHECK_CUDA(fc1_scale);
    CHECK_CUDA(fc2_scale);

    CHECK_CONTIGUOUS(input_q);
    CHECK_CONTIGUOUS(w1_q);
    CHECK_CONTIGUOUS(w2_q);
    CHECK_CONTIGUOUS(topk_weights);
    CHECK_CONTIGUOUS(topk_ids);
    CHECK_CONTIGUOUS(input_scale);
    CHECK_CONTIGUOUS(fc1_scale);
    CHECK_CONTIGUOUS(fc2_scale);

    CHECK_FLOAT32(topk_weights);
    CHECK_INT32(topk_ids);
    CHECK_FLOAT32(input_scale);
    CHECK_FLOAT32(fc1_scale);
    CHECK_FLOAT32(fc2_scale);

    TORCH_CHECK(input_q.is_floating_point() && input_q.element_size() == 1, "input_q must be fp8 tensor");
    TORCH_CHECK(w1_q.is_floating_point() && w1_q.element_size() == 1, "w1_q must be fp8 tensor");
    TORCH_CHECK(w2_q.is_floating_point() && w2_q.element_size() == 1, "w2_q must be fp8 tensor");

    TORCH_CHECK(input_q.device() == w1_q.device() && input_q.device() == w2_q.device() &&
                    input_q.device() == topk_weights.device() && input_q.device() == topk_ids.device() &&
                    input_q.device() == input_scale.device() && input_q.device() == fc1_scale.device() &&
                    input_q.device() == fc2_scale.device(),
                "all tensors must be on the same device");

    check_shapes(input_q, w1_q, w2_q, topk_weights, topk_ids, input_scale, fc1_scale, fc2_scale);

    const int64_t seq_len_i64 = input_q.size(0);
    TORCH_CHECK(seq_len_i64 <= static_cast<int64_t>(std::numeric_limits<int>::max()), "seq_len too large");
    const int seq_len = static_cast<int>(seq_len_i64);

    hipError_t st = hipSetDevice(input_q.get_device());
    TORCH_CHECK(st == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(st));

    auto out = torch::empty(
        {seq_len_i64, static_cast<int64_t>(MOE_HIDDEN_SIZE)},
        torch::TensorOptions().device(input_q.device()).dtype(torch::kBFloat16));

    const dim3 grid(static_cast<unsigned int>(seq_len), 1, 1);
    const dim3 block(static_cast<unsigned int>(MOE_THREADS), 1, 1);
    hipStream_t stream = c10::hip::getCurrentHIPStream(input_q.get_device()).stream();

    st = ksearch_launch_moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048_kernel(
        grid,
        block,
        0,
        stream,
        reinterpret_cast<const uint8_t*>(input_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_q.data_ptr()),
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        input_scale.data_ptr<float>(),
        fc1_scale.data_ptr<float>(),
        fc2_scale.data_ptr<float>(),
        reinterpret_cast<hip_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        seq_len);

    TORCH_CHECK(st == hipSuccess, "Kernel launch failed: ", hipGetErrorString(st));

    if (return_cpu) {
        return out.cpu();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run",
        &run,
        py::arg("input_q"),
        py::arg("w1_q"),
        py::arg("w2_q"),
        py::arg("topk_weights"),
        py::arg("topk_ids"),
        py::arg("input_scale"),
        py::arg("fc1_scale"),
        py::arg("fc2_scale"));
}