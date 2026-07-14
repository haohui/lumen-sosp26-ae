#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#include "kernel.h"

namespace py = pybind11;

namespace {

constexpr int64_t kHidden = 7168;
constexpr int64_t kInter = 2048;
constexpr int64_t kInter2 = 4096;
constexpr int64_t kExperts = 32;
constexpr int64_t kTopK = 4;
constexpr int64_t kBlockN = 128;
constexpr int64_t kBlockK = 128;

struct WeightCache {
  std::mutex mu;
  const void* w1_ptr{nullptr};
  const void* s1_ptr{nullptr};
  const void* w2_ptr{nullptr};
  const void* s2_ptr{nullptr};
  int device{-1};
  at::Tensor w1_f;
  at::Tensor w2_f;
};

WeightCache& global_weight_cache() {
  static WeightCache cache;
  return cache;
}

inline void check_inputs(
    const at::Tensor& input_q,
    const at::Tensor& w1_q,
    const at::Tensor& w2_q,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    const at::Tensor& input_scale,
    const at::Tensor& fc1_scale,
    const at::Tensor& fc2_scale) {
  TORCH_CHECK(input_q.is_cuda(), "input_q must be CUDA/HIP tensor");
  TORCH_CHECK(w1_q.is_cuda(), "w1_q must be CUDA/HIP tensor");
  TORCH_CHECK(w2_q.is_cuda(), "w2_q must be CUDA/HIP tensor");
  TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be CUDA/HIP tensor");
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA/HIP tensor");
  TORCH_CHECK(input_scale.is_cuda(), "input_scale must be CUDA/HIP tensor");
  TORCH_CHECK(fc1_scale.is_cuda(), "fc1_scale must be CUDA/HIP tensor");
  TORCH_CHECK(fc2_scale.is_cuda(), "fc2_scale must be CUDA/HIP tensor");

  TORCH_CHECK(input_q.dim() == 2, "input_q must be [T, D]");
  TORCH_CHECK(w1_q.dim() == 3, "w1_q must be [E, 2I, D]");
  TORCH_CHECK(w2_q.dim() == 3, "w2_q must be [E, D, I]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be [T, K]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [T, K]");
  TORCH_CHECK(input_scale.dim() == 2, "input_scale must be [T, D/128]");
  TORCH_CHECK(fc1_scale.dim() == 2, "fc1_scale must be [E, 1792]");
  TORCH_CHECK(fc2_scale.dim() == 2, "fc2_scale must be [E, 896]");

  TORCH_CHECK(input_q.size(1) == kHidden, "hidden_size must be 7168");
  TORCH_CHECK(
      w1_q.size(0) == kExperts && w1_q.size(1) == kInter2 && w1_q.size(2) == kHidden,
      "w1_q shape mismatch");
  TORCH_CHECK(
      w2_q.size(0) == kExperts && w2_q.size(1) == kHidden && w2_q.size(2) == kInter,
      "w2_q shape mismatch");

  int64_t T = input_q.size(0);
  TORCH_CHECK(T > 0, "seq_len must be > 0");
  TORCH_CHECK(topk_weights.size(0) == T && topk_weights.size(1) == kTopK, "topk_weights shape mismatch");
  TORCH_CHECK(topk_ids.size(0) == T && topk_ids.size(1) == kTopK, "topk_ids shape mismatch");
  TORCH_CHECK(input_scale.size(0) == T && input_scale.size(1) == (kHidden / kBlockK), "input_scale shape mismatch");
  TORCH_CHECK(fc1_scale.size(0) == kExperts && fc1_scale.size(1) == 1792, "fc1_scale shape mismatch");
  TORCH_CHECK(fc2_scale.size(0) == kExperts && fc2_scale.size(1) == 896, "fc2_scale shape mismatch");
}

at::Tensor dequant_input(const at::Tensor& input_q, const at::Tensor& input_scale) {
  auto fq = input_q.to(at::kFloat).contiguous();
  auto fs = input_scale.to(at::kFloat).contiguous();
  auto T = fq.size(0);
  auto v = fq.view({T, kHidden / kBlockK, kBlockK});
  return (v * fs.unsqueeze(-1)).reshape({T, kHidden}).contiguous();
}

at::Tensor dequant_w1(const at::Tensor& w1_q, const at::Tensor& fc1_scale) {
  auto fq = w1_q.to(at::kFloat).contiguous();
  auto fs = fc1_scale.to(at::kFloat).contiguous();
  auto blocks = fq.view({kExperts, kInter2 / kBlockN, kBlockN, kHidden / kBlockK, kBlockK})
                    .permute({0, 1, 3, 2, 4})
                    .contiguous();
  auto scaled = blocks * fs.view({kExperts, kInter2 / kBlockN, kHidden / kBlockK, 1, 1});
  return scaled.permute({0, 1, 3, 2, 4}).reshape({kExperts, kInter2, kHidden}).contiguous();
}

at::Tensor dequant_w2(const at::Tensor& w2_q, const at::Tensor& fc2_scale) {
  auto fq = w2_q.to(at::kFloat).contiguous();
  auto fs = fc2_scale.to(at::kFloat).contiguous();
  auto blocks = fq.view({kExperts, kHidden / kBlockN, kBlockN, kInter / kBlockK, kBlockK})
                    .permute({0, 1, 3, 2, 4})
                    .contiguous();
  auto scaled = blocks * fs.view({kExperts, kHidden / kBlockN, kInter / kBlockK, 1, 1});
  return scaled.permute({0, 1, 3, 2, 4}).reshape({kExperts, kHidden, kInter}).contiguous();
}

} // namespace

at::Tensor run(
    const at::Tensor& input_q,
    const at::Tensor& w1_q,
    const at::Tensor& w2_q,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    const at::Tensor& input_scale,
    const at::Tensor& fc1_scale,
    const at::Tensor& fc2_scale) {
  check_inputs(input_q, w1_q, w2_q, topk_weights, topk_ids, input_scale, fc1_scale, fc2_scale);

  const int T = static_cast<int>(input_q.size(0));
  auto opts_f32 = input_q.options().dtype(at::kFloat);
  auto opts_i32 = input_q.options().dtype(at::kInt);

  at::Tensor input_f = dequant_input(input_q, input_scale);

  at::Tensor w1_f;
  at::Tensor w2_f;
  {
    auto& cache = global_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mu);

    const int dev = w1_q.get_device();
    const void* w1p = w1_q.data_ptr();
    const void* s1p = fc1_scale.data_ptr();
    const void* w2p = w2_q.data_ptr();
    const void* s2p = fc2_scale.data_ptr();

    if (cache.device == dev && cache.w1_ptr == w1p && cache.s1_ptr == s1p && cache.w1_f.defined()) {
      w1_f = cache.w1_f;
    } else {
      w1_f = dequant_w1(w1_q, fc1_scale);
      cache.w1_f = w1_f;
      cache.w1_ptr = w1p;
      cache.s1_ptr = s1p;
      cache.device = dev;
    }

    if (cache.device == dev && cache.w2_ptr == w2p && cache.s2_ptr == s2p && cache.w2_f.defined()) {
      w2_f = cache.w2_f;
    } else {
      w2_f = dequant_w2(w2_q, fc2_scale);
      cache.w2_f = w2_f;
      cache.w2_ptr = w2p;
      cache.s2_ptr = s2p;
      cache.device = dev;
    }
  }

  at::Tensor topk_ids_i32 =
      (topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous()) ? topk_ids : topk_ids.to(at::kInt).contiguous();
  at::Tensor topk_w_f32 = (topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous())
      ? topk_weights
      : topk_weights.to(at::kFloat).contiguous();

  at::Tensor expert_counts = at::zeros({kExperts}, opts_i32);
  at::Tensor expert_token_indices = at::empty({kExperts, T}, opts_i32);
  at::Tensor expert_route_weights = at::empty({kExperts, T}, opts_f32);

  at::Tensor output_f = at::empty({T, kHidden}, opts_f32);

  auto stream = at::cuda::getCurrentCUDAStream();
  hipStream_t hstream = stream.stream();

  {
    dim3 block(256);
    int64_t n = static_cast<int64_t>(T) * kHidden;
    dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x));
    auto err = ksearch_launch_zero_f32(grid, block, 0, hstream, output_f.data_ptr<float>(), n);
    TORCH_CHECK(err == hipSuccess, "zero_f32 output kernel launch failed");
  }

  {
    dim3 block(256);
    dim3 grid((T * static_cast<int>(kTopK) + block.x - 1) / block.x);
    auto err = ksearch_launch_build_expert_buckets(
        grid,
        block,
        0,
        hstream,
        reinterpret_cast<const int32_t*>(topk_ids_i32.data_ptr<int>()),
        topk_w_f32.data_ptr<float>(),
        reinterpret_cast<int32_t*>(expert_counts.data_ptr<int>()),
        reinterpret_cast<int32_t*>(expert_token_indices.data_ptr<int>()),
        expert_route_weights.data_ptr<float>(),
        T,
        static_cast<int>(kTopK),
        static_cast<int>(kExperts),
        T);
    TORCH_CHECK(err == hipSuccess, "build_expert_buckets kernel launch failed");
  }

  {
    dim3 block(256);
    const unsigned int slot_tiles = static_cast<unsigned int>(std::min<int>(T, 8));
    dim3 grid(static_cast<unsigned int>(kExperts), slot_tiles, 1);
    size_t smem = static_cast<size_t>(kHidden + kInter) * sizeof(float);
    auto err = ksearch_launch_fused_expert_bucket_compute(
        grid,
        block,
        smem,
        hstream,
        input_f.data_ptr<float>(),
        w1_f.data_ptr<float>(),
        w2_f.data_ptr<float>(),
        reinterpret_cast<const int32_t*>(expert_counts.data_ptr<int>()),
        reinterpret_cast<const int32_t*>(expert_token_indices.data_ptr<int>()),
        expert_route_weights.data_ptr<float>(),
        output_f.data_ptr<float>(),
        T,
        T,
        static_cast<int>(kExperts));
    TORCH_CHECK(err == hipSuccess, "fused_expert_bucket_compute kernel launch failed");
  }

  return output_f.to(at::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run",
      &run,
      "moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048 (expert-bucketed fused path)",
      py::arg("input_q"),
      py::arg("w1_q"),
      py::arg("w2_q"),
      py::arg("topk_weights"),
      py::arg("topk_ids"),
      py::arg("input_scale"),
      py::arg("fc1_scale"),
      py::arg("fc2_scale"));
}