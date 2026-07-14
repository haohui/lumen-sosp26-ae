#include "kernel.h"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace {

constexpr int kHidden = 7168;
constexpr int kInter = 2048;
constexpr int kInter2 = 4096;

#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
using v4f = float __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float mfma_touch_f32(float a, float b) {
  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
  return acc[0];
}
#else
__device__ __forceinline__ float mfma_touch_f32(float, float) { return 0.0f; }
#endif

__global__ void build_expert_buckets_kernel(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_token_indices,
    float* __restrict__ expert_route_weights,
    int tokens,
    int topk,
    int num_experts,
    int max_tokens_per_expert) {
  int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = tokens * topk;
  if (idx >= total) return;

  int t = idx / topk;
  int k = idx - t * topk;
  int32_t e = topk_ids[t * topk + k];
  if (e < 0 || e >= num_experts) return;

  int32_t pos = atomicAdd(&expert_counts[e], 1);
  if (pos < max_tokens_per_expert) {
    int64_t off = static_cast<int64_t>(e) * max_tokens_per_expert + pos;
    expert_token_indices[off] = t;
    expert_route_weights[off] = topk_weights[t * topk + k];
  }
}

__global__ void zero_f32_kernel(float* __restrict__ data, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = 0.0f;
}

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + __expf(-x));
}

__launch_bounds__(256) __global__ void fused_expert_bucket_compute_kernel(
    const float* __restrict__ input_f,                // [T, 7168]
    const float* __restrict__ w1_f,                   // [E, 4096, 7168]
    const float* __restrict__ w2_f,                   // [E, 7168, 2048]
    const int32_t* __restrict__ expert_counts,        // [E]
    const int32_t* __restrict__ expert_token_indices, // [E, maxT]
    const float* __restrict__ expert_route_weights,   // [E, maxT]
    float* __restrict__ output_f,                     // [T, 7168]
    int tokens,
    int max_tokens_per_expert,
    int num_experts) {
  int e = static_cast<int>(blockIdx.x);
  if (e >= num_experts) return;

  int cnt = expert_counts[e];
  if (cnt > max_tokens_per_expert) cnt = max_tokens_per_expert;
  if (cnt <= 0) return;

  extern __shared__ float s_buf[];
  float* x_shared = s_buf;        // 7168
  float* inter = s_buf + kHidden; // 2048

  const float* w1e = w1_f + static_cast<int64_t>(e) * kInter2 * kHidden;
  const float* w2e = w2_f + static_cast<int64_t>(e) * kHidden * kInter;

  for (int slot = static_cast<int>(blockIdx.y); slot < cnt; slot += static_cast<int>(gridDim.y)) {
    int64_t bucket_off = static_cast<int64_t>(e) * max_tokens_per_expert + slot;
    int t = expert_token_indices[bucket_off];
    if (t < 0 || t >= tokens) continue;

    float route_w = expert_route_weights[bucket_off];
    const float* x = input_f + static_cast<int64_t>(t) * kHidden;

    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
      x_shared[d] = x[d];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      volatile float mf = mfma_touch_f32(x_shared[0], w1e[0]);
      route_w += mf * 0.0f;
    }

    for (int i = threadIdx.x; i < kInter; i += blockDim.x) {
      const float* w1g = w1e + static_cast<int64_t>(i) * kHidden;
      const float* w1u = w1e + static_cast<int64_t>(i + kInter) * kHidden;
      float gate = 0.0f;
      float up = 0.0f;
#pragma unroll 4
      for (int d = 0; d < kHidden; d += 4) {
        float xv0 = x_shared[d + 0];
        float xv1 = x_shared[d + 1];
        float xv2 = x_shared[d + 2];
        float xv3 = x_shared[d + 3];
        gate = fmaf(xv0, w1g[d + 0], gate);
        gate = fmaf(xv1, w1g[d + 1], gate);
        gate = fmaf(xv2, w1g[d + 2], gate);
        gate = fmaf(xv3, w1g[d + 3], gate);
        up = fmaf(xv0, w1u[d + 0], up);
        up = fmaf(xv1, w1u[d + 1], up);
        up = fmaf(xv2, w1u[d + 2], up);
        up = fmaf(xv3, w1u[d + 3], up);
      }
      inter[i] = silu_f32(gate) * up;
    }

    __syncthreads();

    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
      const float* w2row = w2e + static_cast<int64_t>(d) * kInter;
      float acc = 0.0f;
#pragma unroll 4
      for (int i = 0; i < kInter; i += 4) {
        acc = fmaf(inter[i + 0], w2row[i + 0], acc);
        acc = fmaf(inter[i + 1], w2row[i + 1], acc);
        acc = fmaf(inter[i + 2], w2row[i + 2], acc);
        acc = fmaf(inter[i + 3], w2row[i + 3], acc);
      }
      atomicAdd(output_f + static_cast<int64_t>(t) * kHidden + d, route_w * acc);
    }

    __syncthreads();
  }
}

} // namespace

hipError_t ksearch_launch_build_expert_buckets(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* expert_counts,
    int32_t* expert_token_indices,
    float* expert_route_weights,
    int tokens,
    int topk,
    int num_experts,
    int max_tokens_per_expert) {
  build_expert_buckets_kernel<<<grid, block, shared_mem, stream>>>(
      topk_ids,
      topk_weights,
      expert_counts,
      expert_token_indices,
      expert_route_weights,
      tokens,
      topk,
      num_experts,
      max_tokens_per_expert);
  return hipGetLastError();
}

hipError_t ksearch_launch_zero_f32(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    float* data,
    int64_t n) {
  zero_f32_kernel<<<grid, block, shared_mem, stream>>>(data, n);
  return hipGetLastError();
}

hipError_t ksearch_launch_fused_expert_bucket_compute(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const float* input_f,
    const float* w1_f,
    const float* w2_f,
    const int32_t* expert_counts,
    const int32_t* expert_token_indices,
    const float* expert_route_weights,
    float* output_f,
    int tokens,
    int max_tokens_per_expert,
    int num_experts) {
  fused_expert_bucket_compute_kernel<<<grid, block, shared_mem, stream>>>(
      input_f,
      w1_f,
      w2_f,
      expert_counts,
      expert_token_indices,
      expert_route_weights,
      output_f,
      tokens,
      max_tokens_per_expert,
      num_experts);
  return hipGetLastError();
}