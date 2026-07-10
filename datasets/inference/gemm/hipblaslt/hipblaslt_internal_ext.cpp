#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt.h>
#include <torch/extension.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace {

constexpr uint64_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;

struct Plan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t matmul_desc = nullptr;
    hipblasLtMatrixLayout_t layout_a = nullptr;
    hipblasLtMatrixLayout_t layout_b = nullptr;
    hipblasLtMatrixLayout_t layout_c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    at::Tensor workspace;
};

std::mutex g_mu;
std::unordered_map<std::string, std::shared_ptr<Plan>> g_plans;

static inline void check_lt(hipblasStatus_t st, const char *msg) {
    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, msg, " hipBLASLt status=", static_cast<int>(st));
}

std::string make_key(int device, int64_t m, int64_t n, int64_t k) {
    std::ostringstream oss;
    oss << device << ":" << m << "x" << n << "x" << k;
    return oss.str();
}

std::shared_ptr<Plan> get_or_create_plan(const at::Tensor &a, const at::Tensor &b, const at::Tensor &d) {
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(0);
    const int device = a.get_device();
    const std::string key = make_key(device, m, n, k);

    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_plans.find(key);
        if (it != g_plans.end()) {
            return it->second;
        }
    }

    auto plan = std::make_shared<Plan>();
    check_lt(hipblasLtCreate(&plan->handle), "hipblasLtCreate failed");
    check_lt(
        hipblasLtMatmulDescCreate(&plan->matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate failed");
    int32_t op_t = HIPBLAS_OP_T;
    int32_t op_n = HIPBLAS_OP_N;
    check_lt(
        hipblasLtMatmulDescSetAttribute(
            plan->matmul_desc,
            HIPBLASLT_MATMUL_DESC_TRANSA,
            &op_t,
            sizeof(op_t)),
        "set TRANSA failed");
    check_lt(
        hipblasLtMatmulDescSetAttribute(
            plan->matmul_desc,
            HIPBLASLT_MATMUL_DESC_TRANSB,
            &op_n,
            sizeof(op_n)),
        "set TRANSB failed");

    // Column-major reinterpretation to compute row-major C = A @ B^T:
    // C^T (n x m, col-major) = B (n x k) * A^T (k x m)
    // B is row-major [n,k], seen by hipBLASLt as col-major [k,n], then TRANSA=T recovers [n,k].
    check_lt(hipblasLtMatrixLayoutCreate(&plan->layout_a, HIP_R_16BF, k, n, k), "layout A(B^T view) failed");
    check_lt(hipblasLtMatrixLayoutCreate(&plan->layout_b, HIP_R_16BF, k, m, k), "layout B(A^T) failed");
    check_lt(hipblasLtMatrixLayoutCreate(&plan->layout_c, HIP_R_16BF, n, m, n), "layout C(C^T) failed");

    hipblasLtMatmulPreference_t preference = nullptr;
    check_lt(hipblasLtMatmulPreferenceCreate(&preference), "preference create failed");
    check_lt(
        hipblasLtMatmulPreferenceSetAttribute(
            preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes,
            sizeof(kWorkspaceBytes)),
        "set workspace preference failed");

    hipblasLtMatmulHeuristicResult_t result;
    int returned = 0;
    check_lt(
        hipblasLtMatmulAlgoGetHeuristic(
            plan->handle,
            plan->matmul_desc,
            plan->layout_a,
            plan->layout_b,
            plan->layout_c,
            plan->layout_c,
            preference,
            1,
            &result,
            &returned),
        "heuristic query failed");
    check_lt(hipblasLtMatmulPreferenceDestroy(preference), "preference destroy failed");

    TORCH_CHECK(returned > 0, "No valid hipBLASLt algorithm found");
    plan->algo = result.algo;
    plan->workspace = torch::empty({static_cast<long long>(kWorkspaceBytes)}, d.options().dtype(torch::kUInt8));

    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto [it, inserted] = g_plans.emplace(key, plan);
        if (!inserted) {
            return it->second;
        }
    }
    return plan;
}

} // namespace

void hipblaslt_bf16_mm_out(torch::Tensor a, torch::Tensor b, torch::Tensor d) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && d.is_cuda(), "A, B, D must be CUDA/HIP tensors");
    TORCH_CHECK(
        a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16 && d.dtype() == torch::kBFloat16,
        "A, B, D must be BF16 tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && d.dim() == 2, "A, B, D must be 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(1), "K mismatch: A.shape[1] must equal B.shape[1] for A @ B^T");
    TORCH_CHECK(d.size(0) == a.size(0) && d.size(1) == b.size(0), "D shape mismatch: expected (A.shape[0], B.shape[0])");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && d.is_contiguous(), "A, B, D must be contiguous tensors");
    TORCH_CHECK(a.get_device() == b.get_device() && a.get_device() == d.get_device(), "A, B, D must be on the same device");

    auto plan = get_or_create_plan(a, b, d);

    static constexpr float kAlpha = 1.0f;
    static constexpr float kBeta = 0.0f;
    check_lt(
        hipblasLtMatmul(
            plan->handle,
            plan->matmul_desc,
            &kAlpha,
            b.data_ptr(),
            plan->layout_a,
            a.data_ptr(),
            plan->layout_b,
            &kBeta,
            d.data_ptr(),
            plan->layout_c,
            d.data_ptr(),
            plan->layout_c,
            &plan->algo,
            plan->workspace.data_ptr(),
            kWorkspaceBytes,
            c10::hip::getCurrentHIPStream()),
        "hipblasLtMatmul failed");
}
