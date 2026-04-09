#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-substrate_rocm711_wheatopt}"
HIP_VISIBLE_DEVICES_VAL="${2:-7}"

REPO_URL="${REPO_URL:-https://github.com/HazyResearch/HipKittens.git}"
COMMIT_SHA="${COMMIT_SHA:-4d15d8e92dfc65b6b33c36ad8b6a7e883c5f7245}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace/kernel_benchmark}"
OUT_DIR="${OUT_DIR:-/data01/home/daifeng/kernel_benchmark/opt_kernel/gemm-openai/07_triton/results}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

mkdir -p "${OUT_DIR}"

commit_short="${COMMIT_SHA:0:7}"
gpu_label="${HIP_VISIBLE_DEVICES_VAL//,/+}"
run_date="$(date -u +%Y%m%d)"
log_name="hipkittens_triton_gemm_v01_${commit_short}_gpu${gpu_label}_${run_date}.log"
host_log_path="${OUT_DIR}/${log_name}"
inner_log_path="/tmp/${log_name}"
tmp_root="/tmp/hipkittens_${commit_short}_test"

if ! docker ps --format '{{.Names}}' | rg -xq "${CONTAINER_NAME}"; then
  echo "[error] container not running: ${CONTAINER_NAME}" >&2
  exit 1
fi

docker exec \
  -e HK_TMP_ROOT="${tmp_root}" \
  -e HK_REPO_URL="${REPO_URL}" \
  -e HK_COMMIT_SHA="${COMMIT_SHA}" \
  -e HK_CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT}" \
  -e HK_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  -e HK_INNER_LOG_PATH="${inner_log_path}" \
  -e MPLBACKEND="Agg" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES_VAL}" \
  "${CONTAINER_NAME}" \
  bash -lc '
set -euo pipefail

if [ ! -e /dev/kfd ]; then
  echo "[error] /dev/kfd is missing in container; GPU runtime is unavailable." >&2
  exit 2
fi

python3 - <<"PY"
import torch
if not torch.cuda.is_available():
    raise SystemExit("[error] torch.cuda.is_available() is False in container")
print(f"[info] torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; device_count={torch.cuda.device_count()}")
PY

rm -rf "${HK_TMP_ROOT}"
git clone --no-checkout --depth 1 "${HK_REPO_URL}" "${HK_TMP_ROOT}"
cd "${HK_TMP_ROOT}"
git fetch --depth 1 origin "${HK_COMMIT_SHA}"
git checkout --detach "${HK_COMMIT_SHA}"
git show -s --format="[info] commit=%H %ci %s" HEAD

cd "${HK_CONTAINER_REPO_ROOT}"
timeout "${HK_TIMEOUT_SECONDS}" \
  python3 "${HK_TMP_ROOT}/analysis/baselines/gemm/triton_gemm_v01.py" \
  | tee "${HK_INNER_LOG_PATH}"
'

docker cp "${CONTAINER_NAME}:${inner_log_path}" "${host_log_path}"

echo "[done] log saved: ${host_log_path}"
