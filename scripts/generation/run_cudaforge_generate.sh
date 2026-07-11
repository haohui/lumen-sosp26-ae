#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=common.sh
source "${REPO_ROOT}/scripts/generation/common.sh"

usage() {
  echo "Usage: run_cudaforge_generate.sh --task attention|gemm|moe [--reference-py PATH]"
}

stage_generated() {
  run_stage_command --run-tag "${RUN_TAG}" --trace-root "${TRACE_ROOT}" \
    --third-party-root "${THIRD_PARTY_ROOT}" --data-root "${DATA_BENCHMARK_ROOT}"
}

TASK="${TASK:-}"
REFERENCE_PY=""
ROUNDS="${ROUNDS:-10}"
RUN_TAG="${RUN_TAG:-}"
TRACE_ROOT="${TRACE_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-gpt-5.3-codex}"
SERVER_TYPE="${SERVER_TYPE:-openai}"
API_PROVIDER="${API_PROVIDER:-openai}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-7}"
GPU_ID="${GPU_ID:-0}"
CF_TOL="${CF_TOL:-1e-2}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${REPO_ROOT}/third_party}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
KEY_FILE="${KEY_FILE:-../key}"
DATA_BENCHMARK_ROOT="${DATA_BENCHMARK_ROOT:-data/benchmarks}"
STAGE_GENERATED="${STAGE_GENERATED:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --reference-py) REFERENCE_PY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

TASK="$(canonical_task "${TASK}")"
PROMPT_ROOT="${PROMPT_ROOT:-${REPO_ROOT}/scripts/generation/prompts}"
STAGE_SCRIPT="${REPO_ROOT}/scripts/generation/stage_cudaforge_output.py"
CUDAFORGE_ROOT="${CUDAFORGE_ROOT:-${THIRD_PARTY_ROOT}/CUDAForge}"
CUDAFORGE_RESOURCE_ROOT="${CUDAFORGE_RESOURCE_ROOT:-${REPO_ROOT}/datasets/inference/cudaforge/resources}"
BASELINE_PROMPT_ROOT="${BASELINE_PROMPT_ROOT:-${CUDAFORGE_RESOURCE_ROOT}/prompts}"
HIP_FEWSHOT_NEW_PATH="${HIP_FEWSHOT_NEW_PATH:-${BASELINE_PROMPT_ROOT}/few_shot_hip/model_new_ex_add.py}"
REF_PATH="$(task_reference_path "${TASK}")"
RUN_TAG="${RUN_TAG:-${TASK}_cudaforge_$(date -u +%Y%m%d_%H%M%S)}"
TRACE_ROOT="${TRACE_ROOT:-$(resolve_repo_path "logs/generation/$(task_data_dir "${TASK}")")/${RUN_TAG}}"
PROMPT_SUFFIX_FILE="$(prepare_prompt_suffix_file "${TASK}" "cudaforge" "${TRACE_ROOT}/02_cudaforge_prompt_suffix.txt")"

configure_api_provider
require_generation_key
ensure_path "${CUDAFORGE_ROOT}"
ensure_path "${REF_PATH}"
ensure_path "${HIP_FEWSHOT_NEW_PATH}"

run_with_trace "02_cudaforge_${TASK}_${SERVER_TYPE}" "
set -euo pipefail
cd '${CUDAFORGE_ROOT}'
if [[ -f '${ENV_FILE}' ]]; then set -a; source '${ENV_FILE}'; set +a; fi
export PYTHONPATH='${REPO_ROOT}'\${PYTHONPATH:+:\"\${PYTHONPATH}\"}
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES='${ROCR_VISIBLE_DEVICES}'
python3 main.py '${REF_PATH}' \
  --backend hip \
  --gpu MI300X \
  --hip-fewshot-new '${HIP_FEWSHOT_NEW_PATH}' \
  --server_type '${SERVER_TYPE}' \
  --model_name '${MODEL_NAME}' \
  --reasoning_effort '${REASONING_EFFORT}' \
  --prompt-suffix-file '${PROMPT_SUFFIX_FILE}' \
  --tol '${CF_TOL}' \
  --device '${GPU_ID}' \
  --round '${ROUNDS}' \
  --subproc_id 0 \
  --work_dir 'run_${TASK}_${RUN_TAG}'
"

stage_generated
