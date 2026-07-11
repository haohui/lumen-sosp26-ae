#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  echo "Usage: run_cudaforge_generate.sh --task attention|gemm|moe [--reference-py PATH]"
}

die() {
  echo "$*" >&2
  exit 2
}

resolve_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

ensure_path() {
  if [[ -e "$1" ]]; then
    return
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY-RUN] Missing path: $1" >&2
    return
  fi
  echo "Missing path: $1" >&2
  exit 1
}

canonical_task() {
  case "$1" in
    attention|attn) printf '%s\n' "attention" ;;
    gemm|moe) printf '%s\n' "$1" ;;
    *) die "Invalid task: $1 (expected: attention|gemm|moe)" ;;
  esac
}

task_data_dir() {
  case "$(canonical_task "$1")" in
    attention) printf '%s\n' "attn" ;;
    gemm) printf '%s\n' "gemm" ;;
    moe) printf '%s\n' "moe" ;;
  esac
}

task_reference_path() {
  local task ref
  task="$(canonical_task "$1")"
  if [[ -n "${REFERENCE_PY:-}" ]]; then
    ref="${REFERENCE_PY}"
  else
    case "${task}" in
      attention) ref="datasets/inference/kernelbench/2_attention.py" ;;
      gemm) ref="datasets/inference/kernelbench/1_gemm.py" ;;
      moe) ref="datasets/inference/kernelbench/3_fused_moe.py" ;;
    esac
  fi
  resolve_repo_path "${ref}"
}

load_api_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
  fi

  local key_file first
  key_file="$(resolve_repo_path "${KEY_FILE}")"
  if [[ -z "${OPENAI_API_KEY:-}" && -f "${key_file}" ]]; then
    first="$(sed -n '1p' "${key_file}" | tr -d '\r')"
    first="${first#OPENAI_API_KEY=}"
    first="${first#export OPENAI_API_KEY=}"
    export OPENAI_API_KEY="${first}"
  fi
}

configure_api_provider() {
  load_api_env
  case "${API_PROVIDER:-openai}" in
    openai)
      OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
      KB_MODEL_NAME="${MODEL_NAME}"
      ;;
    qwen)
      OPENAI_API_KEY="${QWEN_API_KEY:-}"
      OPENAI_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
      KB_MODEL_NAME="openai/${MODEL_NAME}"
      ;;
    *) die "Unsupported API_PROVIDER=${API_PROVIDER} (expected: openai|qwen)" ;;
  esac
  export OPENAI_API_KEY OPENAI_BASE_URL KB_MODEL_NAME
}

require_generation_key() {
  if [[ "${DRY_RUN:-0}" != "1" && "${SERVER_TYPE}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "API key is not set. Configure ENV_FILE=${ENV_FILE}, KEY_FILE=${KEY_FILE}, or export it before running." >&2
    exit 2
  fi
}

prepare_prompt_suffix_file() {
  local out="$1"
  local common_src="${PROMPT_ROOT}/common/${TASK}.md"
  local baseline_common_src="${BASELINE_PROMPT_ROOT}/common.md"
  local baseline_src="${BASELINE_PROMPT_ROOT}/${TASK}.md"

  ensure_path "${common_src}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$(dirname "${out}")"
    cp "${common_src}" "${out}"
    if [[ -f "${baseline_common_src}" ]]; then
      printf '\n' >> "${out}"
      cat "${baseline_common_src}" >> "${out}"
    fi
    if [[ -f "${baseline_src}" ]]; then
      printf '\n' >> "${out}"
      cat "${baseline_src}" >> "${out}"
    fi
  fi
  printf '%s\n' "${out}"
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
REF_PATH="$(task_reference_path "${TASK}")"
RUN_TAG="${RUN_TAG:-${TASK}_cudaforge_$(date -u +%Y%m%d_%H%M%S)}"
TRACE_ROOT="${TRACE_ROOT:-$(resolve_repo_path "logs/generation/$(task_data_dir "${TASK}")")/${RUN_TAG}}"
PROMPT_SUFFIX_FILE="$(prepare_prompt_suffix_file "${TRACE_ROOT}/02_cudaforge_prompt_suffix.txt")"

configure_api_provider
require_generation_key
ensure_path "${CUDAFORGE_ROOT}"
ensure_path "${REF_PATH}"
ensure_path "${CUDAFORGE_RESOURCE_ROOT}"

TRACE_DIR="${TRACE_ROOT}/02_cudaforge_${TASK}_${SERVER_TYPE}"
RUN_CMD="
set -euo pipefail
cd '${CUDAFORGE_ROOT}'
if [[ -f '${ENV_FILE}' ]]; then set -a; source '${ENV_FILE}'; set +a; fi
export PYTHONPATH='${REPO_ROOT}'\${PYTHONPATH:+:\"\${PYTHONPATH}\"}
export CUDAFORGE_PROMPT_PATH_OVERRIDE='${CUDAFORGE_RESOURCE_ROOT}'
export CUDAFORGE_PROMPT_SUFFIX_FILE='${PROMPT_SUFFIX_FILE}'
export CUDAFORGE_REASONING_EFFORT='${REASONING_EFFORT}'
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES='${ROCR_VISIBLE_DEVICES}'
python3 main.py '${REF_PATH}' \
  --backend hip \
  --gpu MI300X \
  --server_type '${SERVER_TYPE}' \
  --model_name '${MODEL_NAME}' \
  --tol '${CF_TOL}' \
  --device '${GPU_ID}' \
  --round '${ROUNDS}' \
  --subproc_id 0 \
  --work_dir 'run_${TASK}_${RUN_TAG}'
"

echo "[$(date -u +%F_%T)] baseline=02_cudaforge_${TASK}_${SERVER_TYPE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] ${RUN_CMD}"
else
  mkdir -p "${TRACE_DIR}"
  if [[ -n "${TRACE_WRAP:-}" ]]; then
    "${TRACE_WRAP}" "${TRACE_DIR}" -- bash -lc "${RUN_CMD}"
  else
    bash -lc "${RUN_CMD}"
  fi
fi

if [[ "${STAGE_GENERATED}" == "1" ]]; then
  STAGE_CMD=(python3 "${STAGE_SCRIPT}" --task "${TASK}" \
    --trace-root "${TRACE_ROOT}" --third-party-root "${THIRD_PARTY_ROOT}" \
    --data-root "${DATA_BENCHMARK_ROOT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] ${STAGE_CMD[*]}"
  else
    "${STAGE_CMD[@]}"
  fi
fi
