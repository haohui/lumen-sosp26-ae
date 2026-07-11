#!/usr/bin/env bash

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

log() {
  echo "[$(date -u +%F_%T)] $*"
}

ensure_path() {
  if [[ ! -e "$1" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[DRY-RUN] Missing path: $1" >&2
      return
    fi
    echo "Missing path: $1" >&2
    exit 1
  fi
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

prepare_prompt_suffix_file() {
  local task="$1"
  local baseline="$2"
  local out="$3"
  local common_src="${PROMPT_ROOT}/common/${task}.md"
  local baseline_src="${PROMPT_ROOT}/${baseline}/${task}.md"
  ensure_path "${common_src}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$(dirname "${out}")"
    cp "${common_src}" "${out}"
    if [[ -f "${baseline_src}" ]]; then
      printf '\n' >> "${out}"
      cat "${baseline_src}" >> "${out}"
    fi
  fi
  printf '%s\n' "${out}"
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

run_with_trace() {
  local name="$1"
  local cmd="$2"
  local out="${TRACE_ROOT}/${name}"
  log "baseline=${name}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY-RUN] ${cmd}"
    return
  fi
  mkdir -p "${out}"
  if [[ -n "${TRACE_WRAP:-}" ]]; then
    "${TRACE_WRAP}" "${out}" -- bash -lc "${cmd}"
  else
    bash -lc "${cmd}"
  fi
}

run_stage_command() {
  [[ "${STAGE_GENERATED}" == "1" ]] || return 0
  local cmd=(python3 "${STAGE_SCRIPT}" --task "${TASK}" "$@")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}
