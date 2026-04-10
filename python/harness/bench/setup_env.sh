#!/usr/bin/env bash
set -euo pipefail

# Environment setup for benchmark perf scripts.
# - Pins AITER / Torch / Triton / ROCm / hipBLASLt versions
# - Builds HipKittens .so files from pinned cdna3 source when missing
# - Performs a minimal import/version smoke check

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BENCHMARK_ROOT="${REPO_ROOT}/data/benchmarks"
GEMM_ROOT="${BENCHMARK_ROOT}/gemm"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PINNED_AITER_VER="0.1.10.post3"
PINNED_TORCH_VER="2.9.1+rocm7.1.1.git351ff442"
PINNED_TRITON_VER="3.5.1+rocm7.1.1.gita272dfa8"
PINNED_TORCH_HIP_VER="7.1.52802-26aae437f6"
PINNED_ROCM_VER="7.1.1"
PINNED_HIPBLASLT_VER="1.1.0"
PINNED_HIPBLASLT_REALNAME="libhipblaslt.so.1.1.70101"
PINNED_HIPBLASLT_SHA256="4d103e5573fcb1d3133d634c96c5c0a44232f15f76dcdb1195c8057e6b42a021"
PINNED_HIPKITTENS_CDNA3_COMMIT="7d58fa1026b4"
PINNED_AITER_COMMIT="6a0e7b26ccf33164785531212cc2ec2cde0b9243"

REINSTALL_AITER=0
REQUIRE_AITER=0
STRICT_VERSIONS=0
SKIP_SMOKE=0
DRY_RUN=0
HIPKITTENS_REPO_URL="${HIPKITTENS_REPO_URL:-https://github.com/HazyResearch/HipKittens.git}"
HIPKITTENS_SRC_CACHE_ROOT="${HIPKITTENS_SRC_CACHE_ROOT:-${REPO_ROOT}/logs/.setup_cache}"
AITER_REPO_URL="${AITER_REPO_URL:-https://github.com/ROCm/aiter.git}"
AITER_SRC_CACHE_ROOT="${AITER_SRC_CACHE_ROOT:-${REPO_ROOT}/logs/.setup_cache}"
AITER_TAG="v${PINNED_AITER_VER}"
AITER_SRC_DIR="${AITER_SRC_CACHE_ROOT}/aiter_${PINNED_AITER_VER}"

MINI_DIR="${GEMM_ROOT}/08_hipketten/build_hipkittens_mini"
UNIFIED_DIR="${GEMM_ROOT}/08_hipketten/build_hipkittens_unified"
SIZES=(1024 2048 4096 8192 16384)

usage() {
  cat <<'USAGE'
Usage:
  bash python/harness/bench/setup_env.sh [options]

Options:
  --python <bin>                  Python executable (default: python3 or $PYTHON_BIN)
  --reinstall-aiter               Force reinstall amd-aiter==0.1.10.post3 (pinned)
  --require-aiter                 Fail setup if pinned amd-aiter is unavailable
  --strict-versions               Fail setup if pinned torch/triton/ROCm stack mismatches
  (AITER is installed from pinned source commit for reproducibility)
  (if HipKittens .so is missing, script auto-fetches pinned HipKittens source and recompiles)
  --skip-smoke                    Skip post-setup import/version checks
  --dry-run                       Print commands without executing
  -h, --help                      Show this help
USAGE
}

log() {
  printf '[setup_env] %s\n' "$*"
}

die() {
  echo "[setup_env] ERROR: $*" >&2
  exit 1
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --reinstall-aiter)
      REINSTALL_AITER=1
      shift
      ;;
    --require-aiter)
      REQUIRE_AITER=1
      shift
      ;;
    --strict-versions)
      STRICT_VERSIONS=1
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PIP_CMD=("${PYTHON_BIN}" -m pip)
COMMON_PIP_ARGS=(--disable-pip-version-check --no-input)

log "benchmark_root=${BENCHMARK_ROOT}"
log "gemm_root=${GEMM_ROOT}"
log "python=$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable, sys.version.split()[0])
PY
)"

get_aiter_state() {
  "${PYTHON_BIN}" - <<'PY'
try:
    import importlib.metadata as md
    print(md.version("amd-aiter"))
except Exception:
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import aiter
        aiter_path = getattr(aiter, "__file__", "")
        print(f"importable:{aiter_path}")
    except Exception:
        print("")
PY
}

cleanup_external_aiter_pth() {
  "${PYTHON_BIN}" - <<'PY'
import glob
import os
import site

target = "/workspace/aiter"
for d in site.getsitepackages():
    for p in glob.glob(os.path.join(d, "*.pth")):
        try:
            text = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if target not in text:
            continue
        lines = [ln for ln in text.splitlines() if target not in ln]
        if lines:
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[setup_env] rewrote pth: {p}")
        else:
            try:
                os.remove(p)
                print(f"[setup_env] removed pth: {p}")
            except Exception:
                pass
PY
}

install_pinned_aiter() {
  log "installing amd-aiter==${PINNED_AITER_VER} from pinned commit ${PINNED_AITER_COMMIT} (current='${current_aiter_ver:-<none>}')"
  cleanup_external_aiter_pth
  run_cmd "${PIP_CMD[@]}" install "${COMMON_PIP_ARGS[@]}" \
    pybind11 ninja packaging pandas psutil einops
  log "installing from source: ${AITER_REPO_URL} @ ${PINNED_AITER_COMMIT} (tag ${AITER_TAG})"
  run_cmd mkdir -p "${AITER_SRC_CACHE_ROOT}"
  if [[ ! -d "${AITER_SRC_DIR}/.git" ]]; then
    run_cmd git clone --recurse-submodules "${AITER_REPO_URL}" "${AITER_SRC_DIR}"
  else
    run_cmd git -C "${AITER_SRC_DIR}" fetch --tags --force "${AITER_REPO_URL}"
  fi
  run_cmd git -C "${AITER_SRC_DIR}" checkout -f "${PINNED_AITER_COMMIT}"
  run_cmd git -C "${AITER_SRC_DIR}" reset --hard "${PINNED_AITER_COMMIT}"
  # aiter build needs composable_kernel from submodule; missing it causes
  # module_moe_sorting JIT compile failure (moe_sorting_api.hpp not found).
  run_cmd git -C "${AITER_SRC_DIR}" submodule sync --recursive
  run_cmd git -C "${AITER_SRC_DIR}" submodule update --init --recursive
  local resolved_commit
  resolved_commit="$(git -C "${AITER_SRC_DIR}" rev-parse HEAD)"
  if [[ "${resolved_commit}" != "${PINNED_AITER_COMMIT}" ]]; then
    die "aiter source commit mismatch: resolved=${resolved_commit}, expected=${PINNED_AITER_COMMIT}"
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '; printf '%q ' "${PIP_CMD[@]}" uninstall "${COMMON_PIP_ARGS[@]}" -y aiter amd-aiter; printf '\n'
  else
    "${PIP_CMD[@]}" uninstall "${COMMON_PIP_ARGS[@]}" -y aiter amd-aiter >/dev/null 2>&1 || true
  fi
  cleanup_external_aiter_pth
  if run_cmd "${PIP_CMD[@]}" install "${COMMON_PIP_ARGS[@]}" \
    --upgrade --no-build-isolation "${AITER_SRC_DIR}"; then
    return 0
  fi
  return 1
}

current_aiter_ver="$(get_aiter_state)"

if [[ "${REINSTALL_AITER}" -eq 1 ]]; then
  if ! install_pinned_aiter; then
    if [[ "${REQUIRE_AITER}" -eq 1 ]]; then
      die "failed to install required amd-aiter==${PINNED_AITER_VER}"
    fi
    log "WARN: pinned amd-aiter install failed; continuing without required aiter"
  fi
elif [[ "${current_aiter_ver}" == "${PINNED_AITER_VER}" ]]; then
  log "amd-aiter version matches (${PINNED_AITER_VER}); enforcing pinned source commit ${PINNED_AITER_COMMIT}"
  if ! install_pinned_aiter; then
    if [[ "${REQUIRE_AITER}" -eq 1 ]]; then
      die "failed to enforce pinned source commit for amd-aiter==${PINNED_AITER_VER}"
    fi
    log "WARN: failed to enforce pinned source commit; existing amd-aiter may remain active"
  fi
elif [[ "${current_aiter_ver}" == importable:* ]]; then
  log "detected source/importable aiter (${current_aiter_ver#importable:}); trying pinned source install"
  if ! install_pinned_aiter; then
    if [[ "${REQUIRE_AITER}" -eq 1 ]]; then
      die "failed to replace source aiter with required amd-aiter==${PINNED_AITER_VER}"
    fi
    log "WARN: pinned amd-aiter install failed; source aiter may still be active"
  fi
else
  if ! install_pinned_aiter; then
    if [[ "${REQUIRE_AITER}" -eq 1 ]]; then
      die "failed to install required amd-aiter==${PINNED_AITER_VER}"
    fi
    log "WARN: pinned amd-aiter install failed; continuing without required aiter"
  fi
fi

current_aiter_ver="$(get_aiter_state)"
if [[ "${REQUIRE_AITER}" -eq 1 ]]; then
  if [[ "${current_aiter_ver}" != "${PINNED_AITER_VER}" ]]; then
    die "required aiter unavailable after setup (state='${current_aiter_ver:-<none>}')"
  fi
elif [[ -z "${current_aiter_ver}" ]]; then
  log "WARN: aiter unavailable; AITER baselines will be skipped by benchmark scripts"
fi
if [[ "${current_aiter_ver}" == importable:* ]]; then
  aiter_src="${current_aiter_ver#importable:}"
  if [[ "${aiter_src}" == /workspace/* || "${aiter_src}" == /data01/* ]]; then
    die "aiter resolved to external source path (${aiter_src}); this benchmark requires pinned wheel install"
  fi
fi

verify_pinned_python_stack() {
  run_cmd env \
    PINNED_TORCH_VER="${PINNED_TORCH_VER}" \
    PINNED_TRITON_VER="${PINNED_TRITON_VER}" \
    PINNED_AITER_VER="${PINNED_AITER_VER}" \
    PINNED_TORCH_HIP_VER="${PINNED_TORCH_HIP_VER}" \
    REQUIRE_AITER="${REQUIRE_AITER}" \
    STRICT_VERSIONS="${STRICT_VERSIONS}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import importlib.metadata as md
import torch
import triton
import pathlib
import sys

expected_torch = os.environ["PINNED_TORCH_VER"]
expected_triton = os.environ["PINNED_TRITON_VER"]
expected_aiter = os.environ["PINNED_AITER_VER"]
expected_hip = os.environ["PINNED_TORCH_HIP_VER"]
require_aiter = os.environ.get("REQUIRE_AITER", "0") == "1"
strict_versions = os.environ.get("STRICT_VERSIONS", "0") == "1"

actual_torch = torch.__version__
actual_triton = triton.__version__
actual_hip = getattr(torch.version, "hip", None)

actual_aiter = None
try:
    actual_aiter = md.version("amd-aiter")
except Exception:
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import aiter
        path = pathlib.Path(getattr(aiter, "__file__", "")).as_posix()
        actual_aiter = f"importable:{path}"
    except Exception:
        actual_aiter = "missing"

if actual_torch != expected_torch:
    msg = f"torch version mismatch: expected {expected_torch}, got {actual_torch}"
    if strict_versions:
        raise SystemExit(msg)
    print(f"[setup_env] WARN: {msg}", file=sys.stderr)
if actual_triton != expected_triton:
    msg = f"triton version mismatch: expected {expected_triton}, got {actual_triton}"
    if strict_versions:
        raise SystemExit(msg)
    print(f"[setup_env] WARN: {msg}", file=sys.stderr)
if require_aiter:
    if actual_aiter != expected_aiter:
        raise SystemExit(f"amd-aiter version mismatch: expected {expected_aiter}, got {actual_aiter}")
elif actual_aiter == "missing":
    print("[setup_env] WARN: amd-aiter unavailable; AITER baselines will be skipped", file=sys.stderr)
elif str(actual_aiter).startswith("importable:/workspace/") or str(actual_aiter).startswith("importable:/data01/"):
    raise SystemExit(f"setup_env disallows external-source aiter path: {actual_aiter}")
if actual_hip != expected_hip:
    msg = f"torch HIP runtime mismatch: expected {expected_hip}, got {actual_hip}"
    if strict_versions:
        raise SystemExit(msg)
    print(f"[setup_env] WARN: {msg}", file=sys.stderr)
PY
}

verify_pinned_rocm_stack() {
  local rocm_ver=""
  if [[ -f /opt/rocm/.info/version ]]; then
    rocm_ver="$(cat /opt/rocm/.info/version)"
  elif [[ -f /opt/rocm-7.1.1/.info/version ]]; then
    rocm_ver="$(cat /opt/rocm-7.1.1/.info/version)"
  fi
  if [[ "${rocm_ver}" != "${PINNED_ROCM_VER}" ]]; then
    if [[ "${STRICT_VERSIONS}" -eq 1 ]]; then
      die "ROCm version mismatch: expected ${PINNED_ROCM_VER}, got '${rocm_ver:-<missing>}'"
    fi
    log "WARN: ROCm version mismatch: expected ${PINNED_ROCM_VER}, got '${rocm_ver:-<missing>}'"
  fi

  local hipblaslt_link=""
  for p in /opt/rocm/lib/libhipblaslt.so /opt/rocm-7.1.1/lib/libhipblaslt.so; do
    if [[ -e "${p}" ]]; then
      hipblaslt_link="${p}"
      break
    fi
  done
  if [[ -z "${hipblaslt_link}" ]]; then
    if [[ "${STRICT_VERSIONS}" -eq 1 ]]; then
      die "libhipblaslt.so not found under /opt/rocm*/lib"
    fi
    log "WARN: libhipblaslt.so not found under /opt/rocm*/lib"
    return 0
  fi

  local hipblaslt_real
  hipblaslt_real="$(readlink -f "${hipblaslt_link}")"
  local hipblaslt_real_name
  hipblaslt_real_name="$(basename "${hipblaslt_real}")"

  local hipblaslt_ver_prefix
  hipblaslt_ver_prefix="$(echo "${PINNED_HIPBLASLT_VER}" | awk -F. '{print $1 "." $2}')"
  if [[ "${hipblaslt_real_name}" != "libhipblaslt.so.${hipblaslt_ver_prefix}."* ]]; then
    if [[ "${STRICT_VERSIONS}" -eq 1 ]]; then
      die "hipBLASLt version mismatch: expected API ${PINNED_HIPBLASLT_VER}, got ${hipblaslt_real_name}"
    fi
    log "WARN: hipBLASLt version mismatch: expected API ${PINNED_HIPBLASLT_VER}, got ${hipblaslt_real_name}"
  fi
  if [[ "${hipblaslt_real_name}" != "${PINNED_HIPBLASLT_REALNAME}" ]]; then
    if [[ "${STRICT_VERSIONS}" -eq 1 ]]; then
      die "hipBLASLt soname mismatch: expected ${PINNED_HIPBLASLT_REALNAME}, got ${hipblaslt_real_name}"
    fi
    log "WARN: hipBLASLt soname mismatch: expected ${PINNED_HIPBLASLT_REALNAME}, got ${hipblaslt_real_name}"
  fi
  local hipblaslt_sha
  hipblaslt_sha="$(sha256sum "${hipblaslt_real}" | cut -d' ' -f1)"
  if [[ "${hipblaslt_sha}" != "${PINNED_HIPBLASLT_SHA256}" ]]; then
    if [[ "${STRICT_VERSIONS}" -eq 1 ]]; then
      die "hipBLASLt hash mismatch: expected ${PINNED_HIPBLASLT_SHA256}, got ${hipblaslt_sha}"
    fi
    log "WARN: hipBLASLt hash mismatch: expected ${PINNED_HIPBLASLT_SHA256}, got ${hipblaslt_sha}"
  fi
}

verify_pinned_python_stack
verify_pinned_rocm_stack

fetch_pinned_hipkittens_source() {
  command -v git >/dev/null 2>&1 || die "git not found; cannot fetch HipKittens source"
  local dst="${HIPKITTENS_SRC_CACHE_ROOT%/}/hipkittens_cdna3_${PINNED_HIPKITTENS_CDNA3_COMMIT}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%s\n' "${dst}"
    return 0
  fi
  if [[ ! -d "${dst}/.git" ]]; then
    run_cmd rm -rf "${dst}"
    run_cmd git clone --filter=blob:none "${HIPKITTENS_REPO_URL}" "${dst}"
  fi

  if ! git -C "${dst}" rev-parse --verify "${PINNED_HIPKITTENS_CDNA3_COMMIT}^{commit}" >/dev/null 2>&1; then
    if ! git -C "${dst}" fetch --depth 1 origin "${PINNED_HIPKITTENS_CDNA3_COMMIT}" >/dev/null 2>&1; then
      run_cmd git -C "${dst}" fetch origin --tags --prune
    fi
  fi
  run_cmd git -C "${dst}" checkout --detach "${PINNED_HIPKITTENS_CDNA3_COMMIT}"

  local head12
  head12="$(git -C "${dst}" rev-parse --short=12 HEAD)"
  if [[ "${head12}" != "${PINNED_HIPKITTENS_CDNA3_COMMIT}" ]]; then
    die "fetched HipKittens commit mismatch: expected ${PINNED_HIPKITTENS_CDNA3_COMMIT}, got ${head12}"
  fi
  printf '%s\n' "${dst}"
}

compile_hipkittens_module() {
  local src_cpp="$1"
  local out_so="$2"
  local tk_root="$3"

  [[ -f "${src_cpp}" ]] || die "missing HipKittens source for rebuild: ${src_cpp}"
  command -v hipcc >/dev/null 2>&1 || die "hipcc not found; cannot rebuild HipKittens modules"
  mkdir -p "$(dirname "${out_so}")"

  local pybind_flags_raw
  pybind_flags_raw="$("${PYTHON_BIN}" -m pybind11 --includes)"
  local -a pybind_flags=()
  read -r -a pybind_flags <<<"${pybind_flags_raw}"

  run_cmd hipcc \
    "${src_cpp}" \
    -O3 \
    -DKITTENS_CDNA3 \
    --offload-arch=gfx942 \
    -std=c++20 \
    -w \
    "-I${tk_root}/include" \
    "-I${tk_root}/prototype" \
    "${pybind_flags[@]}" \
    -shared \
    -fPIC \
    -Rpass-analysis=kernel-resource-usage \
    -I/opt/rocm/include/hip \
    -lpthread \
    -ldl \
    -lutil \
    -lm \
    -o "${out_so}"
}

hipkittens_so_needs_rebuild() {
  local so="$1"
  [[ -f "${so}" ]] || return 0
  command -v ldd >/dev/null 2>&1 || return 1
  if ldd "${so}" 2>/dev/null | grep -q "not found"; then
    return 0
  fi
  return 1
}

rebuild_missing_hipkittens_modules() {
  local ext_suffix="$1"
  local need_rebuild=0
  for s in "${SIZES[@]}"; do
    local mini_so="${MINI_DIR}/tk_kernel_${s}_mini${ext_suffix}"
    if [[ ! -f "${mini_so}" ]] || hipkittens_so_needs_rebuild "${mini_so}"; then
      need_rebuild=1
    fi
  done
  for s in "${SIZES[@]}"; do
    local uni_so="${UNIFIED_DIR}/tk_kernel_unified_${s}${ext_suffix}"
    if [[ ! -f "${uni_so}" ]] || hipkittens_so_needs_rebuild "${uni_so}"; then
      need_rebuild=1
    fi
  done
  [[ "${need_rebuild}" -eq 1 ]] || return 0

  log "missing/incompatible HipKittens modules detected; rebuilding from repository __autogen.cpp sources"
  local tk_root
  tk_root="$(fetch_pinned_hipkittens_source)"
  for s in "${SIZES[@]}"; do
    local mini_out="${MINI_DIR}/tk_kernel_${s}_mini${ext_suffix}"
    if [[ ! -f "${mini_out}" ]] || hipkittens_so_needs_rebuild "${mini_out}"; then
      compile_hipkittens_module "${MINI_DIR}/tk_kernel_${s}_mini__autogen.cpp" "${mini_out}" "${tk_root}"
    fi
  done
  for s in "${SIZES[@]}"; do
    local uni_out="${UNIFIED_DIR}/tk_kernel_unified_${s}${ext_suffix}"
    if [[ ! -f "${uni_out}" ]] || hipkittens_so_needs_rebuild "${uni_out}"; then
      compile_hipkittens_module "${UNIFIED_DIR}/tk_kernel_unified_${s}__autogen.cpp" "${uni_out}" "${tk_root}"
    fi
  done
}

verify_required_modules() {
  local ext_suffix="$1"
  local errors=0

  for s in "${SIZES[@]}"; do
    if [[ ! -f "${MINI_DIR}/tk_kernel_${s}_mini${ext_suffix}" ]]; then
      echo "missing mini module: ${MINI_DIR}/tk_kernel_${s}_mini${ext_suffix}" >&2
      errors=1
    fi
  done

  for s in "${SIZES[@]}"; do
    if [[ ! -f "${UNIFIED_DIR}/tk_kernel_unified_${s}${ext_suffix}" ]]; then
      echo "missing unified module: ${UNIFIED_DIR}/tk_kernel_unified_${s}${ext_suffix}" >&2
      errors=1
    fi
  done

  if [[ "${errors}" -ne 0 ]]; then
    echo "HipKittens modules are incomplete under data/benchmarks/gemm." >&2
    echo "setup_env.sh will rebuild missing modules from pinned HipKittens source." >&2
    return 1
  fi
  return 0
}

ext_suffix="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)"

mkdir -p "${MINI_DIR}" "${UNIFIED_DIR}"
rebuild_missing_hipkittens_modules "${ext_suffix}"
verify_required_modules "${ext_suffix}"

if [[ "${SKIP_SMOKE}" -eq 0 ]]; then
  log "running smoke checks"
  run_cmd env GEMM_ROOT_ENV="${GEMM_ROOT}" PINNED_AITER_VER="${PINNED_AITER_VER}" REQUIRE_AITER="${REQUIRE_AITER}" "${PYTHON_BIN}" - <<'PY'
import importlib.util
import pathlib
import importlib.metadata as md
import os

expected_aiter = os.environ["PINNED_AITER_VER"]
require_aiter = os.environ.get("REQUIRE_AITER", "0") == "1"
actual_aiter = None
try:
    actual_aiter = md.version("amd-aiter")
except Exception:
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import aiter
        actual_aiter = f"importable:{pathlib.Path(getattr(aiter, '__file__', '')).as_posix()}"
    except Exception:
        actual_aiter = "missing"
if require_aiter and actual_aiter != expected_aiter:
    raise AssertionError(f"amd-aiter mismatch: expected {expected_aiter}, got {actual_aiter}")
if str(actual_aiter).startswith("importable:/workspace/") or str(actual_aiter).startswith("importable:/data01/"):
    raise AssertionError(f"external-source aiter path is not allowed: {actual_aiter}")

import torch
import triton

print("[smoke] torch", torch.__version__)
print("[smoke] triton", triton.__version__)
print("[smoke] aiter", actual_aiter)

root = pathlib.Path(os.environ["GEMM_ROOT_ENV"]).resolve()
p7 = root / "07_triton" / "best_kernel.py"
p8 = root / "08_hipketten" / "best_kernel.py"
for tag, path in (("07_triton", p7), ("08_hipketten", p8)):
    spec = importlib.util.spec_from_file_location(f"kb_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    assert hasattr(mod, "kernel_function")
    print(f"[smoke] import ok: {tag}")
PY
fi

log "done"
