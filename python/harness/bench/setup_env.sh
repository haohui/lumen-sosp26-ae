#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BENCHMARK_ROOT="${REPO_ROOT}/data/benchmarks"
GEMM_ROOT="${BENCHMARK_ROOT}/gemm"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PINNED_AITER_VER="0.1.10.post3"
PINNED_AITER_COMMIT="6a0e7b26ccf33164785531212cc2ec2cde0b9243"
PINNED_TORCH_VER="2.9.1+rocm7.1.1.git351ff442"
PINNED_TRITON_VER="3.5.1+rocm7.1.1.gita272dfa8"
PINNED_TORCH_HIP_VER="7.1.52802-26aae437f6"
PINNED_ROCM_VER="7.1.1"
PINNED_HIPBLASLT_VER="1.1.0"
PINNED_HIPBLASLT_REALNAME="libhipblaslt.so.1.1.70101"
PINNED_HIPBLASLT_SHA256="4d103e5573fcb1d3133d634c96c5c0a44232f15f76dcdb1195c8057e6b42a021"
PINNED_HIPKITTENS_CDNA3_COMMIT="7d58fa1026b4"

AITER_REPO_URL="${AITER_REPO_URL:-https://github.com/ROCm/aiter.git}"
HIPKITTENS_REPO_URL="${HIPKITTENS_REPO_URL:-https://github.com/HazyResearch/HipKittens.git}"
AITER_SRC_CACHE_ROOT="${AITER_SRC_CACHE_ROOT:-${REPO_ROOT}/logs/.setup_cache}"
HIPKITTENS_SRC_CACHE_ROOT="${HIPKITTENS_SRC_CACHE_ROOT:-${REPO_ROOT}/logs/.setup_cache}"
AITER_SRC_DIR="${AITER_SRC_CACHE_ROOT}/aiter_${PINNED_AITER_VER}"
HIPKITTENS_SRC_DIR="${HIPKITTENS_SRC_CACHE_ROOT}/hipkittens_cdna3_${PINNED_HIPKITTENS_CDNA3_COMMIT}"
MINI_DIR="${GEMM_ROOT}/08_hipketten/build_hipkittens_mini"
UNIFIED_DIR="${GEMM_ROOT}/08_hipketten/build_hipkittens_unified"
SIZES=(1024 2048 4096 8192 16384)

REQUIRE_AITER=0
STRICT_VERSIONS=0
SKIP_SMOKE=0

usage() {
  cat <<'USAGE'
Usage:
  bash python/harness/bench/setup_env.sh [options]
Options:
  --python <bin>       Python executable
  --require-aiter      Fail if pinned amd-aiter is unavailable
  --strict-versions    Fail on pinned stack mismatch (default: warn)
  --skip-smoke         Skip import smoke checks
  --reinstall-aiter    Accepted for compatibility (always reinstalling pinned aiter)
  -h, --help           Show help
USAGE
}

log() { printf '[setup_env] %s\n' "$*"; }
warn() { printf '[setup_env] WARN: %s\n' "$*" >&2; }
die() { printf '[setup_env] ERROR: %s\n' "$*" >&2; exit 1; }
maybe_fail() { [[ "${STRICT_VERSIONS}" -eq 1 ]] && die "$1" || warn "$1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --require-aiter) REQUIRE_AITER=1; shift ;;
    --strict-versions) STRICT_VERSIONS=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --reinstall-aiter) shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BIN}"
PIP=("${PYTHON_BIN}" -m pip --disable-pip-version-check --no-input)

aiter_state() {
  "${PYTHON_BIN}" - <<'PY'
try:
    import importlib.metadata as md
    print(md.version("amd-aiter"))
except Exception:
    try:
        import contextlib, io, pathlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import aiter
        print(f"importable:{pathlib.Path(getattr(aiter, '__file__', '')).as_posix()}")
    except Exception:
        print("")
PY
}

cleanup_aiter_pth() {
  "${PYTHON_BIN}" - <<'PY'
import glob, os, site
blocked = ("/workspace/aiter", "/data01/")
for d in site.getsitepackages():
    for p in glob.glob(os.path.join(d, "*.pth")):
        try:
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        kept = [ln for ln in txt.splitlines() if not any(b in ln for b in blocked)]
        if kept == txt.splitlines():
            continue
        if kept:
            open(p, "w", encoding="utf-8").write("\n".join(kept) + "\n")
            print(f"[setup_env] rewrote pth: {p}")
        else:
            os.remove(p)
            print(f"[setup_env] removed pth: {p}")
PY
}

install_pinned_aiter() {
  log "installing pinned amd-aiter ${PINNED_AITER_VER} @ ${PINNED_AITER_COMMIT}"
  cleanup_aiter_pth
  "${PIP[@]}" install pybind11 ninja packaging pandas psutil einops
  mkdir -p "${AITER_SRC_CACHE_ROOT}"
  if [[ ! -d "${AITER_SRC_DIR}/.git" ]]; then
    git clone --recurse-submodules "${AITER_REPO_URL}" "${AITER_SRC_DIR}"
  else
    git -C "${AITER_SRC_DIR}" fetch --tags --force "${AITER_REPO_URL}"
  fi
  git -C "${AITER_SRC_DIR}" checkout -f "${PINNED_AITER_COMMIT}"
  git -C "${AITER_SRC_DIR}" reset --hard "${PINNED_AITER_COMMIT}"
  git -C "${AITER_SRC_DIR}" submodule sync --recursive
  git -C "${AITER_SRC_DIR}" submodule update --init --recursive
  [[ "$(git -C "${AITER_SRC_DIR}" rev-parse HEAD)" == "${PINNED_AITER_COMMIT}" ]] || die "aiter commit mismatch"
  "${PIP[@]}" uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
  cleanup_aiter_pth
  "${PIP[@]}" install --upgrade --no-build-isolation "${AITER_SRC_DIR}" || {
    [[ "${REQUIRE_AITER}" -eq 1 ]] && die "failed to install required amd-aiter"
    warn "failed to install pinned amd-aiter"
  }
}

verify_python_stack() {
  env \
    PINNED_TORCH_VER="${PINNED_TORCH_VER}" \
    PINNED_TRITON_VER="${PINNED_TRITON_VER}" \
    PINNED_TORCH_HIP_VER="${PINNED_TORCH_HIP_VER}" \
    PINNED_AITER_VER="${PINNED_AITER_VER}" \
    STRICT_VERSIONS="${STRICT_VERSIONS}" \
    REQUIRE_AITER="${REQUIRE_AITER}" \
    "${PYTHON_BIN}" - <<'PY'
import contextlib, io, importlib.metadata as md, os, pathlib, sys, torch, triton
def chk(ok, msg):
    if ok: return
    if os.environ.get("STRICT_VERSIONS","0")=="1": raise SystemExit(f"[setup_env] ERROR: {msg}")
    print(f"[setup_env] WARN: {msg}", file=sys.stderr)
chk(torch.__version__ == os.environ["PINNED_TORCH_VER"], f"torch mismatch: expected {os.environ['PINNED_TORCH_VER']}, got {torch.__version__}")
chk(triton.__version__ == os.environ["PINNED_TRITON_VER"], f"triton mismatch: expected {os.environ['PINNED_TRITON_VER']}, got {triton.__version__}")
chk(getattr(torch.version, "hip", None) == os.environ["PINNED_TORCH_HIP_VER"], f"torch HIP mismatch: expected {os.environ['PINNED_TORCH_HIP_VER']}, got {getattr(torch.version,'hip',None)}")
try: aiter = md.version("amd-aiter")
except Exception:
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf): import aiter as _aiter
        aiter = f"importable:{pathlib.Path(getattr(_aiter, '__file__', '')).as_posix()}"
    except Exception: aiter = "missing"
if os.environ.get("REQUIRE_AITER","0") == "1" and aiter != os.environ["PINNED_AITER_VER"]:
    raise SystemExit(f"[setup_env] ERROR: amd-aiter mismatch: expected {os.environ['PINNED_AITER_VER']}, got {aiter}")
if aiter.startswith("importable:/workspace/") or aiter.startswith("importable:/data01/"):
    raise SystemExit(f"[setup_env] ERROR: external-source aiter path not allowed: {aiter}")
if os.environ.get("REQUIRE_AITER","0") != "1" and aiter == "missing":
    print("[setup_env] WARN: amd-aiter unavailable; AITER baselines may be skipped", file=sys.stderr)
PY
}

verify_rocm_stack() {
  local rocm_ver=""
  [[ -f /opt/rocm/.info/version ]] && rocm_ver="$(cat /opt/rocm/.info/version)"
  [[ -z "${rocm_ver}" && -f /opt/rocm-7.1.1/.info/version ]] && rocm_ver="$(cat /opt/rocm-7.1.1/.info/version)"
  [[ "${rocm_ver}" == "${PINNED_ROCM_VER}" ]] || maybe_fail "ROCm mismatch: expected ${PINNED_ROCM_VER}, got '${rocm_ver:-<missing>}'"
  local link=""
  for p in /opt/rocm/lib/libhipblaslt.so /opt/rocm-7.1.1/lib/libhipblaslt.so; do [[ -e "${p}" ]] && link="${p}" && break; done
  [[ -n "${link}" ]] || { maybe_fail "libhipblaslt.so not found"; return 0; }
  local real name sha prefix
  real="$(readlink -f "${link}")"; name="$(basename "${real}")"; sha="$(sha256sum "${real}" | cut -d' ' -f1)"; prefix="$(echo "${PINNED_HIPBLASLT_VER}" | awk -F. '{print $1 "." $2}')"
  [[ "${name}" == libhipblaslt.so.${prefix}.* ]] || maybe_fail "hipBLASLt API mismatch: expected ${PINNED_HIPBLASLT_VER}, got ${name}"
  [[ "${name}" == "${PINNED_HIPBLASLT_REALNAME}" ]] || maybe_fail "hipBLASLt soname mismatch: expected ${PINNED_HIPBLASLT_REALNAME}, got ${name}"
  [[ "${sha}" == "${PINNED_HIPBLASLT_SHA256}" ]] || maybe_fail "hipBLASLt sha mismatch: expected ${PINNED_HIPBLASLT_SHA256}, got ${sha}"
}

fetch_hipkittens_source() {
  mkdir -p "${HIPKITTENS_SRC_CACHE_ROOT}"
  if [[ ! -d "${HIPKITTENS_SRC_DIR}/.git" ]]; then rm -rf "${HIPKITTENS_SRC_DIR}"; git clone --filter=blob:none "${HIPKITTENS_REPO_URL}" "${HIPKITTENS_SRC_DIR}"; fi
  git -C "${HIPKITTENS_SRC_DIR}" rev-parse --verify "${PINNED_HIPKITTENS_CDNA3_COMMIT}^{commit}" >/dev/null 2>&1 || \
    git -C "${HIPKITTENS_SRC_DIR}" fetch --depth 1 origin "${PINNED_HIPKITTENS_CDNA3_COMMIT}" >/dev/null 2>&1 || \
    git -C "${HIPKITTENS_SRC_DIR}" fetch origin --tags --prune
  git -C "${HIPKITTENS_SRC_DIR}" checkout --detach "${PINNED_HIPKITTENS_CDNA3_COMMIT}"
  [[ "$(git -C "${HIPKITTENS_SRC_DIR}" rev-parse --short=12 HEAD)" == "${PINNED_HIPKITTENS_CDNA3_COMMIT}" ]] || die "HipKittens commit mismatch"
}

compile_hipkittens() {
  local src="$1" out="$2" pybind="$3"
  [[ -f "${src}" ]] || die "missing source: ${src}"
  hipcc "${src}" -O3 -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w \
    -I"${HIPKITTENS_SRC_DIR}/include" -I"${HIPKITTENS_SRC_DIR}/prototype" ${pybind} \
    -shared -fPIC -Rpass-analysis=kernel-resource-usage -I/opt/rocm/include/hip \
    -lpthread -ldl -lutil -lm -o "${out}"
}

rebuild_hipkittens_if_needed() {
  local ext need=0
  ext="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)"
  mkdir -p "${MINI_DIR}" "${UNIFIED_DIR}"
  for s in "${SIZES[@]}"; do
    local m="${MINI_DIR}/tk_kernel_${s}_mini${ext}" u="${UNIFIED_DIR}/tk_kernel_unified_${s}${ext}"
    [[ -f "${m}" ]] || need=1; [[ -f "${u}" ]] || need=1
    [[ -f "${m}" ]] && ldd "${m}" 2>/dev/null | grep -q "not found" && need=1 || true
    [[ -f "${u}" ]] && ldd "${u}" 2>/dev/null | grep -q "not found" && need=1 || true
  done
  [[ "${need}" -eq 1 ]] || return 0
  command -v hipcc >/dev/null 2>&1 || die "hipcc not found; cannot rebuild HipKittens modules"
  fetch_hipkittens_source
  local pybind
  pybind="$("${PYTHON_BIN}" -m pybind11 --includes)"
  for s in "${SIZES[@]}"; do compile_hipkittens "${MINI_DIR}/tk_kernel_${s}_mini__autogen.cpp" "${MINI_DIR}/tk_kernel_${s}_mini${ext}" "${pybind}"; done
  for s in "${SIZES[@]}"; do compile_hipkittens "${UNIFIED_DIR}/tk_kernel_unified_${s}__autogen.cpp" "${UNIFIED_DIR}/tk_kernel_unified_${s}${ext}" "${pybind}"; done
}

smoke_check() {
  env GEMM_ROOT="${GEMM_ROOT}" PINNED_AITER_VER="${PINNED_AITER_VER}" REQUIRE_AITER="${REQUIRE_AITER}" "${PYTHON_BIN}" - <<'PY'
import contextlib, io, importlib.metadata as md, importlib.util, os, pathlib, torch, triton
try: aiter = md.version("amd-aiter")
except Exception:
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf): import aiter as _aiter
        aiter = f"importable:{pathlib.Path(getattr(_aiter, '__file__', '')).as_posix()}"
    except Exception: aiter = "missing"
if os.environ.get("REQUIRE_AITER","0") == "1" and aiter != os.environ["PINNED_AITER_VER"]:
    raise SystemExit(f"[setup_env] ERROR: amd-aiter mismatch: {aiter}")
if aiter.startswith("importable:/workspace/") or aiter.startswith("importable:/data01/"):
    raise SystemExit(f"[setup_env] ERROR: external-source aiter path not allowed: {aiter}")
root = pathlib.Path(os.environ["GEMM_ROOT"])
for p in (root / "07_triton" / "best_kernel.py", root / "08_hipketten" / "best_kernel.py"):
    spec = importlib.util.spec_from_file_location("kb_mod", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    assert hasattr(mod, "kernel_function")
print("[smoke] torch", torch.__version__)
print("[smoke] triton", triton.__version__)
print("[smoke] aiter", aiter)
print("[smoke] import ok: 07_triton, 08_hipketten")
PY
}

log "benchmark_root=${BENCHMARK_ROOT}"
log "gemm_root=${GEMM_ROOT}"
log "python=$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable, sys.version.split()[0])
PY
)"

install_pinned_aiter
state="$(aiter_state)"
if [[ "${REQUIRE_AITER}" -eq 1 && "${state}" != "${PINNED_AITER_VER}" ]]; then die "required aiter unavailable (state='${state:-<none>}')"; fi
if [[ "${state}" == importable:/workspace/* || "${state}" == importable:/data01/* ]]; then die "aiter resolved to external path: ${state}"; fi
[[ -n "${state}" || "${REQUIRE_AITER}" -eq 1 ]] || warn "amd-aiter unavailable; AITER baselines may be skipped"

verify_python_stack
verify_rocm_stack
rebuild_hipkittens_if_needed
if [[ "${SKIP_SMOKE}" -eq 0 ]]; then log "running smoke checks"; smoke_check; fi
log "done"
