ARG BASE_IMAGE=kernel-benchmark-rocm:traffic_20260330
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PINNED_TORCH_VER=2.9.1+rocm7.1.1.lw.git351ff442
ARG PINNED_TRITON_VER=3.5.1+rocm7.1.1.gita272dfa8
ARG PINNED_AITER_VER=0.1.10.post3
ARG PINNED_AITER_COMMIT=6a0e7b26ccf33164785531212cc2ec2cde0b9243
ARG PINNED_HIPBLASLT_REALNAME=libhipblaslt.so.1.1.70101
ARG PINNED_HIPBLASLT_SHA256=4d103e5573fcb1d3133d634c96c5c0a44232f15f76dcdb1195c8057e6b42a021
ARG PINNED_HIPKITTENS_COMMIT=7d58fa1026b4

WORKDIR /workspace/lumen-sosp26-ae

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 --disable-pip-version-check --no-input install --upgrade pip \
 && pip3 --disable-pip-version-check --no-input install \
    einops \
    ninja \
    packaging \
    pandas \
    psutil \
    pybind11 \
 && rm -rf /tmp/aiter \
 && git clone --recurse-submodules https://github.com/ROCm/aiter.git /tmp/aiter \
 && git -C /tmp/aiter checkout -f "${PINNED_AITER_COMMIT}" \
 && git -C /tmp/aiter submodule sync --recursive \
 && git -C /tmp/aiter submodule update --init --recursive \
 && pip3 --disable-pip-version-check --no-input install --no-build-isolation /tmp/aiter \
 && mkdir -p .locks \
 && printf '%s\n' "${PINNED_AITER_COMMIT}" > .locks/aiter_commit.lock \
 && rm -rf /tmp/aiter \
 && pip3 --disable-pip-version-check --no-input cache purge

RUN PINNED_HIPKITTENS_COMMIT="${PINNED_HIPKITTENS_COMMIT}" python3 - <<'PY'
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

commit = os.environ.get("PINNED_HIPKITTENS_COMMIT", "7d58fa1026b4")
repo = Path("tmp/HipKittens")
if repo.exists():
    shutil.rmtree(repo)

subprocess.run(
    ["git", "clone", "--filter=blob:none", "https://github.com/HazyResearch/HipKittens.git", str(repo)],
    check=True,
)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", commit], check=True)

mini_dir = Path("data/benchmarks/gemm/08_hipketten/build_hipkittens_mini")
if mini_dir.exists():
    shutil.rmtree(mini_dir)
mini_dir.mkdir(parents=True, exist_ok=True)

ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
pybind_includes = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip().split()

hipcc = shutil.which("hipcc")
if not hipcc:
    raise SystemExit("hipcc not found in PATH")
hipcc_path = Path(hipcc).resolve()
hip_include = None
for root in (hipcc_path.parents[1], hipcc_path.parents[2]):
    for cand in (root / "include", root / "include" / "hip"):
        if (cand / "hip_bf16.h").exists():
            hip_include = cand
            break
    if hip_include is not None:
        break
if hip_include is None:
    for env_name in ("ROCM_HOME", "ROCM_PATH"):
        env_home = os.environ.get(env_name)
        if not env_home:
            continue
        for cand in (Path(env_home) / "include", Path(env_home) / "include" / "hip"):
            if (cand / "hip_bf16.h").exists():
                hip_include = cand
                break
        if hip_include is not None:
            break
if hip_include is None:
    raise SystemExit("cannot locate HIP include directory containing hip_bf16.h")

for size in (1024, 2048, 4096, 8192, 16384):
    src_in = repo / "analysis" / "bf16_gemm" / "mi325x" / f"kernel_{size}.cpp"
    if not src_in.exists():
        raise SystemExit(f"missing HipKittens source: {src_in}")
    src_out = mini_dir / f"tk_kernel_{size}_mini__autogen.cpp"
    out_so = mini_dir / f"tk_kernel_{size}_mini{ext}"

    text = src_in.read_text(encoding="utf-8")
    needle = "PYBIND11_MODULE(tk_kernel, m)"
    replacement = f"PYBIND11_MODULE(tk_kernel_{size}_mini, m)"
    if needle not in text:
        raise SystemExit(f"missing module macro in {src_in}")
    src_out.write_text(text.replace(needle, replacement), encoding="utf-8")

    cmd = [
        "hipcc",
        str(src_out),
        "-O3",
        "-DKITTENS_CDNA3",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-w",
        f"-I{hip_include}",
        f"-I{repo / 'include'}",
        f"-I{repo / 'prototype'}",
        *pybind_includes,
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        "-lpthread",
        "-ldl",
        "-lutil",
        "-lm",
        "-o",
        str(out_so),
    ]
    subprocess.run(cmd, check=True)

shutil.rmtree(repo)
PY

RUN python3 -c 'import importlib.metadata as m; assert m.version("torch")=="2.9.1+rocm7.1.1.lw.git351ff442"; assert m.version("triton")=="3.5.1+rocm7.1.1.gita272dfa8"; assert m.version("amd-aiter")=="0.1.10.post3"'

RUN test "$(cat .locks/aiter_commit.lock)" = "${PINNED_AITER_COMMIT}" \
 && HIPBLASLT_CANDIDATE="/opt/rocm/lib/libhipblaslt.so.1" \
 && if [ ! -e "${HIPBLASLT_CANDIDATE}" ]; then HIPBLASLT_CANDIDATE="/opt/rocm-7.1.1/lib/libhipblaslt.so.1"; fi \
 && test -e "${HIPBLASLT_CANDIDATE}" \
 && REAL_HIPBLASLT="$(readlink -f "${HIPBLASLT_CANDIDATE}")" \
 && test "$(basename "${REAL_HIPBLASLT}")" = "${PINNED_HIPBLASLT_REALNAME}" \
 && test "$(sha256sum "${REAL_HIPBLASLT}" | awk '{print $1}')" = "${PINNED_HIPBLASLT_SHA256}"

RUN for size in 1024 2048 4096 8192 16384; do \
      ls data/benchmarks/gemm/08_hipketten/build_hipkittens_mini/tk_kernel_${size}_mini*.so >/dev/null; \
   done \
 && for p in \
      data/benchmarks/gemm/01_kernelbench/best_kernel.py \
      data/benchmarks/gemm/02_cudaforge/best_kernel.py \
      data/benchmarks/gemm/03_kernelfalcon/best_kernel.py \
      data/benchmarks/gemm/04_ksearch/best_kernel.py \
      data/benchmarks/gemm/05_aiter/best_kernel.py \
      data/benchmarks/gemm/06_hipblaslt/run_hipblaslt.py \
      data/benchmarks/gemm/07_triton/best_kernel.py \
      data/benchmarks/gemm/08_hipketten/best_kernel.py \
      data/benchmarks/attn/01_kernelbench/best_kernel.py \
      data/benchmarks/attn/02_cudaforge/best_kernel.py \
      data/benchmarks/attn/03_kernelfalcon/best_kernel.py \
      data/benchmarks/attn/04_ksearch/best_kernel.py \
      data/benchmarks/attn/05_triton/best_kernel.py \
      data/benchmarks/attn/06_aiter/best_kernel.py \
      data/benchmarks/moe/01_kernelbench/best_kernel.py \
      data/benchmarks/moe/02_cudaforge/best_kernel.py \
      data/benchmarks/moe/03_kernelfalcon/best_kernel.py \
      data/benchmarks/moe/04_ksearch/best_kernel.py \
      data/benchmarks/moe/05_aiter/run_aiter.py; do \
      test -f "$p"; \
   done
