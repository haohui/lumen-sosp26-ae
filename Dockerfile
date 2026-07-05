ARG BASE_IMAGE=rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PINNED_TORCH_VER=2.9.1+rocm7.1.1.lw.git351ff442
ARG PINNED_TRITON_VER=3.5.1+rocm7.1.1.gita272dfa8
ARG PINNED_AITER_VER=0.1.10.post3
ARG PINNED_HIPBLASLT_REALNAME=libhipblaslt.so.1.1.70101
ARG PINNED_HIPBLASLT_SHA256=b31653aac55665d1610d9db318e6d1b3940394e53bb2573419ead8c7c2f99e61

WORKDIR lumen-sosp26-ae
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASHINFER_CACHE_DIR=.cache/flashinfer

COPY . .

# Keep only runtime deps missing from base image and install pinned AITER from source.
RUN python3 -m pip --disable-pip-version-check --no-input install --no-cache-dir \
    "git+https://github.com/ROCm/aiter.git@v0.1.10.post3" \
    einops \
    "litellm[proxy]==1.90.3" \
    "omegaconf==2.3.0" \
    "openai==2.44.0" \
    "apache-tvm-ffi==0.1.2" \
    "pydantic>=2.0.0" \
    psutil \
    "pydra-config==0.0.17.post1" \
    pybind11 \
    "safetensors>=0.5.0" \
 && ln -sf libhipblaslt.so.1.1.70101 /opt/rocm-7.1.1/lib/libhipblaslt.so.1 \
 && ln -sf libhipblaslt.so.1 /opt/rocm-7.1.1/lib/libhipblaslt.so \
 && ln -sf /opt/rocm/lib/libamdhip64.so.7.1.70101 /opt/rocm/lib/libamdhip64.so.6 \
 && ln -sf /opt/rocm/lib/libamdhip64.so /opt/venv/lib/libamdhip64.so \
 && ln -sf /opt/rocm/lib/libamdhip64.so.6 /opt/venv/lib/libamdhip64.so.6 \
 && ln -sf /opt/rocm/bin/hipcc /opt/venv/bin/hipcc \
 && ln -sf /opt/rocm/bin/amdclang++ /opt/venv/bin/amdclang++

RUN python3 -m pip --disable-pip-version-check --no-input install --no-cache-dir --no-deps \
    "git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git@92679cb2e576786a2710574c977a5317e17cf7c1"

RUN python3 -m pip --disable-pip-version-check --no-input install --no-cache-dir --no-index --find-links third_party/wheels \
    "amd-flashinfer==0.3.1+amd.1" \
 && mkdir -p .cache/flashinfer .aiter \
 && chmod -R a+rwX .cache .aiter

# Keep mitmproxy isolated because mitmproxy 10.x and OpenAI 2.x have
# incompatible typing-extensions constraints on Python 3.10.
RUN python3 -m venv /opt/mitmproxy-venv \
 && /opt/mitmproxy-venv/bin/python -m pip --disable-pip-version-check --no-input install --no-cache-dir \
    "mitmproxy==10.4.2" \
 && ln -sf /opt/mitmproxy-venv/bin/mitmweb /usr/local/bin/mitmweb \
 && ln -sf /opt/mitmproxy-venv/bin/mitmdump /usr/local/bin/mitmdump

RUN python3 - <<'PY'
import importlib.metadata as md
import shutil
import torch
import triton
import aiter
import openai
import litellm
import pydra
import omegaconf
import flashinfer
import flashinfer_bench

assert md.version("torch") == "2.9.1+rocm7.1.1.lw.git351ff442"
assert md.version("triton") == "3.5.1+rocm7.1.1.gita272dfa8"
assert md.version("amd-aiter") == "0.1.10.post3"
assert md.version("amd-flashinfer") == "0.3.1+amd.1"
assert "g92679cb2e" in md.version("flashinfer-bench")
assert shutil.which("mitmweb"), "mitmweb not found"
assert shutil.which("mitmdump"), "mitmdump not found"
assert torch.version.hip == "7.1.52802-26aae437f6"
_ = triton.__version__
_ = aiter.__file__
_ = flashinfer.__file__
_ = flashinfer_bench.__file__
print("version_check_ok")
PY

RUN mkdir -p .cache/flashinfer .aiter \
 && chmod -R a+rwX .cache .aiter

RUN HIPBLASLT_CANDIDATE="/opt/rocm/lib/libhipblaslt.so.1" \
 && if [ ! -e "${HIPBLASLT_CANDIDATE}" ]; then HIPBLASLT_CANDIDATE="/opt/rocm-7.1.1/lib/libhipblaslt.so.1"; fi \
 && test -e "${HIPBLASLT_CANDIDATE}" \
 && REAL_HIPBLASLT="$(readlink -f "${HIPBLASLT_CANDIDATE}")" \
 && test "$(basename "${REAL_HIPBLASLT}")" = "${PINNED_HIPBLASLT_REALNAME}" \
 && test "$(sha256sum "${REAL_HIPBLASLT}" | awk '{print $1}')" = "${PINNED_HIPBLASLT_SHA256}"

RUN python3 - <<'PY'
import shutil
import subprocess
import sysconfig
from pathlib import Path

repo = Path("third_party/HipKittens")
if not repo.exists():
    raise SystemExit(f"missing {repo}")

mini_dir = Path("data/benchmarks/gemm/08_hipketten/build_hipkittens_mini")
if mini_dir.exists():
    shutil.rmtree(mini_dir)
mini_dir.mkdir(parents=True, exist_ok=True)

ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
pybind_includes = subprocess.check_output(
    ["python3", "-m", "pybind11", "--includes"], text=True
).strip().split()

hipcc = shutil.which("hipcc")
if not hipcc:
    raise SystemExit("hipcc not found in PATH")

hipcc_path = Path(hipcc).resolve()
hip_include = None
for root in (Path("/opt/rocm"), Path("/opt/rocm-7.1.1"), hipcc_path.parents[1]):
    for cand in (root / "include", root / "include" / "hip"):
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
    if needle not in text:
        raise SystemExit(f"missing module macro in {src_in}")
    src_out.write_text(
        text.replace(needle, f"PYBIND11_MODULE(tk_kernel_{size}_mini, m)"),
        encoding="utf-8",
    )

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

print("hipkittens_mini_build_ok")
PY

RUN python3 - <<'PY'
from pathlib import Path
from sys import version_info

base = Path("data/benchmarks/gemm/08_hipketten/build_hipkittens_mini")
assert base.exists(), f"missing {base}"
tag = f"cpython-{version_info.major}{version_info.minor}"
for size in (1024, 2048, 4096, 8192, 16384):
    matches = sorted(base.glob(f"tk_kernel_{size}_mini*.so"))
    assert matches, f"missing HipKittens mini so for size={size}"
    assert any(tag in p.name for p in matches), f"no {tag} so for size={size}: {[p.name for p in matches]}"
print("hipkittens_artifacts_ok")
PY

RUN for p in \
      python/harness/bench/run_all.py \
      python/harness/bench/benchmark_gemm_unified_graph.py \
      python/harness/bench/benchmark_attention_unified_graph.py \
      python/harness/bench/benchmark_moe_unified_graph.py \
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

CMD ["sleep", "infinity"]
