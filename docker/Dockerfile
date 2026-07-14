# syntax=docker/dockerfile:1.7

FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
        cmake git libedit-dev libncurses-dev \
        libxml2-dev libzstd-dev \
        ninja-build openssh-client \
        patch pkg-config \
        python3-dev nodejs npm \
        z3 libz3-dev vim-nox \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /workspace

ARG LUMEN_UID=1001
ARG LUMEN_GID=1001

RUN getent group 44 >/dev/null || groupadd -g 44 video \
    && getent group 109 >/dev/null || groupadd -g 109 render \
    && groupadd -g ${LUMEN_GID} lumen \
    && useradd -m -u ${LUMEN_UID} -g ${LUMEN_GID} -G 44,109 -m -d /workspace/lumen -s /bin/bash lumen \
    && python3 -m pip install --no-cache-dir uv \
    && chown lumen:lumen /workspace -R

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone --depth 1 --branch lumen \
        git@github.com:haohui/llvm-project.git /workspace/llvm-project \
    && cmake -S /workspace/llvm-project/llvm -B /workspace/llvm-project/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_PROJECTS="clang;mlir;compiler-rt;lld" \
        -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    && cmake --build /workspace/llvm-project/build --parallel "$(nproc)" \
    && cmake --install /workspace/llvm-project/build \
    && rm -rf /workspace/llvm-project

RUN npm install -g @openai/codex 

USER lumen
COPY --chown=lumen:lumen . /workspace/lumen

ENV VIRTUAL_ENV=/workspace/lumen/.venv \
    PATH=/workspace/lumen/.venv/bin:/opt/llvm/bin:/opt/rocm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/llvm/lib:/opt/rocm/lib:${LD_LIBRARY_PATH} \
    AITER_JIT_DIR=/workspace/lumen/.aiter/jit

RUN --mount=type=ssh,uid=${LUMEN_UID},gid=${LUMEN_GID} uv venv --python 3.12 /workspace/lumen/.venv \
    && cp -as /opt/venv/lib/python3.12/site-packages/torch* \
        /opt/venv/lib/python3.12/site-packages/triton* \
        /workspace/lumen/.venv/lib/python3.12/site-packages/ \
    && mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts \
    && git clone --depth 1 --branch lumen git@github.com:haohui/avelang.git /workspace/avelang \
    && /workspace/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KernelBench/source.toml \
        --dest /workspace/third_party/KernelBench/KernelBench \
    && /workspace/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KSearch/source.toml \
        --dest /workspace/third_party/KSearch/K-Search \
    && /workspace/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KernelFalcon/source.toml \
        --dest /workspace/third_party/KernelFalcon/KernelAgent \
    && /workspace/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/CUDAForge/source.toml \
        --dest /workspace/third_party/CUDAForge/CudaForge \
    && /workspace/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/HipKittens/source.toml \
        --dest /workspace/third_party/HipKittens/HipKittens

# Reuse the ROCm image's PyTorch/Triton copied into the isolated UV environment.
RUN uv pip install --python /workspace/lumen/.venv/bin/python Jinja2 numpy pip pybind11 pytest setuptools wheel \
        scikit-build-core setuptools-scm \
        sympy networkx \
        ninja pandas matplotlib einops psutil \
        openai python-dotenv pydra-config anthropic omegaconf requests \
        modal datasets transformers tabulate tomli litellm mitmproxy \
    && FLASHINFER_HIP_ARCHITECTURES=gfx942 uv pip install --python /workspace/lumen/.venv/bin/python --no-deps \
        git+https://github.com/ROCm/flashinfer.git@v0.3.1+amd.1 \
    && uv pip install --python /workspace/lumen/.venv/bin/python --no-deps \
        git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git \
    && uv pip install --python /workspace/lumen/.venv/bin/python --no-deps \
        git+https://github.com/ROCm/aiter.git@v0.1.10.post3 \
    && uv pip install --python /workspace/lumen/.venv/bin/python --no-deps \
        /workspace/third_party/KernelBench/KernelBench \
    && CC=/opt/llvm/bin/clang CXX=/opt/llvm/bin/clang++ \
        CMAKE_ARGS="-DAVE_LANG_BACKEND=rocm -DWITH_PYTHON=ON -DCMAKE_PREFIX_PATH=/opt/llvm;/opt/rocm" \
        uv pip install --python /workspace/lumen/.venv/bin/python --no-deps /workspace/avelang \
    && HIPKITTENS_ROOT=/workspace/third_party/HipKittens/HipKittens HIPKITTENS_ARCH=gfx942 \
        uv pip install --python /workspace/lumen/.venv/bin/python --no-build-isolation \
        -e /workspace/lumen/packages/hipkittens \
    && uv pip install --python /workspace/lumen/.venv/bin/python -e /workspace/lumen

WORKDIR /workspace/lumen

CMD ["bash"]
