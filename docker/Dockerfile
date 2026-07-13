# syntax=docker/dockerfile:1.7

FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libedit-dev \
        libncurses-dev \
        libxml2-dev \
        libzstd-dev \
        ninja-build \
        openssh-client \
        patch \
        pkg-config \
        python3-dev \
        z3 \
        libz3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir uv \
    && uv venv --python 3.12 /home/lumen/.venv \
    && cp -al /opt/venv/lib/python3.12/site-packages/torch* \
        /opt/venv/lib/python3.12/site-packages/triton* \
        /home/lumen/.venv/lib/python3.12/site-packages/

RUN getent group render >/dev/null || groupadd --system render \
    && getent group video >/dev/null || groupadd --system video \
    && groupmod --new-name lumen ubuntu \
    && usermod --login lumen --home /home/lumen ubuntu \
    && chown -R lumen:lumen /home/lumen \
    && usermod --gid lumen --append --groups render,video lumen

ENV VIRTUAL_ENV=/home/lumen/.venv \
    PATH=/home/lumen/.venv/bin:/opt/llvm/bin:/opt/rocm/bin:${PATH} \
    ROCM_PATH=/opt/rocm \
    CMAKE_PREFIX_PATH=/opt/llvm/lib/cmake/llvm:/opt/llvm/lib/cmake/mlir:/opt/rocm \
    LD_LIBRARY_PATH=/opt/llvm/lib:/opt/rocm/lib:${LD_LIBRARY_PATH} \
    AITER_JIT_DIR=/workspace/lumen/.aiter/jit

WORKDIR /workspace

# The LLVM repository is private. Build with: docker build --ssh default -f docker/Dockerfile .
RUN --mount=type=ssh git clone --depth 1 --branch lumen \
        git@github.com:haohui/llvm-project.git /workspace/llvm-project \
    && cmake -S /workspace/llvm-project/llvm -B /workspace/llvm-project/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_PROJECTS=clang\;mlir\;compiler-rt \
        -DLLVM_TARGETS_TO_BUILD=X86\;AMDGPU \
    && cmake --build /workspace/llvm-project/build --parallel "$(nproc)" \
    && cmake --install /workspace/llvm-project/build \
    && rm -rf /workspace/llvm-project

RUN git clone --depth 1 https://github.com/haohui/avelang.git /workspace/avelang

COPY . /workspace/lumen

RUN /home/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KernelBench/source.toml \
        --dest /workspace/third_party/KernelBench/KernelBench \
    && /home/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KSearch/source.toml \
        --dest /workspace/third_party/KSearch/K-Search \
    && /home/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/KernelFalcon/source.toml \
        --dest /workspace/third_party/KernelFalcon/KernelAgent \
    && /home/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/CUDAForge/source.toml \
        --dest /workspace/third_party/CUDAForge/CudaForge \
    && /home/lumen/.venv/bin/python /workspace/lumen/dev-support/prepare-dependency.py \
        --source /workspace/lumen/third_party/HipKittens/source.toml \
        --dest /workspace/third_party/HipKittens/HipKittens

# Reuse the ROCm image's PyTorch/Triton copied into the isolated UV environment.
RUN uv pip install --python /home/lumen/.venv/bin/python Jinja2 numpy pip pybind11 pytest setuptools wheel \
        scikit-build-core setuptools-scm \
        sympy networkx \
        ninja pandas matplotlib einops psutil \
        openai python-dotenv pydra-config anthropic omegaconf requests \
        modal datasets transformers tabulate tomli litellm mitmproxy \
    && git clone --depth 1 --branch v0.3.1+amd.1 \
        https://github.com/ROCm/flashinfer.git /workspace/flashinfer-rocm \
    && FLASHINFER_HIP_ARCHITECTURES=gfx942 \
        /home/lumen/.venv/bin/python -m pip wheel /workspace/flashinfer-rocm \
        --wheel-dir=/workspace/flashinfer-wheelhouse --no-deps --no-build-isolation -v \
    && uv pip install --python /home/lumen/.venv/bin/python --no-index \
        --find-links /workspace/flashinfer-wheelhouse amd-flashinfer==0.3.1+amd.1 \
    && rm -rf /workspace/flashinfer-rocm /workspace/flashinfer-wheelhouse \
    && uv pip install --python /home/lumen/.venv/bin/python --no-deps \
        git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git \
    && uv pip install --python /home/lumen/.venv/bin/python --no-deps \
        git+https://github.com/ROCm/aiter.git@v0.1.10.post3 \
    && CC=/opt/llvm/bin/clang CXX=/opt/llvm/bin/clang++ \
        CMAKE_ARGS="-DAVE_LANG_BACKEND=rocm -DWITH_PYTHON=ON -DCMAKE_PREFIX_PATH=/opt/llvm;/opt/rocm -DLLVM_DIR=/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/opt/llvm/lib/cmake/mlir -DClang_DIR=/opt/llvm/lib/cmake/clang" \
        uv pip install --python /home/lumen/.venv/bin/python --no-deps /workspace/avelang \
    && HIPKITTENS_ROOT=/workspace/third_party/HipKittens/HipKittens HIPKITTENS_ARCH=gfx942 \
        uv pip install --python /home/lumen/.venv/bin/python --no-build-isolation \
        -e /workspace/lumen/packages/hipkittens \
    && uv pip install --python /home/lumen/.venv/bin/python -e /workspace/lumen

RUN mkdir -p /workspace/lumen/.aiter/jit \
    && chown -R lumen:lumen /home/lumen /workspace

WORKDIR /workspace/lumen
USER lumen

CMD ["bash"]
