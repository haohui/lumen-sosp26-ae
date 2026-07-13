import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    idx = bid * block_size + tid

    if idx < numel:
        val = x[idx]
        val_f32 = al.convert(val, al.f32)
        zero_f32 = al.convert(0.0, al.f32)
        if val_f32 > zero_f32:
            out[idx] = val
        else:
            out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


ELEMS_PER_THREAD = 8
BLOCK_SIZE = 256


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    elems = al.constexpr(ELEMS_PER_THREAD)

    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    base = (bid * block_size + tid) * elems

    for i in al.range(elems):
        idx = base + i
        if idx < numel:
            val = x[idx]
            val_f32 = al.convert(val, al.f32)
            if val_f32 > al.convert(0.0, al.f32):
                out[idx] = val
            else:
                out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 256
ELEMS_PER_THREAD = 8


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    ELEMS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    base = (bid * block_size + tid) * ELEMS_PER_THREAD

    for i in al.range(ELEMS_PER_THREAD):
        idx = base + i
        if idx < numel:
            val = x[idx]
            val_f32 = al.convert(val, al.f32)
            if val_f32 > al.convert(0.0, al.f32):
                out[idx] = val
            else:
                out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, ELEMS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 1024
ELEMS_PER_THREAD = 8


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    ELEMS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    base = (bid * block_size + tid) * ELEMS_PER_THREAD

    for i in al.range(ELEMS_PER_THREAD):
        idx = base + i
        if idx < numel:
            val = x[idx]
            val_f32 = al.convert(val, al.f32)
            if val_f32 > al.convert(0.0, al.f32):
                out[idx] = val
            else:
                out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, ELEMS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 256
ELEMS_PER_THREAD = 16


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    ELEMS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    base = (bid * block_size + tid) * ELEMS_PER_THREAD

    for i in al.range(ELEMS_PER_THREAD):
        idx = base + i
        if idx < numel:
            val = x[idx]
            val_f32 = al.convert(val, al.f32)
            if val_f32 > al.convert(0.0, al.f32):
                out[idx] = val
            else:
                out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, ELEMS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 256
VECTORS_PER_THREAD = 2


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    VECTORS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    x_rsrc = al.amdgpu.make_rsrc(x, numel * 2)
    out_rsrc = al.amdgpu.make_rsrc(out, numel * 2)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    # Each vector = 4 x i32 = 8 x bf16 elements
    base_elem = (bid * block_size + tid) * VECTORS_PER_THREAD * 8

    zero_f32 = al.convert(0.0, al.f32)

    for v in al.range(VECTORS_PER_THREAD):
        elem_offset = base_elem + v * 8
        if elem_offset < numel:
            remaining = numel - elem_offset

            if remaining >= 8:
                # Full vector: use 128-bit buffer load/store
                byte_offset = elem_offset * 2
                vec = al.amdgpu.raw_buffer_load_x4(x_rsrc, byte_offset, 0, 0)
                bf16v = al.view(vec, al.Tensor((8,), al.bf16))

                results = al.make_local((8,), al.bf16)
                for i in al.range(8):
                    val = bf16v[i]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        results[i] = val
                    else:
                        results[i] = al.convert(0.0, al.bf16)

                store_vec = al.view(results, al.Tensor((4,), al.i32))
                al.amdgpu.raw_buffer_store_x4(store_vec, out_rsrc, byte_offset, 0, 0)
            else:
                # Partial tail: scalar load/store for remaining elements
                idx = elem_offset
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 1
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 2
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 3
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 4
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 5
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)

                idx = elem_offset + 6
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = al.convert(0.0, al.bf16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * VECTORS_PER_THREAD * 8 - 1) // (BLOCK_SIZE * VECTORS_PER_THREAD * 8)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, VECTORS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 256
VECTORS_PER_THREAD = 1


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    VECTORS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    x_rsrc = al.amdgpu.make_rsrc(x, numel * 2)
    out_rsrc = al.amdgpu.make_rsrc(out, numel * 2)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    # Each vector = 4 x i32 = 8 x bf16 elements
    base_elem = (bid * block_size + tid) * VECTORS_PER_THREAD * 8

    zero_f32 = al.convert(0.0, al.f32)
    zero_bf16 = al.convert(0.0, al.bf16)

    for v in al.range(VECTORS_PER_THREAD):
        elem_offset = base_elem + v * 8
        if elem_offset < numel:
            remaining = numel - elem_offset

            if remaining >= 8:
                byte_offset = elem_offset * 2
                vec = al.amdgpu.raw_buffer_load_x4(x_rsrc, byte_offset, 0, 0)
                bf16v = al.view(vec, al.Tensor((8,), al.bf16))

                results = al.make_local((8,), al.bf16)
                for i in al.range(8):
                    val = bf16v[i]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        results[i] = val
                    else:
                        results[i] = zero_bf16

                store_vec = al.view(results, al.Tensor((4,), al.i32))
                al.amdgpu.raw_buffer_store_x4(store_vec, out_rsrc, byte_offset, 0, 0)
            else:
                # Partial tail for the last chunk
                idx = elem_offset
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 1
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 2
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 3
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 4
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 5
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 6
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * VECTORS_PER_THREAD * 8 - 1) // (BLOCK_SIZE * VECTORS_PER_THREAD * 8)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, VECTORS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE = 256
VECTORS_PER_THREAD = 1


@avelang.jit
def relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    VECTORS_PER_THREAD: al.constexpr,
):
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    x_rsrc = al.amdgpu.make_rsrc(x, numel * 2)
    out_rsrc = al.amdgpu.make_rsrc(out, numel * 2)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    base_elem = (bid * block_size + tid) * VECTORS_PER_THREAD * 8

    zero_f32 = al.convert(0.0, al.f32)
    zero_bf16 = al.convert(0.0, al.bf16)

    for v in al.range(VECTORS_PER_THREAD):
        elem_offset = base_elem + v * 8
        if elem_offset < numel:
            remaining = numel - elem_offset

            if remaining >= 8:
                byte_offset = elem_offset * 2
                vec = al.amdgpu.raw_buffer_load_x4(x_rsrc, byte_offset, 0, 0)
                bf16v = al.view(vec, al.Tensor((8,), al.bf16))

                results = al.make_local((8,), al.bf16)
                for i in al.range(8):
                    val = bf16v[i]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        results[i] = val
                    else:
                        results[i] = zero_bf16

                store_vec = al.view(results, al.Tensor((4,), al.i32))
                al.amdgpu.raw_buffer_store_x4(store_vec, out_rsrc, byte_offset, 0, 0)
            else:
                # Scalar tail for the final partial vector
                idx = elem_offset
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 1
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 2
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 3
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 4
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 5
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16

                idx = elem_offset + 6
                if idx < numel:
                    val = x[idx]
                    val_f32 = al.convert(val, al.f32)
                    if val_f32 > zero_f32:
                        out[idx] = val
                    else:
                        out[idx] = zero_bf16


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = (numel + BLOCK_SIZE * VECTORS_PER_THREAD * 8 - 1) // (BLOCK_SIZE * VECTORS_PER_THREAD * 8)

    relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x, out, numel, VECTORS_PER_THREAD)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avelang_relu(x)
