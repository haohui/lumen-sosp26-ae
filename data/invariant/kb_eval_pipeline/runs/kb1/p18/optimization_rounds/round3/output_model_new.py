import atexit
import ctypes

import torch
import torch.nn as nn


M = 2048
K = 8192
N = 4096


class _RocBlas:
    _lib = None
    _handle = None
    _alpha = ctypes.c_float(1.0)
    _beta = ctypes.c_float(0.0)

    ROCBLAS_STATUS_SUCCESS = 0
    ROCBLAS_OP_N = 111
    ROCBLAS_DATATYPE_F32_R = 151
    ROCBLAS_DATATYPE_BF16_R = 168
    ROCBLAS_GEMM_ALGO_STANDARD = 0
    ROCBLAS_GEMM_FLAGS_NONE = 0

    @classmethod
    def _check(cls, status: int, opname: str) -> None:
        if status != cls.ROCBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{opname} failed with rocBLAS status {status}")

    @classmethod
    def _ensure_init(cls) -> None:
        if cls._lib is not None:
            return

        lib = ctypes.CDLL("librocblas.so")
        lib.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.rocblas_create_handle.restype = ctypes.c_int
        lib.rocblas_destroy_handle.argtypes = [ctypes.c_void_p]
        lib.rocblas_destroy_handle.restype = ctypes.c_int
        lib.rocblas_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.rocblas_set_stream.restype = ctypes.c_int
        lib.rocblas_gemm_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        lib.rocblas_gemm_ex.restype = ctypes.c_int

        handle = ctypes.c_void_p()
        cls._check(lib.rocblas_create_handle(ctypes.byref(handle)), "rocblas_create_handle")
        cls._lib = lib
        cls._handle = handle
        atexit.register(cls._shutdown)

    @classmethod
    def _shutdown(cls) -> None:
        if cls._lib is None or cls._handle is None:
            return
        cls._lib.rocblas_destroy_handle(cls._handle)
        cls._handle = None

    @classmethod
    def gemm_km_nk_to_mn(cls, a: torch.Tensor, b: torch.Tensor, out_col_major: torch.Tensor) -> None:
        cls._ensure_init()
        stream = torch.cuda.current_stream(device=a.device).cuda_stream
        cls._check(
            cls._lib.rocblas_set_stream(cls._handle, ctypes.c_void_p(stream)),
            "rocblas_set_stream",
        )

        cls._check(
            cls._lib.rocblas_gemm_ex(
                cls._handle,
                cls.ROCBLAS_OP_N,
                cls.ROCBLAS_OP_N,
                M,
                N,
                K,
                ctypes.byref(cls._alpha),
                ctypes.c_void_p(a.data_ptr()),
                cls.ROCBLAS_DATATYPE_BF16_R,
                M,
                ctypes.c_void_p(b.data_ptr()),
                cls.ROCBLAS_DATATYPE_BF16_R,
                K,
                ctypes.byref(cls._beta),
                ctypes.c_void_p(out_col_major.data_ptr()),
                cls.ROCBLAS_DATATYPE_BF16_R,
                M,
                ctypes.c_void_p(out_col_major.data_ptr()),
                cls.ROCBLAS_DATATYPE_BF16_R,
                M,
                cls.ROCBLAS_DATATYPE_F32_R,
                cls.ROCBLAS_GEMM_ALGO_STANDARD,
                0,
                cls.ROCBLAS_GEMM_FLAGS_NONE,
            ),
            "rocblas_gemm_ex",
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._out_col_major = None
        self._out_device = None

    def _ensure_output(self, device: torch.device) -> torch.Tensor:
        if self._out_col_major is None or self._out_device != device:
            # Shape (N, M) row-major matches an (M, N) column-major GEMM output buffer.
            self._out_col_major = torch.empty((N, M), device=device, dtype=torch.bfloat16)
            self._out_device = device
        return self._out_col_major

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if tuple(A.shape) != (K, M) or tuple(B.shape) != (N, K):
            raise ValueError(
                f"expected A.shape == ({K}, {M}) and B.shape == ({N}, {K}), "
                f"got {tuple(A.shape)} and {tuple(B.shape)}"
            )
        if A.device.type != "cuda" or B.device.type != "cuda":
            raise ValueError("expected CUDA tensors")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("expected bfloat16 tensors")
        if not A.is_contiguous() or not B.is_contiguous():
            raise ValueError("expected contiguous tensors")

        out_col_major = self._ensure_output(A.device)
        _RocBlas.gemm_km_nk_to_mn(A, B, out_col_major)
        return out_col_major.t()
