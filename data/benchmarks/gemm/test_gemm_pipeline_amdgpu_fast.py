#!/usr/bin/env python3
import unittest

import torch

import amdgpu_gemm_fast


class TestGEMM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)
        self.rtol = 1e-2
        self.atol = 1e-3

    def test_gemm_1stage(self):
        m = 1024
        n = 1024
        k = 1024
        A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        B_t = B.t().contiguous()
        expected = (A @ B).to(dtype=torch.bfloat16, device="cpu")
        C = amdgpu_gemm_fast.gemm_1stage_pipeline_transposed_b(A, B_t)

        actual = C.to("cpu")

        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )


if __name__ == "__main__":
    unittest.main()
