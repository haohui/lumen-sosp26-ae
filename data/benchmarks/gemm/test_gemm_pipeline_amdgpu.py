#!/usr/bin/env python3
import unittest

import torch

import amdgpu_gemm


class TestGEMM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)
        self.rtol = 1e-2
        self.atol = 1e-3

    def _assert_matches_reference(self, actual, expected):
        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )

    def test_gemm_1stage_row_major_and_transposed_b(self):
        for m, n, k in ((256, 256, 128), (512, 512, 256)):
            with self.subTest(m=m, n=n, k=k):
                A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                B = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
                B_t = B.t().contiguous()
                expected = (A @ B).to(dtype=torch.bfloat16, device="cpu")

                actual_row_major = amdgpu_gemm.gemm_1stage_pipeline_row_major(A, B).to("cpu")
                actual_transposed = amdgpu_gemm.gemm_1stage_pipeline_transposed_b(A, B_t).to("cpu")
                actual_alias = amdgpu_gemm.gemm_1stage_pipeline(A, B).to("cpu")

                self._assert_matches_reference(actual_row_major, expected)
                self._assert_matches_reference(actual_transposed, expected)
                self._assert_matches_reference(actual_alias, expected)

    def test_bmm_1stage_row_major_and_transposed_b(self):
        for batch, m, n, k in ((2, 256, 256, 128), (3, 512, 512, 256)):
            with self.subTest(batch=batch, m=m, n=n, k=k):
                A = torch.randn((batch, m, k), dtype=torch.bfloat16, device="cuda")
                B = torch.randn((batch, k, n), dtype=torch.bfloat16, device="cuda")
                B_t = B.transpose(1, 2).contiguous()
                expected = torch.bmm(A, B).to(dtype=torch.bfloat16, device="cpu")

                actual_row_major = amdgpu_gemm.gemm_1stage_pipeline_row_major(A, B).to("cpu")
                actual_transposed = amdgpu_gemm.gemm_1stage_pipeline_transposed_b(A, B_t).to("cpu")
                actual_alias = amdgpu_gemm.gemm_1stage_pipeline(A, B).to("cpu")

                self._assert_matches_reference(actual_row_major, expected)
                self._assert_matches_reference(actual_transposed, expected)
                self._assert_matches_reference(actual_alias, expected)

    def test_row_major_entrypoint_rejects_transposed_b_layout(self):
        m = 256
        n = 512
        k = 128
        A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(ValueError, r"standard row-major GEMM shape"):
            amdgpu_gemm.gemm_1stage_pipeline_row_major(A, B.t().contiguous())

    def test_transposed_b_entrypoint_rejects_row_major_b_layout(self):
        m = 256
        n = 512
        k = 128
        A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(ValueError, r"transposed B must have shape"):
            amdgpu_gemm.gemm_1stage_pipeline_transposed_b(A, B)

    def test_batched_row_major_entrypoint_rejects_transposed_b_layout(self):
        batch = 2
        m = 256
        n = 512
        k = 128
        A = torch.randn((batch, m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((batch, k, n), dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(ValueError, r"batched row-major GEMM"):
            amdgpu_gemm.gemm_1stage_pipeline_row_major(A, B.transpose(1, 2).contiguous())

    def test_batched_transposed_b_entrypoint_rejects_row_major_b_layout(self):
        batch = 2
        m = 256
        n = 512
        k = 128
        A = torch.randn((batch, m, k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((batch, k, n), dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(ValueError, r"batched transposed GEMM"):
            amdgpu_gemm.gemm_1stage_pipeline_transposed_b(A, B)


if __name__ == "__main__":
    unittest.main()
