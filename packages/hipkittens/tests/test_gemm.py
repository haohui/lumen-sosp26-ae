from __future__ import annotations

import hipkittens
import pytest
import torch


def test_rejects_unsupported_size() -> None:
    a = torch.empty((128, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA tensor"):
        hipkittens.gemm(a, a, a)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires a ROCm-enabled PyTorch installation",
)
def test_gemm_1024_matches_torch() -> None:
    torch.manual_seed(0)
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    hipkittens.gemm(a, b, out)
    torch.cuda.synchronize()

    expected = (a.float() @ b.float().T).to(dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-1, atol=1e-1)
