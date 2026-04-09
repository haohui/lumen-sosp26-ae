import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F


# Summary:
# Reference model performs causal scaled dot-product attention on:
# Q: [16, 8, S, 128] bf16
# K,V: [16, 1, S, 128] bf16, expanded to 8 heads via repeat_interleave when kv_heads == 1
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if K.shape[1] == 1:
            K = K.repeat_interleave(8, dim=1)
            V = V.repeat_interleave(8, dim=1)

        return F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )


def get_inputs():
    batch_size = 16
    num_q_heads = 8
    num_kv_heads = 1
    sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
    head_dim = 128

    Q = torch.randn(
        batch_size, num_q_heads, sequence_length, head_dim, dtype=torch.bfloat16
    )
    K = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16
    )
    V = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16
    )
    return [Q, K, V]


def get_init_inputs():
    return []


def _print_tensor_debug(name: str, t: torch.Tensor, max_elems: int = 10):
    flat = t.flatten()
    sample = flat[:max_elems]
    print(
        f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
        f"sample={sample}"
    )


def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function
    except Exception as e:
        print(f"Failed to import kernel_function from kernel.py: {e}")
        traceback.print_exc()
        return False

    try:
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        if not torch.cuda.is_available():
            raise RuntimeError("HIP/ROCm device not available via torch.cuda.is_available()")

        device = torch.device("cuda")

        # Build exact inputs from provided spec
        init_inputs = get_init_inputs()
        if len(init_inputs) != 0:
            print(f"Warning: get_init_inputs() returned unexpected values: {init_inputs}")

        q_cpu, k_cpu, v_cpu = get_inputs()
        q = q_cpu.to(device)
        k = k_cpu.to(device)
        v = v_cpu.to(device)

        # Reference output from provided Model semantics
        model = Model().to(device)
        with torch.no_grad():
            y_ref = model(q, k, v)

        # Kernel call as a normal Python function (no Triton launch syntax)
        with torch.no_grad():
            y = kernel_function(q, k, v)

        if not isinstance(y, torch.Tensor):
            print(f"kernel_function output is not a torch.Tensor, got type: {type(y)}")
            return False

        # Device check per requirement
        if y.device != q.device:
            print(f"Device mismatch: result.device={y.device}, input.device={q.device}")
            return False

        # Shape/dtype checks
        if y.shape != y_ref.shape:
            print(f"Shape mismatch: got {tuple(y.shape)} vs expected {tuple(y_ref.shape)}")
            return False

        # bf16 attention can have larger numeric drift; use slightly looser tolerance.
        # Documented per requirement.
        rtol = 1e-2
        atol = 2e-2

        try:
            close = torch.allclose(y, y_ref, rtol=rtol, atol=atol)
        except Exception as cmp_e:
            print(f"Exception during torch.allclose comparison: {cmp_e}")
            _print_tensor_debug("Q", q)
            _print_tensor_debug("K", k)
            _print_tensor_debug("V", v)
            _print_tensor_debug("y_ref", y_ref)
            _print_tensor_debug("y", y)
            return False

        if not close:
            diff = (y - y_ref).abs()
            max_abs = diff.max().item()
            denom = y_ref.abs() + 1e-8
            rel = (diff / denom).max().item()

            print("NUMERICAL MISMATCH:")
            print(f"Tolerance used: rtol={rtol}, atol={atol} (bf16 attention tolerance)")
            _print_tensor_debug("Q", q)
            _print_tensor_debug("K", k)
            _print_tensor_debug("V", v)
            _print_tensor_debug("Expected y_ref", y_ref)
            _print_tensor_debug("Actual y", y)
            print(f"Max absolute difference: {max_abs}")
            print(f"Max relative difference: {rel}")

            # Extra focused samples
            flat_ref = y_ref.flatten()
            flat_y = y.flatten()
            n = min(10, flat_ref.numel())
            print(f"Expected first {n}: {flat_ref[:n]}")
            print(f"Got first {n}:      {flat_y[:n]}")
            return False

        print("Test passed: kernel output matches reference.")
        return True

    except Exception as e:
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)