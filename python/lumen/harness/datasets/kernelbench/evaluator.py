"""CUDA graph evaluators for KernelBench-style model files."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from kernelbench.eval import KernelExecResult

DEFAULT_SEED = 42
DEFAULT_NUM_WARMUP = 5
DEFAULT_NUM_TRIALS = 100
DEFAULT_BACKEND = "cuda"
TEMPFILE_BACKENDS = {"triton", "tilelang", "cute", "substrate"}


def evaluate_reference_file(filename: str | Path) -> KernelExecResult:
    """Evaluate a KernelBench reference file with CUDA graph replay timing."""
    import torch
    from kernelbench.eval import KernelExecResult, get_error_name

    path = Path(filename).expanduser()
    metadata: dict[str, Any] = {"filename": str(path)}
    validation_error = _validate_input_file(path, "Reference")
    if validation_error is not None:
        metadata["error"] = validation_error
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    if not torch.cuda.is_available():
        metadata["error"] = "CUDA is not available, cannot run KernelBench eval"
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    metadata["device"] = str(device)

    try:
        torch.cuda.set_device(device)
        metadata["hardware"] = torch.cuda.get_device_name(device=device)
        Model, get_init_inputs, get_inputs = _load_original_model(path)
        eval_settings = _get_eval_settings({})
        model = _instantiate_model(
            Model=Model,
            get_init_inputs=get_init_inputs,
            device=device,
            backend=DEFAULT_BACKEND,
            precision=torch.float32,
        )
        inputs = _make_inputs(
            get_inputs=get_inputs,
            device=device,
            backend=DEFAULT_BACKEND,
            precision=torch.float32,
        )
        runtime_stats = _measure_model_with_cuda_graph(
            model=model,
            inputs=inputs,
            device=device,
            num_warmup=eval_settings["num_warmup"],
            num_trials=eval_settings["num_trials"],
        )
        metadata["precision"] = str(torch.float32)
        metadata["backend"] = DEFAULT_BACKEND
        metadata["seed"] = DEFAULT_SEED
        metadata["num_warmup"] = eval_settings["num_warmup"]
        metadata["num_trials"] = eval_settings["num_trials"]
        metadata["cuda_graph"] = True

        return KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=runtime_stats["mean"],
            runtime_stats=runtime_stats,
            metadata=metadata,
        )
    except Exception as exc:
        metadata["error"] = str(exc)
        metadata["error_name"] = get_error_name(exc)
        metadata["traceback"] = traceback.format_exc()
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
    finally:
        _cleanup_cuda(torch, device)


def evaluate_generated_model(
    original_model_file: str | Path,
    generated_model_file: str | Path,
    eval_config: dict[str, Any] | None = None,
) -> KernelExecResult:
    """Evaluate a generated KernelBench ModelNew against an original Model file."""
    import torch
    from kernelbench.eval import (
        KernelExecResult,
        get_error_name,
        load_custom_model,
        load_custom_model_with_tempfile,
        run_and_check_correctness,
    )

    original_path = Path(original_model_file).expanduser()
    generated_path = Path(generated_model_file).expanduser()
    config = dict(eval_config or {})
    backend = str(config.get("backend", DEFAULT_BACKEND)).lower()

    metadata: dict[str, Any] = {
        "original_model_file": str(original_path),
        "generated_model_file": str(generated_path),
        "backend": backend,
        "eval_config": config,
    }
    if "gpu_arch" in config:
        metadata["gpu_arch"] = config["gpu_arch"]

    for path, label in (
        (original_path, "Original model"),
        (generated_path, "Generated model"),
    ):
        validation_error = _validate_input_file(path, label)
        if validation_error is not None:
            metadata["error"] = validation_error
            return KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )

    if not torch.cuda.is_available():
        metadata["error"] = "CUDA is not available, cannot run KernelBench eval"
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    metadata["device"] = str(device)
    temp_file = None

    try:
        torch.cuda.set_device(device)
        metadata["hardware"] = torch.cuda.get_device_name(device=device)
        precision = _get_precision(config)
        eval_settings = _get_eval_settings(config)
        metadata["precision"] = str(precision)
        metadata["seed"] = eval_settings["seed"]
        metadata["num_correct_trials"] = eval_settings["num_correct_trials"]
        metadata["num_warmup"] = eval_settings["num_warmup"]
        metadata["num_trials"] = eval_settings["num_trials"]
        metadata["cuda_graph"] = True

        Model, get_init_inputs, get_inputs = _load_original_model(original_path)
        original_model = _instantiate_model(
            Model=Model,
            get_init_inputs=get_init_inputs,
            device=device,
            backend=backend,
            precision=precision,
            seed=eval_settings["seed"],
        )

        custom_src = generated_path.read_text(encoding="utf-8")
        context: dict[str, Any] = {}
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if backend in TEMPFILE_BACKENDS:
            ModelNew, temp_file = load_custom_model_with_tempfile(
                custom_src,
                entry_point="ModelNew",
            )
        else:
            ModelNew = load_custom_model(custom_src, context)

        if ModelNew is None:
            metadata["compilation_error_name"] = "ModelNewNotFound"
            metadata["compilation_error"] = (
                "Generated model must define ModelNew and compile successfully"
            )
            return KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )

        set_seed = _kernelbench_set_seed()
        set_seed(eval_settings["seed"])
        init_inputs = get_init_inputs()
        init_inputs = _process_inputs(
            init_inputs,
            device=device,
            backend=backend,
            precision=precision,
        )

        with torch.no_grad(), torch.cuda.device(device):
            set_seed(eval_settings["seed"])
            custom_model = ModelNew(*init_inputs)
            custom_model = custom_model.to(device=device, dtype=precision)
            custom_model.eval()
            original_model = original_model.to(device=device, dtype=precision)
            original_model.eval()
            torch.cuda.synchronize(device=device)

        correctness_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=eval_settings["num_correct_trials"],
            verbose=False,
            seed=eval_settings["seed"],
            device=device,
            backend=backend,
            precision=precision,
        )
        if not correctness_result.correctness:
            return correctness_result

        set_seed(eval_settings["seed"])
        inputs = _make_inputs(
            get_inputs=get_inputs,
            device=device,
            backend=backend,
            precision=precision,
        )

        runtime_stats = _measure_model_with_cuda_graph(
            model=custom_model,
            inputs=inputs,
            device=device,
            num_warmup=eval_settings["num_warmup"],
            num_trials=eval_settings["num_trials"],
        )
        ref_runtime_stats = _measure_model_with_cuda_graph(
            model=original_model,
            inputs=inputs,
            device=device,
            num_warmup=eval_settings["num_warmup"],
            num_trials=eval_settings["num_trials"],
        )
        correctness_result.runtime = runtime_stats["mean"]
        correctness_result.runtime_stats = runtime_stats
        correctness_result.ref_runtime = ref_runtime_stats["mean"]
        correctness_result.ref_runtime_stats = ref_runtime_stats
        return correctness_result
    except Exception as exc:
        metadata["error"] = str(exc)
        metadata["error_name"] = get_error_name(exc)
        metadata["traceback"] = traceback.format_exc()
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
    finally:
        if temp_file is not None:
            try:
                temp_file.close()
                os.remove(temp_file.name)
            except OSError:
                pass
        _cleanup_cuda(torch, device)


def _validate_input_file(path: Path, label: str) -> str | None:
    if not path.is_file():
        return f"{label} file not found: {path}"
    return None


def _load_original_model(path: Path) -> tuple[Any, Any, Any]:
    from kernelbench.eval import load_original_model_and_inputs

    model_src = path.read_text(encoding="utf-8")
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(model_src, {})
    if Model is None or get_init_inputs is None or get_inputs is None:
        raise ValueError(
            "Original model file must define Model, get_init_inputs, and get_inputs"
        )
    return Model, get_init_inputs, get_inputs


def _instantiate_model(
    *,
    Model: type[torch.nn.Module],
    get_init_inputs: Any,
    device: torch.device,
    backend: str,
    precision: torch.dtype,
    seed: int = DEFAULT_SEED,
) -> torch.nn.Module:
    import torch

    set_seed = _kernelbench_set_seed()
    with torch.no_grad(), torch.cuda.device(device):
        set_seed(seed)
        init_inputs = get_init_inputs()
        init_inputs = _process_inputs(
            init_inputs,
            device=device,
            backend=backend,
            precision=precision,
        )

        set_seed(seed)
        model = Model(*init_inputs)
        model = model.to(device=device, dtype=precision)
        model.eval()
        torch.cuda.synchronize(device=device)
        return model


def _make_inputs(
    *,
    get_inputs: Any,
    device: torch.device,
    backend: str,
    precision: torch.dtype,
) -> list[Any]:
    values = get_inputs()
    return _process_inputs(
        values,
        device=device,
        backend=backend,
        precision=precision,
    )


def _process_inputs(
    values: list[Any],
    *,
    device: torch.device,
    backend: str,
    precision: torch.dtype,
) -> list[Any]:
    from kernelbench.eval import _process_input_tensor

    return [
        _process_input_tensor(
            value,
            device=device,
            backend=backend,
            precision=precision,
        )
        for value in values
    ]


def _measure_model_with_cuda_graph(
    *,
    model: torch.nn.Module,
    inputs: list[Any],
    device: torch.device,
    num_warmup: int,
    num_trials: int,
) -> dict[str, Any]:
    import torch
    from kernelbench.timing import get_timing_stats

    with torch.no_grad(), torch.cuda.device(device):
        torch.cuda.synchronize(device=device)
        for _ in range(num_warmup):
            model(*inputs)
        torch.cuda.synchronize(device=device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(*inputs)
        torch.cuda.synchronize(device=device)

        elapsed_times = []
        for _ in range(num_trials):
            torch.cuda.synchronize(device=device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            torch.cuda.synchronize(device=device)
            elapsed_times.append(start_event.elapsed_time(end_event))

    stats = get_timing_stats(elapsed_times, device=device)
    stats["timing_mode"] = "cudagraph"
    return stats


def _get_precision(config: dict[str, Any]) -> torch.dtype:
    from kernelbench.eval import get_torch_dtype_from_string

    precision = config.get("precision", "fp32")
    if isinstance(precision, str):
        return get_torch_dtype_from_string(precision)
    return precision


def _get_eval_settings(config: dict[str, Any]) -> dict[str, int]:
    return {
        "seed": _get_int_config(config, "seed", DEFAULT_SEED),
        "num_correct_trials": _get_int_config(config, "num_correct_trials", 1),
        "num_warmup": _get_int_config(config, "num_warmup", DEFAULT_NUM_WARMUP),
        "num_trials": _get_int_config(config, "num_trials", DEFAULT_NUM_TRIALS),
    }


def _get_int_config(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _kernelbench_set_seed() -> Any:
    from kernelbench.eval import set_seed

    return set_seed


def _cleanup_cuda(torch_module: Any, device: Any) -> None:
    if device is None or not torch_module.cuda.is_available():
        return
    with torch_module.cuda.device(device):
        torch_module.cuda.empty_cache()
        torch_module.cuda.synchronize(device=device)
