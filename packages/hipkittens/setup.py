from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

SIZES = (1024, 2048, 4096, 8192, 16384)


class BuildHipKittensExtension(build_ext):
    """Build the single HIPKittens extension with hipcc."""

    def build_extension(self, ext: Extension) -> None:
        if ext.name != "hipkittens._C":
            super().build_extension(ext)
            return

        source_root = _hipkittens_root()
        generated_dir = Path(self.build_temp) / "hipkittens_sources"
        generated_dir.mkdir(parents=True, exist_ok=True)
        sources = [
            _write_kernel_adapter(source_root, generated_dir, size) for size in SIZES
        ]
        binding_source = generated_dir / "bindings.cpp"
        binding_source.write_text(_binding_source(), encoding="utf-8")

        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        hipcc = os.environ.get("HIPCC") or shutil.which("hipcc")
        if hipcc is None:
            raise RuntimeError("hipcc was not found; set HIPCC or add hipcc to PATH")

        arch = os.environ.get("HIPKITTENS_ARCH", "gfx942")
        hip_include = _hip_include_dir(Path(hipcc).resolve())
        command = [
            hipcc,
            *map(str, sources),
            str(binding_source),
            "-O3",
            "-DKITTENS_CDNA3",
            f"--offload-arch={arch}",
            "-std=c++20",
            "-w",
            "-shared",
            "-fPIC",
            f"-I{hip_include}",
            f"-I{source_root / 'include'}",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_path('include')}",
            "-lpthread",
            "-ldl",
            "-lutil",
            "-lm",
            "-o",
            str(output),
        ]
        subprocess.run(command, check=True)


def _hipkittens_root() -> Path:
    value = os.environ.get("HIPKITTENS_ROOT")
    if not value:
        raise RuntimeError(
            "HIPKITTENS_ROOT is required and must point to the pinned HipKittens "
            "source checkout"
        )
    root = Path(value).expanduser().resolve()
    required = [
        root / "include" / "kittens.cuh",
        root / "include" / "pyutils" / "pyutils.cuh",
        *[
            root / "analysis" / "bf16_gemm" / "mi325x" / f"kernel_{size}.cpp"
            for size in SIZES
        ],
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"HIPKITTENS_ROOT is incomplete; missing: {formatted}")
    return root


def _hip_include_dir(hipcc: Path) -> Path:
    configured_roots = [
        os.environ.get("ROCM_PATH", ""),
        os.environ.get("ROCM_HOME", ""),
        os.environ.get("HIP_PATH", ""),
    ]
    roots = [Path(root) for root in configured_roots if root]
    roots.extend([Path("/opt/rocm"), hipcc.parents[1]])
    for root in dict.fromkeys(roots):
        for candidate in (root / "include" / "hip", root / "include"):
            if (candidate / "hip_bf16.h").is_file():
                return candidate
    raise RuntimeError(
        "cannot locate the HIP headers containing hip_bf16.h; set ROCM_PATH"
    )


def _write_kernel_adapter(source_root: Path, generated_dir: Path, size: int) -> Path:
    source_path = (
        source_root / "analysis" / "bf16_gemm" / "mi325x" / f"kernel_{size}.cpp"
    )
    source = source_path.read_text(encoding="utf-8")
    module_marker = "PYBIND11_MODULE(tk_kernel, m)"
    module_start = source.rfind(module_marker)
    if module_start < 0:
        raise RuntimeError(
            f"unexpected HIPKittens source (module missing): {source_path}"
        )

    kernel = source[:module_start]
    renamed = f"""\
#define micro_globals hipkittens_micro_globals_{size}
#define micro_tk hipkittens_micro_tk_{size}
#define dispatch_micro hipkittens_dispatch_micro_{size}
{kernel}
#include <cstdint>

void hipkittens_launch_{size}(
    pybind11::object a,
    pybind11::object b,
    pybind11::object c,
    std::uintptr_t stream_ptr
) {{
    hipkittens_micro_globals_{size} g {{
        kittens::py::from_object<_gl_A>::make(a),
        kittens::py::from_object<_gl_B>::make(b),
        kittens::py::from_object<_gl_C>::make(c),
        reinterpret_cast<hipStream_t>(stream_ptr),
    }};
    hipkittens_dispatch_micro_{size}(g);
}}
"""
    output = generated_dir / f"kernel_{size}.cpp"
    output.write_text(renamed, encoding="utf-8")
    return output


def _binding_source() -> str:
    declarations = "\n".join(
        f"void hipkittens_launch_{size}(pybind11::object, pybind11::object, "
        "pybind11::object, std::uintptr_t);"
        for size in SIZES
    )
    bindings = "\n".join(
        f'    m.def("gemm_{size}", &hipkittens_launch_{size}, '
        f'"Launch the {size} GEMM kernel.");'
        for size in SIZES
    )
    return f"""\
#include <cstdint>
#include <pybind11/pybind11.h>

{declarations}

PYBIND11_MODULE(_C, m) {{
    m.doc() = "Pinned HIPKittens BF16 GEMM dispatch kernels";
{bindings}
}}
"""


setup(
    ext_modules=[Extension("hipkittens._C", sources=[])],
    cmdclass={"build_ext": BuildHipKittensExtension},
)
