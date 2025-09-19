from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

try:
    import pybind11
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pybind11 is required: pip install pybind11") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required: pip install numpy") from exc

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "packboost" / "backends" / "src"

CPU_SOURCES = [str(SRC_DIR / "backend_cpu.cpp")]
CUDA_SOURCE = SRC_DIR / "backend_cuda.cu"

NVCC = which("nvcc")
if not NVCC:
    default_nvcc = Path("/usr/local/cuda/bin/nvcc")
    if default_nvcc.exists():
        NVCC = str(default_nvcc)
ENABLE_CUDA = NVCC and os.environ.get("PACKBOOST_DISABLE_CUDA") != "1"

macros: list[tuple[str, str | None]] = []
extra_objects: list[str] = []

if ENABLE_CUDA:
    build_temp = ROOT / "build" / "temp"
    build_temp.mkdir(parents=True, exist_ok=True)
    cuda_obj = build_temp / "backend_cuda.o"
    include_dirs = [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        sysconfig.get_paths()["include"],
        np.get_include(),
        str(SRC_DIR),
    ]
    arch_list_env = os.environ.get("PACKBOOST_CUDA_ARCHS")
    if arch_list_env:
        arch_list = [arch.strip() for arch in arch_list_env.split(",") if arch.strip()]
    else:
        arch_list = ["70", "75", "80", "86"]
    cmd = [
        NVCC,
        "-std=c++17",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-c",
        str(CUDA_SOURCE),
        "-o",
        str(cuda_obj),
    ]
    for arch in arch_list:
        cmd.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        cmd.extend(["-gencode", f"arch=compute_{arch},code=compute_{arch}"])
    for inc in include_dirs:
        cmd.extend(["-I", inc])
    cmd.extend(["-DPACKBOOST_ENABLE_CUDA=1"])
    print("[setup_native] compiling CUDA backend:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(f"nvcc failed with exit code {exc.returncode}") from exc
    macros.append(("PACKBOOST_ENABLE_CUDA", "1"))
    extra_objects.append(str(cuda_obj))
else:
    sys.stderr.write("[setup_native] CUDA compiler not found – building CPU backend only\n")

ext_modules = [
    Pybind11Extension(
        "packboost._backend",
        CPU_SOURCES,
        define_macros=macros,
        include_dirs=[str(SRC_DIR), np.get_include()],
        extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        extra_objects=extra_objects,
    )
]

setup(
    name="packboost_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
