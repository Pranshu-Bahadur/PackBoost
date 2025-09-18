from __future__ import annotations

import os
import sys
from pathlib import Path
from shutil import which

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "packboost" / "backends" / "src"

sources = [str(SRC_DIR / "backend_cpu.cpp")]
macros = []
extra_compile_args = {"cxx": ["-O3", "-std=c++17", "-fvisibility=hidden"]}
extra_link_args: list[str] = []

NVCC = which("nvcc")
if NVCC and os.environ.get("PACKBOOST_DISABLE_CUDA") != "1":
    sources.append(str(SRC_DIR / "backend_cuda.cu"))
    macros.append(("PACKBOOST_ENABLE_CUDA", "1"))
    extra_compile_args["nvcc"] = [
        "-O3",
        "-arch=sm_70",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
    ]
else:
    sys.stderr.write("[setup_native] CUDA compiler not found – building CPU backend only\n")

ext_modules = [
    Pybind11Extension(
        "packboost._backend",
        sources,
        define_macros=macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[str(SRC_DIR)],
    )
]

setup(
    name="packboost_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
