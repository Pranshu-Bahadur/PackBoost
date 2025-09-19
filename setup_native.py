from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which
from typing import Dict, List, Tuple

from pybind11.setup_helpers import Pybind11Extension, build_ext

try:
    import pybind11
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pybind11 is required: pip install pybind11") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required: pip install numpy") from exc

SRC_DIR = Path("packboost") / "backends" / "src"
CPU_SOURCES = [str(SRC_DIR / "backend_cpu.cpp")]
CUDA_SOURCE = SRC_DIR / "backend_cuda.cu"
ROOT = Path(__file__).parent.resolve()


def _locate_nvcc() -> str | None:
    nvcc_path = which("nvcc")
    if nvcc_path:
        return nvcc_path
    default_nvcc = Path("/usr/local/cuda/bin/nvcc")
    if default_nvcc.exists():
        return str(default_nvcc)
    return None


def build_native_extension() -> Tuple[List[Pybind11Extension], Dict[str, object]]:
    nvcc_path = _locate_nvcc()
    enable_cuda = nvcc_path is not None and os.environ.get("PACKBOOST_DISABLE_CUDA") != "1"

    macros: list[tuple[str, str | None]] = []
    extra_objects: list[str] = []
    cuda_library_dirs: list[str] = []
    cuda_libraries: list[str] = []
    cuda_runtime_library_dirs: list[str] = []

    if enable_cuda:
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
            nvcc_path,
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-c",
            str((ROOT / CUDA_SOURCE).resolve()),
            "-o",
            str(cuda_obj),
        ]
        for arch in arch_list:
            cmd.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
            cmd.extend(["-gencode", f"arch=compute_{arch},code=compute_{arch}"])
        for inc in include_dirs:
            cmd.extend(["-I", inc])
        cmd.extend(["-DPACKBOOST_ENABLE_CUDA=1"])
        print("[setup] compiling CUDA backend:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise SystemExit(f"nvcc failed with exit code {exc.returncode}") from exc
        macros.append(("PACKBOOST_ENABLE_CUDA", "1"))
        extra_objects.append(str(cuda_obj))

        cuda_root_env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root_env:
            cuda_root = Path(cuda_root_env)
        else:
            cuda_root = Path(nvcc_path).resolve().parent.parent

        lib_dir_env = os.environ.get("PACKBOOST_CUDA_LIBDIR")
        candidate_dirs: list[Path] = []
        if lib_dir_env:
            candidate_dirs.append(Path(lib_dir_env))
        candidate_dirs.extend([cuda_root / "lib64", cuda_root / "lib"])
        for directory in candidate_dirs:
            if directory.exists():
                cuda_library_dirs.append(str(directory))
                cuda_runtime_library_dirs.append(str(directory))
                break

        cuda_libraries.append("cudart")
    else:
        print("[setup] CUDA compiler not found – building CPU backend only", file=sys.stderr)

    ext_modules = [
        Pybind11Extension(
            "packboost._backend",
            CPU_SOURCES,
            define_macros=macros,
            include_dirs=[str(SRC_DIR), np.get_include()],
            extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            runtime_library_dirs=cuda_runtime_library_dirs,
            extra_objects=extra_objects,
        )
    ]

    return ext_modules, {"build_ext": build_ext}


if __name__ == "__main__":
    from setuptools import setup

    extensions, cmdclass = build_native_extension()
    setup(ext_modules=extensions, cmdclass=cmdclass)
