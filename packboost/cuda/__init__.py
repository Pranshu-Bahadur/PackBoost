from __future__ import annotations

from pathlib import Path
from types import ModuleType
import platform
import os
import subprocess

import torch
from torch.utils.cpp_extension import load

__all__ = ["kernels"]


def init_msvc_env():
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    
    # Add environment variable path if it exists
    if "VCVARS64_BAT_FILEPATH" in os.environ:
        candidates.insert(0, os.environ["VCVARS64_BAT_FILEPATH"])
    
    for c in candidates:
        if Path(c).exists():
            # Run vcvars and capture environment by running 'set' command
            # Pass as separate list items - no string formatting needed
            completed = subprocess.run(
                ["cmd", "/c", c, "&&", "set"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output and update current process environment
            for line in completed.stdout.split('\n'):
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            
            os.environ["CL"] = "/std:c++17 /permissive- /Zc:__cplusplus /DNOMINMAX /DWIN32_LEAN_AND_MEAN /EHsc"
            return completed.returncode
        
    raise FileExistsError("Could not find vcvars64.bat to set up MSVC environment | Install Visual Studio c++ 2022 Build Tools and/or set up environment variable VCVARS64_BAT_FILEPATH if installed at custom path (see readme).")

def _load_extension() -> ModuleType:
    src_dir = Path(__file__).resolve().parent
    sources = [
        str(src_dir / "kernels.cpp"),
        str(src_dir / "encode_cuts.cu"),
        str(src_dir / "et_sample_1b.cu"),
        str(src_dir / "prep_vars.cu"),
        str(src_dir / "h0.cu"),
        str(src_dir / "h.cu"),
        str(src_dir / "cut.cu"),
        str(src_dir / "h0_des.cu"),
        str(src_dir / "h_des.cu"),
        str(src_dir / "cut_des.cu"),
        str(src_dir / "adv.cu"),
        str(src_dir / "repack.cu"),
    ]

    # Method 1: Using platform module
    if platform.system() == "Windows":
        init_msvc_env()

    return load(
        name="packboost_cuda_kernels",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


if torch.cuda.is_available():
    kernels = _load_extension()
else:
    class _CudaUnavailable(ModuleType):
        def __init__(self) -> None:
            super().__init__("packboost.cuda.kernels")

        def __getattr__(self, name: str):  # pragma: no cover - error path
            raise RuntimeError(
                "packboost CUDA kernels are unavailable because torch.cuda.is_available() is False"
            )


    kernels = _CudaUnavailable()
