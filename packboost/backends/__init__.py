"""Bindings to the optional C++/CUDA histogram backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

cpu_frontier_evaluate: Optional[Any]
cuda_frontier_evaluate: Optional[Any]
cuda_predict_forest: Optional[Any]
CudaFrontierWorkspace: Optional[Any]

try:  # pragma: no cover - optional extension
    backend = import_module("packboost._backend")
    backend_load_error: ImportError | None = None
    cpu_frontier_evaluate = getattr(backend, "cpu_fastpath_evaluate", None)
    cuda_frontier_evaluate = getattr(backend, "cuda_fastpath_evaluate", None)
    cuda_predict_forest = getattr(backend, "cuda_predict_forest", None)
    CudaFrontierWorkspace = getattr(backend, "CudaFrontierWorkspace", None)
except ImportError as exc:  # pragma: no cover - extension not built
    backend_load_error = exc
    cpu_frontier_evaluate = None
    cuda_frontier_evaluate = None
    cuda_predict_forest = None
    CudaFrontierWorkspace = None


def cpu_available() -> bool:
    """Return ``True`` if the compiled CPU backend is available."""
    return cpu_frontier_evaluate is not None


def cuda_available() -> bool:
    """Return ``True`` if the compiled CUDA backend is available."""
    return cuda_frontier_evaluate is not None


__all__ = [
    "cpu_frontier_evaluate",
    "cuda_frontier_evaluate",
    "cuda_predict_forest",
    "CudaFrontierWorkspace",
    "backend_load_error",
    "cpu_available",
    "cuda_available",
]
