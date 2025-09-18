"""Bindings to the optional C++/CUDA histogram backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

cpu_histogram: Optional[Any]
cpu_frontier_histogram: Optional[Any]
cuda_histogram: Optional[Any]

try:  # pragma: no cover - optional extension
    backend = import_module("packboost._backend")
    cpu_histogram = getattr(backend, "cpu_histogram", None)
    cpu_frontier_histogram = getattr(backend, "cpu_frontier_histogram", None)
    cuda_histogram = getattr(backend, "cuda_histogram", None)
except ImportError:  # pragma: no cover - extension not built
    cpu_histogram = None
    cpu_frontier_histogram = None
    cuda_histogram = None


def cpu_available() -> bool:
    """Return ``True`` if the compiled CPU backend is available."""
    return cpu_histogram is not None or cpu_frontier_histogram is not None


def cuda_available() -> bool:
    """Return ``True`` if the compiled CUDA backend is available."""
    return cuda_histogram is not None


__all__ = [
    "cpu_histogram",
    "cpu_frontier_histogram",
    "cuda_histogram",
    "cpu_available",
    "cuda_available",
]
