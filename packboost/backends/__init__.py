"""Native backend bindings for PackBoost."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Optional

__all__ = [
    "cpu_available",
    "find_best_splits_batched",
    "cuda_available",
    "find_best_splits_batched_cuda",
    "partition_frontier_cuda",
    "predict_bins_cuda",
    "predict_pack_cuda",
]


def _load_backend() -> Optional[ModuleType]:
    """
    Try to import the compiled extension module.

    Expected export names (if present):
      - _cpu_available() -> bool
      - find_best_splits_batched_cpu(...)
      - _cuda_available() -> bool
      - find_best_splits_batched_cuda(...)
      - partition_frontier_cuda(...)
      - predict_bins_cuda(...)
      - predict_pack_cuda(...)
    """
    try:
        return import_module("packboost._backend")
    except Exception:
        return None


_BACKEND = _load_backend()


# ------------------------------ helpers ---------------------------------


def _get_attr_callable(mod: Optional[ModuleType], name: str):
    """Return attribute if it exists *and* is callable; otherwise None."""
    if mod is None:
        return None
    fn = getattr(mod, name, None)
    return fn if callable(fn) else None


def _bool_from_flag(mod: Optional[ModuleType], name: str) -> bool:
    """
    Some builds expose *_available flags as functions.
    Only call them if they are callable; otherwise fall back to probing
    for at least one expected callable symbol of that family.
    """
    if mod is None:
        return False
    flag = getattr(mod, name, None)
    if callable(flag):
        try:
            return bool(flag())
        except Exception:
            # If the flag throws, conservatively report unavailable
            return False
    return False


# ------------------------------ CPU API ---------------------------------


def cpu_available() -> bool:
    """Check whether the native CPU backend is present."""
    if _BACKEND is None:
        return False

    # Preferred: explicit flag
    if _bool_from_flag(_BACKEND, "_cpu_available"):
        return True

    # Fallback: probe for expected symbols
    return _get_attr_callable(_BACKEND, "find_best_splits_batched_cpu") is not None


def find_best_splits_batched(*args: Any, **kwargs: Any):
    """Dispatch into the native CPU splitter when available."""
    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")

    fn = _get_attr_callable(_BACKEND, "find_best_splits_batched_cpu")
    if fn is None:
        raise RuntimeError("CPU backend is unavailable in this build.")
    return fn(*args, **kwargs)


# ------------------------------ CUDA API --------------------------------


def cuda_available() -> bool:
    """Check whether the native CUDA backend is present."""
    if _BACKEND is None:
        return False

    # Preferred: explicit flag
    if _bool_from_flag(_BACKEND, "_cuda_available"):
        return True

    # Fallback: probe for any CUDA callable
    for name in (
        "find_best_splits_batched_cuda",
        "partition_frontier_cuda",
        "predict_bins_cuda",
        "predict_pack_cuda",
    ):
        if _get_attr_callable(_BACKEND, name) is not None:
            return True
    return False


def find_best_splits_batched_cuda(*args: Any, **kwargs: Any):
    """Dispatch into the native CUDA splitter when available."""
    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")

    fn = _get_attr_callable(_BACKEND, "find_best_splits_batched_cuda")
    if fn is None:
        raise RuntimeError("CUDA backend is unavailable in this build.")
    return fn(*args, **kwargs)


def partition_frontier_cuda(*args: Any, **kwargs: Any):
    """Batched frontier partition (count + scatter) on CUDA."""
    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")

    fn = _get_attr_callable(_BACKEND, "partition_frontier_cuda")
    if fn is None:
        raise RuntimeError("CUDA partition_frontier is unavailable in this build.")
    return fn(*args, **kwargs)


def predict_bins_cuda(*args: Any, **kwargs: Any):
    """Fast CUDA route for a single tree over binned features."""
    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")

    fn = _get_attr_callable(_BACKEND, "predict_bins_cuda")
    if fn is None:
        raise RuntimeError("CUDA predict_bins is unavailable in this build.")
    return fn(*args, **kwargs)


def predict_pack_cuda(*args: Any, **kwargs: Any):
    """Warp-parallel CUDA predictor over a pack of trees (sum and scale)."""
    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")

    fn = _get_attr_callable(_BACKEND, "predict_pack_cuda")
    if fn is None:
        raise RuntimeError("CUDA predict_pack is unavailable in this build.")
    return fn(*args, **kwargs)
