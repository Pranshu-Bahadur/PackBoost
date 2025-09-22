"""Native backend bindings for PackBoost."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    "cpu_available",
    "find_best_splits_batched",
]


def _load_backend() -> ModuleType | None:
    try:
        return import_module("packboost._backend")
    except ImportError:  # pragma: no cover - optional native build
        return None


_BACKEND = _load_backend()


def cpu_available() -> bool:
    """Check whether the native CPU backend is present."""

    if _BACKEND is None:
        return False
    has_flag = getattr(_BACKEND, "_cpu_available", None)
    if has_flag is None:
        return False
    return bool(has_flag())


def find_best_splits_batched(*args: Any, **kwargs: Any):
    """Dispatch into the native CPU splitter when available."""

    if _BACKEND is None:
        raise RuntimeError("Native backend is not built; run setup_native.py first.")
    fn = getattr(_BACKEND, "find_best_splits_batched_cpu", None)
    if fn is None:
        raise RuntimeError("CPU backend is unavailable in this build.")
    return fn(*args, **kwargs)
