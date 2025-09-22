"""Helpers for interacting with the native CPU backend."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["available", "find_best_splits_batched"]


def _load() -> Any | None:
    try:
        return import_module("packboost._backend")
    except ImportError:  # pragma: no cover - optional native build
        return None


_MOD = _load()


def available() -> bool:
    if _MOD is None:
        return False
    flag = getattr(_MOD, "_cpu_available", None)
    if flag is None:
        return False
    return bool(flag())


def find_best_splits_batched(*args: Any, **kwargs: Any):
    if _MOD is None:
        raise RuntimeError("Native backend not built; run setup_native.py")
    fn = getattr(_MOD, "find_best_splits_batched_cpu", None)
    if fn is None:
        raise RuntimeError("CPU backend is unavailable in this build.")
    return fn(*args, **kwargs)
