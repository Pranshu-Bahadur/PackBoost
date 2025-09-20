"""Data preprocessing utilities for PackBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(slots=True)
class BinningResult:
    """Container capturing preprocessed training data."""

    bins: np.ndarray
    bin_edges: np.ndarray | None
    prebinned: bool


def ensure_numpy(array: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    """Convert ``array`` to a contiguous ``np.ndarray`` of ``float32`` when possible."""

    if isinstance(array, np.ndarray):
        return np.asarray(array)
    if isinstance(array, torch.Tensor):  # pragma: no cover - convenience path
        return array.detach().cpu().numpy()
    return np.asarray(array)


def detect_prebinned(X: np.ndarray, max_bins: int) -> bool:
    """Return ``True`` when ``X`` already stores integer bin ids compatible with ``max_bins``."""

    if not np.issubdtype(X.dtype, np.integer):
        return False
    if X.min(initial=0) < 0:
        return False
    return int(X.max(initial=0)) < max_bins


def quantile_bin(X: np.ndarray, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin ``X`` column-wise into ``max_bins`` buckets."""

    if max_bins > 256:
        raise ValueError("PackBoost currently supports up to 256 bins (uint8 storage)")
    quantiles = np.linspace(0.0, 1.0, max_bins + 1, dtype=np.float64)[1:-1]
    edges = np.quantile(X, quantiles, axis=0, method="linear").astype(np.float32)
    bins = np.empty_like(X, dtype=np.uint8)
    for j in range(X.shape[1]):
        column_edges = edges[:, j]
        bins[:, j] = np.searchsorted(column_edges, X[:, j], side="left").astype(np.uint8)
    np.clip(bins, 0, max_bins - 1, out=bins)
    return bins, edges


def preprocess_features(X: np.ndarray, max_bins: int) -> BinningResult:
    """Return (possibly) quantile-binned features stored as ``uint8``."""

    X_np = ensure_numpy(X)
    if detect_prebinned(X_np, max_bins):
        bins = X_np.astype(np.uint8, copy=False)
        return BinningResult(bins=bins, bin_edges=None, prebinned=True)
    bins, edges = quantile_bin(X_np.astype(np.float32, copy=False), max_bins=max_bins)
    return BinningResult(bins=bins, bin_edges=edges, prebinned=False)


def apply_bins(X: np.ndarray, bin_edges: np.ndarray | None, max_bins: int) -> np.ndarray:
    """Bin ``X`` using previously-computed ``bin_edges``."""

    X_np = ensure_numpy(X)
    if bin_edges is None:
        if not detect_prebinned(X_np, max_bins):
            raise ValueError("Input must already be prebinned when no edges are provided")
        return X_np.astype(np.uint8, copy=False)
    X_float = X_np.astype(np.float32, copy=False)
    bins = np.empty_like(X_float, dtype=np.uint8)
    for j in range(X_float.shape[1]):
        column_edges = bin_edges[:, j]
        bins[:, j] = np.searchsorted(column_edges, X_float[:, j], side="left").astype(np.uint8)
    np.clip(bins, 0, max_bins - 1, out=bins)
    return bins


def build_era_index(era_ids: Iterable[int] | np.ndarray, num_eras: int) -> list[torch.Tensor]:
    """Create per-era row index tensors."""

    era_array = torch.as_tensor(np.asarray(era_ids, dtype=np.int64))
    indices: list[torch.Tensor] = []
    for era in range(num_eras):
        rows = torch.nonzero(era_array == era, as_tuple=False).flatten().to(dtype=torch.int64)
        indices.append(rows)
    return indices
