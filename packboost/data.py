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
    max_val = int(X.max(initial=0))
    if max_val >= max_bins:
        return False
    return max_val <= 127


def quantile_bin(X: np.ndarray, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin ``X`` column-wise into ``max_bins`` buckets using ``int8`` bins."""

    if max_bins > 128:
        raise ValueError("PackBoost currently supports up to 128 bins (int8 storage)")
    quantiles = np.linspace(0.0, 1.0, max_bins + 1, dtype=np.float64)[1:-1]
    edges = np.quantile(X, quantiles, axis=0, method="linear").astype(np.float32)
    bins = np.empty_like(X, dtype=np.int8)
    for j in range(X.shape[1]):
        column_edges = edges[:, j]
        col = np.searchsorted(column_edges, X[:, j], side="left").astype(np.int16)
        np.clip(col, 0, max_bins - 1, out=col)
        bins[:, j] = col.astype(np.int8, copy=False)
    return bins, edges


def preprocess_features(
    X: np.ndarray,
    max_bins: int,
    *,
    assume_prebinned: bool = False,
) -> BinningResult:
    """Return (possibly) quantile-binned features stored as ``int8``.

    Parameters
    ----------
    X:
        Feature matrix. When ``assume_prebinned`` is ``True`` this must already
        contain integer bin identifiers in ``[0, max_bins)``.
    max_bins:
        Number of histogram bins per feature.
    assume_prebinned:
        Skip quantile binning altogether and treat ``X`` as already binned.
    """

    X_np = ensure_numpy(X)
    if assume_prebinned:
        if not detect_prebinned(X_np, max_bins):
            raise ValueError(
                "prebinned=True requires X to contain integer bins within [0, max_bins)."
            )
        bins = X_np.astype(np.int8, copy=False)
        return BinningResult(bins=bins, bin_edges=None, prebinned=True)

    if detect_prebinned(X_np, max_bins):
        bins = X_np.astype(np.int8, copy=False)
        return BinningResult(bins=bins, bin_edges=None, prebinned=True)

    bins, edges = quantile_bin(X_np.astype(np.float32, copy=False), max_bins=max_bins)
    return BinningResult(bins=bins, bin_edges=edges, prebinned=False)


def apply_bins(X: np.ndarray, bin_edges: np.ndarray | None, max_bins: int) -> np.ndarray:
    """Bin ``X`` using previously-computed ``bin_edges``."""

    X_np = ensure_numpy(X)
    if bin_edges is None:
        if not detect_prebinned(X_np, max_bins):
            raise ValueError("Input must already be prebinned when no edges are provided")
        return X_np.astype(np.int8, copy=False)
    X_float = X_np.astype(np.float32, copy=False)
    bins = np.empty_like(X_float, dtype=np.int8)
    for j in range(X_float.shape[1]):
        column_edges = bin_edges[:, j]
        col = np.searchsorted(column_edges, X_float[:, j], side="left").astype(np.int16)
        np.clip(col, 0, max_bins - 1, out=col)
        bins[:, j] = col.astype(np.int8, copy=False)
    return bins


def build_era_index(
    era_ids: Iterable[int] | np.ndarray,
    num_eras: int | None,
) -> list[torch.Tensor]:
    """Create per-era row index tensors."""

    era_np = np.asarray(era_ids, dtype=np.int16)
    era_array = torch.as_tensor(era_np, dtype=torch.int16)
    if era_array.ndim != 1:
        raise ValueError("era_ids must be 1D")
    if num_eras is None:
        if era_array.numel() == 0:
            num_eras = 1
        else:
            num_eras = int(era_array.max().item()) + 1
    num_eras = int(num_eras)
    indices: list[torch.Tensor] = []
    for era in range(num_eras):
        rows = torch.nonzero(era_array == era, as_tuple=False).flatten().to(dtype=torch.int64)
        indices.append(rows)
    return indices
