"""Quantile-based binning utilities."""

from __future__ import annotations

import numpy as np


def ensure_prebinned(X: np.ndarray, max_bins: int) -> np.ndarray:
    """Validate and return pre-binned features as ``uint8``."""

    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("Pre-binned features must be a 2D array")

    if arr.size == 0:
        return arr.astype(np.uint8, copy=False)

    if not np.issubdtype(arr.dtype, np.integer):
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("Pre-binned features must be integer-valued")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Pre-binned features must be finite")
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, atol=0.0):
            raise ValueError("Pre-binned features contain non-integer values")
        arr = rounded.astype(np.int64, copy=False)

    if arr.min() < 0 or arr.max() >= max_bins:
        raise ValueError("Pre-binned features must lie within [0, max_bins)")

    return arr.astype(np.uint8, copy=False)


def quantile_binning(
    X: np.ndarray,
    max_bins: int,
    subsample: int | None = 200_000,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quantile bins and transform the data.

    Parameters
    ----------
    X: np.ndarray
        Feature matrix of shape (n_samples, n_features).
    max_bins: int
        Number of quantile bins per feature.
    subsample: int | None, default=200_000
        Optional subsample size used to estimate quantiles for large data.
    random_state: int | None
        Seed for subsampling when used.

    Returns
    -------
    X_binned: np.ndarray
        Array of shape (n_samples, n_features) with dtype ``uint8`` representing
        bin indices in ``[0, max_bins-1]``.
    bin_edges: np.ndarray
        Array of shape (n_features, max_bins + 1) containing bin edges used to
        transform future data.
    """
    if max_bins > 255:
        raise ValueError("max_bins cannot exceed 255 when using uint8 storage")

    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    if subsample is not None and subsample < n_samples:
        indices = rng.choice(n_samples, size=subsample, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    quantiles = np.linspace(0, 1, max_bins + 1)
    bin_edges = np.empty((n_features, max_bins + 1), dtype=np.float32)

    for j in range(n_features):
        feature = X_sample[:, j]
        edges = np.quantile(feature, quantiles, method="linear")
        # Ensure monotonic edges and a small epsilon spread for constant features
        edges[0] = -np.inf
        edges[-1] = np.inf
        for k in range(1, len(edges) - 1):
            if edges[k] == edges[k - 1]:
                edges[k] = np.nextafter(edges[k], np.inf)
        bin_edges[j] = edges

    X_binned = np.empty_like(X, dtype=np.uint8)
    for j in range(n_features):
        bins = bin_edges[j, 1:-1]
        X_binned[:, j] = np.digitize(X[:, j], bins, right=False)
    return X_binned, bin_edges


def apply_binning(X: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin new data using previously computed bin edges.

    Parameters
    ----------
    X: np.ndarray
        Input features of shape (n_samples, n_features).
    bin_edges: np.ndarray
        Bin edges obtained from :func:`quantile_binning` with shape
        (n_features, n_bins + 1).

    Returns
    -------
    np.ndarray
        Binned representation with dtype ``uint8``.
    """
    n_features = X.shape[1]
    X_binned = np.empty_like(X, dtype=np.uint8)
    for j in range(n_features):
        bins = bin_edges[j, 1:-1]
        X_binned[:, j] = np.digitize(X[:, j], bins, right=False)
    return X_binned
