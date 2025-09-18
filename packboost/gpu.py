"""GPU-accelerated helpers built on top of CuPy."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import PackBoostConfig
from .des import SplitDecision, _empty_decision, evaluate_node_split_from_hist

try:  # pragma: no cover - optional dependency
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


def has_cuda() -> bool:
    """Return ``True`` if a CUDA-capable device is available."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def evaluate_node_split_gpu(
    *,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    era_ids: np.ndarray,
    node_indices: np.ndarray,
    features: Iterable[int],
    config: PackBoostConfig,
) -> SplitDecision:
    """GPU-backed DES scoring using CuPy for histogram accumulation."""
    if not has_cuda():  # pragma: no cover - requires GPU
        raise RuntimeError("CUDA device not available")

    features_arr = np.array(list(features), dtype=np.int32)
    if features_arr.size == 0 or node_indices.size == 0:
        return _empty_decision(node_indices)

    node_bins = X_binned[np.ix_(node_indices, features_arr)]
    gradients_node = gradients[node_indices]
    hessians_node = hessians[node_indices]

    unique_eras, era_inverse = np.unique(era_ids[node_indices], return_inverse=True)
    n_eras = unique_eras.size
    if n_eras == 0:
        return _empty_decision(node_indices)

    # Transfer to GPU
    bins_gpu = cp.asarray(node_bins, dtype=cp.uint8)
    grad_gpu = cp.asarray(gradients_node, dtype=cp.float32)
    hess_gpu = cp.asarray(hessians_node, dtype=cp.float32)
    era_inv_gpu = cp.asarray(era_inverse, dtype=cp.int32)

    hist_grad = cp.zeros((features_arr.size, config.max_bins, n_eras), dtype=cp.float32)
    hist_hess = cp.zeros_like(hist_grad)
    hist_count = cp.zeros((features_arr.size, config.max_bins, n_eras), dtype=cp.int32)

    for feature_idx in range(features_arr.size):  # pragma: no branch - executed on GPU
        fbins = bins_gpu[:, feature_idx].astype(cp.int32)
        cp.add.at(hist_grad[feature_idx], (fbins, era_inv_gpu), grad_gpu)
        cp.add.at(hist_hess[feature_idx], (fbins, era_inv_gpu), hess_gpu)
        cp.add.at(hist_count[feature_idx], (fbins, era_inv_gpu), 1)

    hist_grad_cpu = cp.asnumpy(hist_grad)
    hist_hess_cpu = cp.asnumpy(hist_hess)
    hist_count_cpu = cp.asnumpy(hist_count)

    return evaluate_node_split_from_hist(
        features=features_arr,
        node_indices=node_indices,
        node_bins=node_bins,
        gradients=gradients,
        hessians=hessians,
        config=config,
        hist_grad=hist_grad_cpu,
        hist_hess=hist_hess_cpu,
        hist_count=hist_count_cpu,
    )


__all__ = ["evaluate_node_split_gpu", "has_cuda"]
