"""Directional era splitting (DES) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .config import PackBoostConfig
from .backends import cpu_histogram


@dataclass
class SplitDecision:
    """Container describing the best split for a node."""

    feature: Optional[int]
    threshold: Optional[int]
    score: float
    direction_agreement: float
    left_value: float
    right_value: float
    left_indices: np.ndarray
    right_indices: np.ndarray


# ---------------------------------------------------------------------------
# Public CPU interface
# ---------------------------------------------------------------------------


def evaluate_node_split(
    *,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    era_ids: np.ndarray,
    node_indices: np.ndarray,
    features: Iterable[int],
    config: PackBoostConfig,
) -> SplitDecision:
    """Return the DES-optimal split for ``node_indices`` using the CPU backend."""
    evaluator = _CPUHistogramEvaluator(X_binned, gradients, hessians, era_ids, config)
    return evaluator.evaluate(node_indices=node_indices, features=features)


# ---------------------------------------------------------------------------
# Shared helpers (CPU & GPU)
# ---------------------------------------------------------------------------


def _empty_decision(node_indices: np.ndarray) -> SplitDecision:
    return SplitDecision(
        feature=None,
        threshold=None,
        score=float("-inf"),
        direction_agreement=0.0,
        left_value=0.0,
        right_value=0.0,
        left_indices=node_indices,
        right_indices=np.array([], dtype=np.int32),
    )


def _score_from_histograms(
    *,
    features: np.ndarray,
    node_indices: np.ndarray,
    node_bins: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    config: PackBoostConfig,
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    hist_count: np.ndarray,
) -> SplitDecision:
    """Select the best split from pre-computed histograms."""
    if node_indices.size == 0:
        return _empty_decision(node_indices)

    lambda_l2 = config.lambda_l2
    min_leaf = config.min_samples_leaf

    n_features, max_bins, n_eras = hist_grad.shape
    if n_eras == 0:
        return _empty_decision(node_indices)

    # Prefix sums across bins (axis=1) for left child statistics
    prefix_grad = np.cumsum(hist_grad, axis=1)[:, :-1, :]
    prefix_hess = np.cumsum(hist_hess, axis=1)[:, :-1, :]
    prefix_count = np.cumsum(hist_count, axis=1)[:, :-1, :]

    total_grad = hist_grad.sum(axis=1)[:, None, :]  # (n_features, 1, n_eras)
    total_hess = hist_hess.sum(axis=1)[:, None, :]
    total_count = hist_count.sum(axis=1)[:, None, :]

    suffix_grad = total_grad - prefix_grad
    suffix_hess = total_hess - prefix_hess
    suffix_count = total_count - prefix_count

    parent_score = 0.5 * (total_grad ** 2) / np.maximum(total_hess + lambda_l2, 1e-12)

    denom_left = np.maximum(prefix_hess + lambda_l2, 1e-12)
    denom_right = np.maximum(suffix_hess + lambda_l2, 1e-12)

    gain_left = 0.5 * (prefix_grad ** 2) / denom_left
    gain_right = 0.5 * (suffix_grad ** 2) / denom_right
    gains = gain_left + gain_right - parent_score

    valid_mask = (prefix_count >= min_leaf) & (suffix_count >= min_leaf)
    gains = np.where(valid_mask, gains, np.nan)

    valid_counts = np.sum(~np.isnan(gains), axis=2)
    sum_gain = np.nansum(gains, axis=2)
    mean_gain = np.divide(
        sum_gain,
        valid_counts,
        out=np.full(sum_gain.shape, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )
    mean_for_var = np.where(np.isnan(mean_gain), 0.0, mean_gain)
    sum_sq = np.nansum((gains - mean_for_var[..., None]) ** 2, axis=2)
    variance = np.divide(
        sum_sq,
        valid_counts,
        out=np.zeros_like(sum_sq),
        where=valid_counts > 0,
    )
    std_gain = np.sqrt(variance, dtype=np.float32)
    dro_score = mean_gain - config.lambda_dro * std_gain

    # Directional agreement
    left_value = -prefix_grad / denom_left
    right_value = -suffix_grad / denom_right
    direction = np.where(valid_mask, np.where(left_value >= right_value, 1.0, -1.0), 0.0)
    direction_counts = np.sum(valid_mask, axis=2)
    agreement = np.zeros_like(dro_score)
    nonzero = direction_counts > 0
    agreement[nonzero] = (
        np.abs(np.sum(direction, axis=2)[nonzero]) / direction_counts[nonzero]
    )

    final_score = dro_score + config.direction_weight * agreement
    final_score = np.where(np.isnan(final_score), -np.inf, final_score)

    if not np.isfinite(final_score).any():
        return _empty_decision(node_indices)

    best_feature_idx, best_threshold_idx = np.unravel_index(
        np.nanargmax(final_score), final_score.shape
    )
    best_score = final_score[best_feature_idx, best_threshold_idx]
    best_agreement = agreement[best_feature_idx, best_threshold_idx]

    feature = int(features[best_feature_idx])
    threshold = int(best_threshold_idx)

    feature_bins = node_bins[:, best_feature_idx]
    left_mask = feature_bins <= threshold
    right_mask = ~left_mask

    left_indices = node_indices[left_mask]
    right_indices = node_indices[right_mask]

    if left_indices.size < min_leaf or right_indices.size < min_leaf:
        return _empty_decision(node_indices)

    g_left_total = gradients[left_indices].sum()
    h_left_total = hessians[left_indices].sum()
    g_right_total = gradients[right_indices].sum()
    h_right_total = hessians[right_indices].sum()

    left_value_scalar = -g_left_total / (h_left_total + lambda_l2)
    right_value_scalar = -g_right_total / (h_right_total + lambda_l2)

    return SplitDecision(
        feature=feature,
        threshold=threshold,
        score=float(best_score),
        direction_agreement=float(best_agreement),
        left_value=float(left_value_scalar),
        right_value=float(right_value_scalar),
        left_indices=left_indices.astype(np.int32, copy=False),
        right_indices=right_indices.astype(np.int32, copy=False),
    )


# ---------------------------------------------------------------------------
# CPU histogram builder
# ---------------------------------------------------------------------------


class _CPUHistogramEvaluator:
    """Helper that accumulates histograms on the CPU."""

    def __init__(
        self,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        config: PackBoostConfig,
    ) -> None:
        self.X_binned = X_binned
        self.gradients = gradients
        self.hessians = hessians
        self.era_ids = era_ids
        self.config = config

    def evaluate(self, node_indices: np.ndarray, features: Iterable[int]) -> SplitDecision:
        if node_indices.size == 0:
            return _empty_decision(node_indices)

        features_arr = np.array(list(features), dtype=np.int32)
        node_bins = self.X_binned[np.ix_(node_indices, features_arr)]
        gradients_node = self.gradients[node_indices]
        hessians_node = self.hessians[node_indices]

        unique_eras, era_inverse = np.unique(self.era_ids[node_indices], return_inverse=True)
        n_eras = unique_eras.size
        if n_eras == 0:
            return _empty_decision(node_indices)

        max_bins = self.config.max_bins
        if cpu_histogram is not None:
            hist_grad, hist_hess, hist_count = cpu_histogram(
                node_bins,
                gradients_node,
                hessians_node,
                era_inverse,
                max_bins,
                n_eras,
            )
        else:
            hist_grad = np.zeros((features_arr.size, max_bins, n_eras), dtype=np.float32)
            hist_hess = np.zeros_like(hist_grad)
            hist_count = np.zeros((features_arr.size, max_bins, n_eras), dtype=np.int32)

            flat_multiplier = n_eras
            for feature_idx in range(features_arr.size):
                feature_bins = node_bins[:, feature_idx].astype(np.int32)
                keys = feature_bins * flat_multiplier + era_inverse

                grad_flat = np.bincount(
                    keys,
                    weights=gradients_node,
                    minlength=max_bins * n_eras,
                )
                hess_flat = np.bincount(
                    keys,
                    weights=hessians_node,
                    minlength=max_bins * n_eras,
                )
                count_flat = np.bincount(keys, minlength=max_bins * n_eras)

                hist_grad[feature_idx] = grad_flat.reshape(max_bins, n_eras)
                hist_hess[feature_idx] = hess_flat.reshape(max_bins, n_eras)
                hist_count[feature_idx] = count_flat.reshape(max_bins, n_eras)

        return _score_from_histograms(
            features=features_arr,
            node_indices=node_indices,
            node_bins=node_bins,
            gradients=self.gradients,
            hessians=self.hessians,
            config=self.config,
            hist_grad=hist_grad,
            hist_hess=hist_hess,
            hist_count=hist_count,
        )


# ---------------------------------------------------------------------------
# GPU hook
# ---------------------------------------------------------------------------


def evaluate_node_split_from_hist(
    *,
    features: np.ndarray,
    node_indices: np.ndarray,
    node_bins: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    config: PackBoostConfig,
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    hist_count: np.ndarray,
) -> SplitDecision:
    """Expose histogram scoring for GPU backends."""
    return _score_from_histograms(
        features=features,
        node_indices=node_indices,
        node_bins=node_bins,
        gradients=gradients,
        hessians=hessians,
        config=config,
        hist_grad=hist_grad,
        hist_hess=hist_hess,
        hist_count=hist_count,
    )


__all__ = ["SplitDecision", "evaluate_node_split", "evaluate_node_split_from_hist"]
