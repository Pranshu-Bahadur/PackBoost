"""Directional era splitting (DES) scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .config import PackBoostConfig


@dataclass
class SplitDecision:
    """Best split metadata for a node."""

    feature: Optional[int]
    threshold: Optional[int]
    score: float
    direction_agreement: float
    left_value: float
    right_value: float
    left_indices: np.ndarray
    right_indices: np.ndarray


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
    """Return the DES-optimal split for ``node_indices``."""
    if node_indices.size == 0:
        return _empty_decision(node_indices)

    lambda_l2 = config.lambda_l2
    min_leaf = config.min_samples_leaf

    total_grad = gradients[node_indices].sum()
    total_hess = hessians[node_indices].sum()
    base_value = -total_grad / (total_hess + lambda_l2)

    best = SplitDecision(
        feature=None,
        threshold=None,
        score=float("-inf"),
        direction_agreement=0.0,
        left_value=base_value,
        right_value=base_value,
        left_indices=node_indices,
        right_indices=np.array([], dtype=np.int32),
    )

    unique_eras = np.unique(era_ids[node_indices])
    if unique_eras.size == 0:
        return best

    thresholds = np.arange(0, config.max_bins - 1, dtype=np.int32)
    for feature in features:
        bins = X_binned[node_indices, feature]
        gain_sum = np.zeros_like(thresholds, dtype=np.float64)
        gain_sumsq = np.zeros_like(thresholds, dtype=np.float64)
        gain_count = np.zeros_like(thresholds, dtype=np.int32)
        dir_sum = np.zeros_like(thresholds, dtype=np.float64)
        dir_count = np.zeros_like(thresholds, dtype=np.int32)

        for start in range(0, unique_eras.size, config.era_tile_size):
            era_tile = unique_eras[start : start + config.era_tile_size]
            for era in era_tile:
                mask = era_ids[node_indices] == era
                if not np.any(mask):
                    continue
                era_bins = bins[mask]
                grad = gradients[node_indices][mask]
                hess = hessians[node_indices][mask]

                counts = np.bincount(era_bins, minlength=config.max_bins)
                grad_sum = np.bincount(era_bins, weights=grad, minlength=config.max_bins)
                hess_sum = np.bincount(era_bins, weights=hess, minlength=config.max_bins)

                if counts.sum() < 2 * min_leaf:
                    continue

                prefix_grad = np.cumsum(grad_sum[:-1])
                prefix_hess = np.cumsum(hess_sum[:-1])
                prefix_count = np.cumsum(counts[:-1])

                total_grad = grad_sum.sum()
                total_hess = hess_sum.sum()
                total_count = counts.sum()

                suffix_grad = total_grad - prefix_grad
                suffix_hess = total_hess - prefix_hess
                suffix_count = total_count - prefix_count

                parent_score = 0.5 * (total_grad ** 2) / (total_hess + lambda_l2)

                for t_idx, threshold in enumerate(thresholds):
                    left_count = prefix_count[t_idx]
                    right_count = suffix_count[t_idx]
                    if left_count < min_leaf or right_count < min_leaf:
                        continue

                    g_left = prefix_grad[t_idx]
                    g_right = suffix_grad[t_idx]
                    h_left = prefix_hess[t_idx]
                    h_right = suffix_hess[t_idx]

                    gain_left = 0.5 * (g_left ** 2) / (h_left + lambda_l2)
                    gain_right = 0.5 * (g_right ** 2) / (h_right + lambda_l2)
                    gain = gain_left + gain_right - parent_score

                    if not np.isfinite(gain):
                        continue

                    gain_sum[t_idx] += gain
                    gain_sumsq[t_idx] += gain * gain
                    gain_count[t_idx] += 1

                    v_left = -g_left / (h_left + lambda_l2)
                    v_right = -g_right / (h_right + lambda_l2)
                    dir_sum[t_idx] += 1.0 if v_left >= v_right else -1.0
                    dir_count[t_idx] += 1

        best = _update_best_split(
            best=best,
            feature=feature,
            thresholds=thresholds,
            gain_sum=gain_sum,
            gain_sumsq=gain_sumsq,
            gain_count=gain_count,
            dir_sum=dir_sum,
            dir_count=dir_count,
            bins=bins,
            gradients=gradients,
            hessians=hessians,
            node_indices=node_indices,
            lambda_l2=lambda_l2,
            config=config,
        )

    return best


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


def _update_best_split(
    *,
    best: SplitDecision,
    feature: int,
    thresholds: np.ndarray,
    gain_sum: np.ndarray,
    gain_sumsq: np.ndarray,
    gain_count: np.ndarray,
    dir_sum: np.ndarray,
    dir_count: np.ndarray,
    bins: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_indices: np.ndarray,
    lambda_l2: float,
    config: PackBoostConfig,
) -> SplitDecision:
    for t_idx, threshold in enumerate(thresholds):
        count = gain_count[t_idx]
        if count == 0:
            continue
        mean = gain_sum[t_idx] / count
        variance = max(gain_sumsq[t_idx] / count - mean * mean, 0.0)
        std = variance ** 0.5
        dro_score = mean - config.lambda_dro * std
        agreement = abs(dir_sum[t_idx]) / dir_count[t_idx] if dir_count[t_idx] else 0.0
        final_score = dro_score + config.direction_weight * agreement

        if final_score <= best.score:
            continue

        left_mask = bins <= threshold
        right_mask = ~left_mask
        left_indices = node_indices[left_mask]
        right_indices = node_indices[right_mask]

        g_left_total = gradients[left_indices].sum()
        h_left_total = hessians[left_indices].sum()
        g_right_total = gradients[right_indices].sum()
        h_right_total = hessians[right_indices].sum()

        left_value = -g_left_total / (h_left_total + lambda_l2)
        right_value = -g_right_total / (h_right_total + lambda_l2)

        best = SplitDecision(
            feature=int(feature),
            threshold=int(threshold),
            score=float(final_score),
            direction_agreement=float(agreement),
            left_value=float(left_value),
            right_value=float(right_value),
            left_indices=left_indices.astype(np.int32, copy=False),
            right_indices=right_indices.astype(np.int32, copy=False),
        )
    return best
