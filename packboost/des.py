"""Directional era splitting (DES) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .config import PackBoostConfig
from .backends import cpu_histogram, cpu_frontier_histogram


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
    decisions = evaluate_frontier(
        X_binned=X_binned,
        gradients=gradients,
        hessians=hessians,
        era_ids=era_ids,
        node_indices_list=[node_indices],
        features=features,
        config=config,
    )
    return decisions[0]


def evaluate_frontier(
    *,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    era_ids: np.ndarray,
    node_indices_list: Iterable[np.ndarray],
    features: Iterable[int],
    config: PackBoostConfig,
) -> list[SplitDecision]:
    """Evaluate a batch of frontier nodes and return split decisions."""
    node_indices_list = [np.asarray(idx, dtype=np.int32) for idx in node_indices_list]
    if not node_indices_list:
        return []

    features_arr = np.asarray(list(features), dtype=np.int32)
    if features_arr.size == 0:
        # No features to split on: all nodes become leaves
        decisions: list[SplitDecision] = []
        for idx in node_indices_list:
            base_val = float(-(gradients[idx].sum()) / (hessians[idx].sum() + config.lambda_l2)) if idx.size else 0.0
            decisions.append(
                SplitDecision(
                    feature=None,
                    threshold=None,
                    score=float("-inf"),
                    direction_agreement=0.0,
                    left_value=base_val,
                    right_value=base_val,
                    left_indices=idx,
                    right_indices=np.empty(0, dtype=np.int32),
                )
            )
        return decisions

    if cpu_frontier_histogram is not None:
        offsets = np.zeros(len(node_indices_list) + 1, dtype=np.int32)
        for i, idx in enumerate(node_indices_list, start=1):
            offsets[i] = offsets[i - 1] + idx.size
        concatenated = np.concatenate(node_indices_list) if offsets[-1] > 0 else np.empty(0, dtype=np.int32)
        hist_grad, hist_hess, hist_count = cpu_frontier_histogram(
            X_binned,
            concatenated,
            offsets,
            features_arr,
            gradients,
            hessians,
            era_ids,
            config.max_bins,
            int(era_ids.max() + 1),
        )
        scores = _score_from_histograms(
            hist_grad=hist_grad,
            hist_hess=hist_hess,
            hist_count=hist_count,
            config=config,
            lambda_l2=config.lambda_l2,
            min_leaf=config.min_samples_leaf,
        )

        decisions: list[SplitDecision] = []
        node_idx = 0
        for samples in node_indices_list:
            best_feature = int(scores["feature"][node_idx])
            threshold = int(scores["threshold"][node_idx])
            score = float(scores["score"][node_idx])
            agreement = float(scores["agreement"][node_idx])
            left_value = float(scores["left_value"][node_idx])
            right_value = float(scores["right_value"][node_idx])
            base_value = float(scores["base_value"][node_idx])

            if best_feature < 0 or score <= 0 or samples.size == 0:
                decisions.append(
                    SplitDecision(
                        feature=None,
                        threshold=None,
                        score=score,
                        direction_agreement=0.0,
                        left_value=base_value,
                        right_value=base_value,
                        left_indices=samples,
                        right_indices=np.empty(0, dtype=np.int32),
                    )
                )
            else:
                feature_idx = features_arr[best_feature]
                node_bins = X_binned[samples, feature_idx]
                left_mask = node_bins <= threshold
                left_indices = samples[left_mask]
                right_indices = samples[~left_mask]

                if left_indices.size < config.min_samples_leaf or right_indices.size < config.min_samples_leaf:
                    decisions.append(
                        SplitDecision(
                            feature=None,
                            threshold=None,
                            score=score,
                            direction_agreement=0.0,
                            left_value=base_value,
                            right_value=base_value,
                            left_indices=samples,
                            right_indices=np.empty(0, dtype=np.int32),
                        )
                    )
                else:
                    decisions.append(
                        SplitDecision(
                            feature=int(feature_idx),
                            threshold=int(threshold),
                            score=score,
                            direction_agreement=agreement,
                            left_value=left_value,
                            right_value=right_value,
                            left_indices=left_indices.astype(np.int32, copy=False),
                            right_indices=right_indices.astype(np.int32, copy=False),
                        )
                    )
            node_idx += 1

        return decisions

    # Fallback to per-node evaluation if native backend is unavailable
    fallback = []
    for samples in node_indices_list:
        fallback.append(
            _CPUHistogramEvaluator(X_binned, gradients, hessians, era_ids, config).evaluate(
                node_indices=samples,
                features=features,
            )
        )
    return fallback


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
    hist_grad: np.ndarray,
    hist_hess: np.ndarray,
    hist_count: np.ndarray,
    config: PackBoostConfig,
    lambda_l2: float,
    min_leaf: int,
):
    hist_grad = np.asarray(hist_grad, dtype=np.float32)
    hist_hess = np.asarray(hist_hess, dtype=np.float32)
    hist_count = np.asarray(hist_count, dtype=np.int32)

    if hist_grad.ndim == 3:
        hist_grad = hist_grad[None, ...]
        hist_hess = hist_hess[None, ...]
        hist_count = hist_count[None, ...]

    n_nodes, n_features, max_bins, n_eras = hist_grad.shape
    if max_bins <= 1 or n_eras == 0:
        return {
            "feature": np.full(n_nodes, -1, dtype=np.int32),
            "threshold": np.zeros(n_nodes, dtype=np.int32),
            "score": np.full(n_nodes, -np.inf, dtype=np.float32),
            "agreement": np.zeros(n_nodes, dtype=np.float32),
            "left_value": np.zeros(n_nodes, dtype=np.float32),
            "right_value": np.zeros(n_nodes, dtype=np.float32),
            "base_value": np.zeros(n_nodes, dtype=np.float32),
        }

    thresholds = max_bins - 1

    total_grad = hist_grad.sum(axis=2, keepdims=True)
    total_hess = hist_hess.sum(axis=2, keepdims=True)
    parent_grad = total_grad.sum(axis=3).squeeze(2)
    parent_hess = total_hess.sum(axis=3).squeeze(2)
    base_value = -parent_grad[:, 0] / (parent_hess[:, 0] + lambda_l2)

    left_grad = np.cumsum(hist_grad, axis=2)[:, :, :-1, :]
    left_hess = np.cumsum(hist_hess, axis=2)[:, :, :-1, :]
    left_count = np.cumsum(hist_count, axis=2)[:, :, :-1, :]

    right_grad = total_grad - left_grad
    right_hess = total_hess - left_hess
    total_count = hist_count.sum(axis=2, keepdims=True)
    right_count = total_count - left_count

    safe_left_hess = left_hess + lambda_l2
    safe_right_hess = right_hess + lambda_l2
    parent_safe_hess = total_hess + lambda_l2

    gain_left = 0.5 * (left_grad ** 2) / np.maximum(safe_left_hess, 1e-12)
    gain_right = 0.5 * (right_grad ** 2) / np.maximum(safe_right_hess, 1e-12)
    parent_gain = 0.5 * (total_grad ** 2) / np.maximum(parent_safe_hess, 1e-12)
    gains = gain_left + gain_right - parent_gain

    left_total = left_count.sum(axis=3)
    right_total = right_count.sum(axis=3)
    valid_mask = (left_total >= min_leaf) & (right_total >= min_leaf)
    valid_mask_expanded = np.broadcast_to(valid_mask[..., None], gains.shape)
    gains = np.where(valid_mask_expanded, gains, np.nan)

    sum_gain = np.nansum(gains, axis=3)
    counts = valid_mask_expanded.sum(axis=3)
    mean_gain = np.divide(sum_gain, counts, out=np.zeros_like(sum_gain, dtype=np.float32), where=counts > 0)

    diff = np.where(valid_mask_expanded, gains - mean_gain[..., None], 0.0)
    sum_sq = np.sum(diff ** 2, axis=3)
    variance = np.divide(sum_sq, counts, out=np.zeros_like(sum_sq, dtype=np.float32), where=counts > 0)
    std_gain = np.sqrt(variance, dtype=np.float32)
    dro_score = mean_gain - config.lambda_dro * std_gain

    left_pred = -left_grad / np.maximum(safe_left_hess, 1e-12)
    right_pred = -right_grad / np.maximum(safe_right_hess, 1e-12)
    direction = np.where(valid_mask_expanded, np.where(left_pred >= right_pred, 1.0, -1.0), 0.0)
    direction_sum = np.nansum(direction, axis=3)
    direction_counts = counts
    agreement = np.zeros_like(dro_score)
    mask_counts = direction_counts > 0
    agreement[mask_counts] = np.abs(direction_sum[mask_counts]) / direction_counts[mask_counts]

    final_score = dro_score + config.direction_weight * agreement
    final_score = np.where(np.isnan(final_score), -np.inf, final_score)
    final_score = np.where(valid_mask, final_score, -np.inf)

    flat_scores = final_score.reshape(n_nodes, -1)
    best_flat = np.argmax(flat_scores, axis=1)
    best_score = flat_scores[np.arange(n_nodes), best_flat]
    valid_nodes = np.isfinite(best_score)

    best_feature_idx = best_flat // thresholds
    best_threshold_idx = best_flat % thresholds

    sum_left_grad = left_grad.sum(axis=3)
    sum_left_hess = left_hess.sum(axis=3)
    sum_right_grad = right_grad.sum(axis=3)
    sum_right_hess = right_hess.sum(axis=3)

    node_idx = np.arange(n_nodes)
    best_left_grad = sum_left_grad[node_idx, best_feature_idx, best_threshold_idx]
    best_left_hess = sum_left_hess[node_idx, best_feature_idx, best_threshold_idx]
    best_right_grad = sum_right_grad[node_idx, best_feature_idx, best_threshold_idx]
    best_right_hess = sum_right_hess[node_idx, best_feature_idx, best_threshold_idx]
    best_agreement = agreement.reshape(n_nodes, -1)[np.arange(n_nodes), best_flat]

    left_value = -best_left_grad / (best_left_hess + lambda_l2)
    right_value = -best_right_grad / (best_right_hess + lambda_l2)

    best_feature_idx = np.where(valid_nodes, best_feature_idx, -1)
    best_threshold_idx = np.where(valid_nodes, best_threshold_idx, 0)

    return {
        "feature": best_feature_idx.astype(np.int32),
        "threshold": best_threshold_idx.astype(np.int32),
        "score": best_score.astype(np.float32),
        "agreement": best_agreement.astype(np.float32),
        "left_value": left_value.astype(np.float32),
        "right_value": right_value.astype(np.float32),
        "base_value": base_value.astype(np.float32),
    }


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
        self.X_binned = X_binned.astype(np.uint8, copy=False)
        self.gradients = gradients
        self.hessians = hessians
        self.era_ids = era_ids.astype(np.int16, copy=False)
        self.config = config

    def evaluate(self, node_indices: np.ndarray, features: Iterable[int]) -> SplitDecision:
        if node_indices.size == 0:
            return _empty_decision(node_indices)

        features_arr = np.array(list(features), dtype=np.int32)
        node_bins = self.X_binned[np.ix_(node_indices, features_arr)].astype(np.uint8, copy=False)
        gradients_node = self.gradients[node_indices]
        hessians_node = self.hessians[node_indices]

        unique_eras, era_inverse = np.unique(self.era_ids[node_indices], return_inverse=True)
        n_eras = unique_eras.size
        if n_eras == 0:
            return _empty_decision(node_indices)

        era_inverse = era_inverse.astype(np.int16, copy=False)

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
            era_inv32 = era_inverse.astype(np.int32, copy=False)
            for feature_idx in range(features_arr.size):
                feature_bins = node_bins[:, feature_idx].astype(np.int32, copy=False)
                keys = feature_bins * flat_multiplier + era_inv32

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

        scores = _score_from_histograms(
            hist_grad=hist_grad[None, ...],
            hist_hess=hist_hess[None, ...],
            hist_count=hist_count[None, ...],
            config=self.config,
            lambda_l2=self.config.lambda_l2,
            min_leaf=self.config.min_samples_leaf,
        )

        best_feature = int(scores["feature"][0])
        threshold = int(scores["threshold"][0])
        score = float(scores["score"][0])
        agreement = float(scores["agreement"][0])
        left_value = float(scores["left_value"][0])
        right_value = float(scores["right_value"][0])
        base_value = float(scores["base_value"][0])

        if best_feature < 0 or score <= 0:
            return SplitDecision(
                feature=None,
                threshold=None,
                score=score,
                direction_agreement=0.0,
                left_value=base_value,
                right_value=base_value,
                left_indices=node_indices,
                right_indices=np.empty(0, dtype=np.int32),
            )

        feature = int(features_arr[best_feature])
        node_bins_feature = node_bins[:, best_feature]
        left_mask = node_bins_feature <= threshold
        left_indices = node_indices[left_mask]
        right_indices = node_indices[~left_mask]

        if left_indices.size < self.config.min_samples_leaf or right_indices.size < self.config.min_samples_leaf:
            return SplitDecision(
                feature=None,
                threshold=None,
                score=score,
                direction_agreement=0.0,
                left_value=base_value,
                right_value=base_value,
                left_indices=node_indices,
                right_indices=np.empty(0, dtype=np.int32),
            )

        return SplitDecision(
            feature=feature,
            threshold=int(threshold),
            score=score,
            direction_agreement=agreement,
            left_value=left_value,
            right_value=right_value,
            left_indices=left_indices.astype(np.int32, copy=False),
            right_indices=right_indices.astype(np.int32, copy=False),
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
    """Return the DES-optimal split given pre-computed histograms."""
    scores = _score_from_histograms(
        hist_grad=hist_grad[None, ...],
        hist_hess=hist_hess[None, ...],
        hist_count=hist_count[None, ...],
        config=config,
        lambda_l2=config.lambda_l2,
        min_leaf=config.min_samples_leaf,
    )

    best_feature = int(scores["feature"][0])
    threshold = int(scores["threshold"][0])
    score = float(scores["score"][0])
    agreement = float(scores["agreement"][0])
    left_value = float(scores["left_value"][0])
    right_value = float(scores["right_value"][0])
    base_value = float(scores["base_value"][0])

    if best_feature < 0 or score <= 0 or node_indices.size == 0:
        return SplitDecision(
            feature=None,
            threshold=None,
            score=score,
            direction_agreement=0.0,
            left_value=base_value,
            right_value=base_value,
            left_indices=node_indices,
            right_indices=np.empty(0, dtype=np.int32),
        )

    feature = int(features[best_feature])
    bins_feature = node_bins[:, best_feature]
    left_mask = bins_feature <= threshold
    left_indices = node_indices[left_mask]
    right_indices = node_indices[~left_mask]

    if left_indices.size < config.min_samples_leaf or right_indices.size < config.min_samples_leaf:
        return SplitDecision(
            feature=None,
            threshold=None,
            score=score,
            direction_agreement=0.0,
            left_value=base_value,
            right_value=base_value,
            left_indices=node_indices,
            right_indices=np.empty(0, dtype=np.int32),
        )

    return SplitDecision(
        feature=feature,
        threshold=threshold,
        score=score,
        direction_agreement=agreement,
        left_value=left_value,
        right_value=right_value,
        left_indices=left_indices.astype(np.int32, copy=False),
        right_indices=right_indices.astype(np.int32, copy=False),
    )


__all__ = ["SplitDecision", "evaluate_node_split", "evaluate_node_split_from_hist"]
