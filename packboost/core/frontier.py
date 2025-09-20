"""FAST PATH frontier evaluator implementing era tiling + Welford DES."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..config import PackBoostConfig
from .shards import EraShard, FrontierDecision, NodeState


def _allocate_workspace(
    n_nodes: int, n_features: int, max_bins: int, era_tile: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (n_nodes, n_features, max_bins, era_tile)
    grad = np.zeros(shape, dtype=np.float32)
    hess = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int32)
    return grad, hess, count


def evaluate_frontier_fastpath(
    *,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_states: Sequence[NodeState],
    features: Iterable[int],
    config: PackBoostConfig,
) -> list[FrontierDecision]:
    """Evaluate ``node_states`` using the FAST PATH DES algorithm."""
    if not node_states:
        return []

    features_arr = np.asarray(list(features), dtype=np.int32)
    n_nodes = len(node_states)
    n_features = features_arr.size
    max_bins = int(config.max_bins)
    thresholds = max_bins - 1

    lambda_l2 = float(config.lambda_l2)
    lambda_dro = float(config.lambda_dro)
    direction_weight = float(config.direction_weight)
    min_leaf = int(config.min_samples_leaf)

    base_values = np.empty(n_nodes, dtype=np.float32)
    parent_grad = np.empty(n_nodes, dtype=np.float32)
    parent_hess = np.empty(n_nodes, dtype=np.float32)
    parent_count = np.empty(n_nodes, dtype=np.int32)
    shards: list[EraShard] = []
    for idx, state in enumerate(node_states):
        shards.append(state.shard)
        parent_grad[idx] = float(state.grad_sum)
        parent_hess[idx] = float(state.hess_sum)
        parent_count[idx] = int(state.sample_count)
        denom = state.hess_sum + lambda_l2
        base_values[idx] = 0.0 if denom == 0 else float(-state.grad_sum / denom)

    decisions: list[FrontierDecision] = []

    if n_features == 0 or thresholds <= 0:
        for node_idx in range(n_nodes):
            base_val = float(base_values[node_idx])
            decisions.append(
                FrontierDecision(
                    feature=None,
                    threshold=0,
                    score=float("-inf"),
                    agreement=0.0,
                    left_value=base_val,
                    right_value=base_val,
                    base_value=base_val,
                    left_grad=0.0,
                    left_hess=0.0,
                    left_count=0,
                )
            )
        return decisions

    n_eras = shards[0].n_eras
    if any(shard.n_eras != n_eras for shard in shards):
        raise ValueError("All shards must have the same number of eras")

    gradients_arr = np.asarray(gradients, dtype=np.float32)
    hessians_arr = np.asarray(hessians, dtype=np.float32)
    X_binned_arr = np.asarray(X_binned, dtype=np.uint8)

    thresholds_shape = (n_nodes, n_features, thresholds)
    means = np.zeros(thresholds_shape, dtype=np.float32)
    m2 = np.zeros_like(means)
    counts = np.zeros(thresholds_shape, dtype=np.int32)
    counts_float = np.zeros(thresholds_shape, dtype=np.float32)
    left_grad_tot = np.zeros(thresholds_shape, dtype=np.float32)
    left_hess_tot = np.zeros(thresholds_shape, dtype=np.float32)
    left_count_tot = np.zeros(thresholds_shape, dtype=np.int32)
    direction_sum = np.zeros(thresholds_shape, dtype=np.float32)
    direction_count = np.zeros(thresholds_shape, dtype=np.int32)

    era_tile = int(min(config.era_tile_size, n_eras))
    if era_tile <= 0:
        era_tile = n_eras

    hist_grad, hist_hess, hist_count = _allocate_workspace(n_nodes, n_features, max_bins, era_tile)
    feature_indexer = np.arange(n_features, dtype=np.intp)[:, None]

    for era_start in range(0, n_eras, era_tile):
        era_end = min(era_start + era_tile, n_eras)
        tile_len = era_end - era_start
        hist_grad.fill(0.0)
        hist_hess.fill(0.0)
        hist_count.fill(0)

        for node_idx, shard in enumerate(shards):
            rows_per_era = shard.rows_per_era
            if parent_count[node_idx] == 0:
                continue
            for local_offset, era in enumerate(range(era_start, era_end)):
                rows = rows_per_era[era]
                if rows.size == 0:
                    continue
                bins_tile = X_binned_arr[rows][:, features_arr]
                if bins_tile.size == 0:
                    continue
                bins_T = bins_tile.T
                grad_rows = gradients_arr[rows][None, :]
                hess_rows = hessians_arr[rows][None, :]

                hist_grad_view = hist_grad[node_idx, :, :, local_offset]
                hist_hess_view = hist_hess[node_idx, :, :, local_offset]
                hist_count_view = hist_count[node_idx, :, :, local_offset]

                np.add.at(hist_grad_view, (feature_indexer, bins_T), grad_rows)
                np.add.at(hist_hess_view, (feature_indexer, bins_T), hess_rows)
                np.add.at(hist_count_view, (feature_indexer, bins_T), 1)

        grad_tile = hist_grad[:, :, :, :tile_len]
        hess_tile = hist_hess[:, :, :, :tile_len]
        count_tile = hist_count[:, :, :, :tile_len]

        total_grad = grad_tile.sum(axis=2, keepdims=True)
        total_hess = hess_tile.sum(axis=2, keepdims=True)
        total_count = count_tile.sum(axis=2, keepdims=True)

        left_grad_tile = np.cumsum(grad_tile, axis=2)[:, :, :-1, :]
        left_hess_tile = np.cumsum(hess_tile, axis=2)[:, :, :-1, :]
        left_count_tile = np.cumsum(count_tile, axis=2)[:, :, :-1, :]

        parent_grad_tile = np.broadcast_to(total_grad, left_grad_tile.shape)
        parent_hess_tile = np.broadcast_to(total_hess, left_hess_tile.shape)
        parent_count_tile = np.broadcast_to(total_count, left_count_tile.shape)

        right_grad_tile = parent_grad_tile - left_grad_tile
        right_hess_tile = parent_hess_tile - left_hess_tile
        right_count_tile = parent_count_tile - left_count_tile

        safe_left_hess = left_hess_tile + lambda_l2
        safe_right_hess = right_hess_tile + lambda_l2
        safe_parent_hess = parent_hess_tile + lambda_l2

        gain_left = 0.5 * (left_grad_tile ** 2) / np.maximum(safe_left_hess, 1e-12)
        gain_right = 0.5 * (right_grad_tile ** 2) / np.maximum(safe_right_hess, 1e-12)
        parent_gain = 0.5 * (parent_grad_tile ** 2) / np.maximum(safe_parent_hess, 1e-12)
        gains = gain_left + gain_right - parent_gain

        for local_offset in range(tile_len):
            gains_slice = gains[:, :, :, local_offset]
            counts += 1
            counts_float += 1.0
            delta = gains_slice - means
            means += delta / counts_float
            m2 += delta * (gains_slice - means)

            left_grad_slice = left_grad_tile[:, :, :, local_offset]
            left_hess_slice = left_hess_tile[:, :, :, local_offset]
            left_count_slice = left_count_tile[:, :, :, local_offset]
            left_grad_tot += left_grad_slice
            left_hess_tot += left_hess_slice
            left_count_tot += left_count_slice.astype(np.int32, copy=False)

            right_grad_slice = right_grad_tile[:, :, :, local_offset]
            right_hess_slice = right_hess_tile[:, :, :, local_offset]

            safe_left = left_hess_slice + lambda_l2
            safe_right = right_hess_slice + lambda_l2
            left_pred = -left_grad_slice / np.maximum(safe_left, 1e-12)
            right_pred = -right_grad_slice / np.maximum(safe_right, 1e-12)
            direction = np.where(left_pred >= right_pred, 1.0, -1.0)
            direction_sum += direction
            direction_count += 1

    valid_counts = counts > 0
    mean_gain = np.where(valid_counts, means, 0.0)
    variance = np.zeros_like(means)
    nonzero_counts = np.where(valid_counts, counts_float, 1.0)
    variance[valid_counts] = m2[valid_counts] / nonzero_counts[valid_counts]
    std = np.sqrt(variance, dtype=np.float32)
    dro_score = mean_gain - lambda_dro * std

    agreement = np.zeros_like(means)
    nonzero_dir = direction_count > 0
    agreement[nonzero_dir] = np.abs(direction_sum[nonzero_dir]) / direction_count[nonzero_dir]

    parent_count_broadcast = parent_count[:, None, None]
    left_count_tot = left_count_tot.astype(np.int32, copy=False)
    right_count_tot = parent_count_broadcast - left_count_tot
    valid_mask = (
        valid_counts
        & (left_count_tot >= min_leaf)
        & (right_count_tot >= min_leaf)
    )

    final_score = np.where(valid_mask, dro_score + direction_weight * agreement, -np.inf)
    agreement = np.where(valid_mask, agreement, 0.0)

    node_range = np.arange(n_nodes)
    flat_scores = final_score.reshape(n_nodes, -1)
    best_flat = np.argmax(flat_scores, axis=1)
    best_score = flat_scores[node_range, best_flat]
    valid_nodes = np.isfinite(best_score)

    best_feature_idx = best_flat // thresholds
    best_threshold_idx = best_flat % thresholds

    best_agreement = agreement.reshape(n_nodes, -1)[node_range, best_flat]
    best_left_grad = left_grad_tot[node_range, best_feature_idx, best_threshold_idx]
    best_left_hess = left_hess_tot[node_range, best_feature_idx, best_threshold_idx]
    best_left_count = left_count_tot[node_range, best_feature_idx, best_threshold_idx]

    decisions.clear()
    for node_idx in range(n_nodes):
        if not valid_nodes[node_idx]:
            base_val = float(base_values[node_idx])
            decisions.append(
                FrontierDecision(
                    feature=None,
                    threshold=0,
                    score=float("-inf"),
                    agreement=0.0,
                    left_value=base_val,
                    right_value=base_val,
                    base_value=base_val,
                    left_grad=0.0,
                    left_hess=0.0,
                    left_count=0,
                )
            )
            continue

        feature_choice = int(features_arr[best_feature_idx[node_idx]])
        threshold_choice = int(best_threshold_idx[node_idx])
        score_choice = float(best_score[node_idx])
        agreement_choice = float(best_agreement[node_idx])
        left_grad_choice = float(best_left_grad[node_idx])
        left_hess_choice = float(best_left_hess[node_idx])
        left_count_choice = int(best_left_count[node_idx])

        parent_grad_sum = float(parent_grad[node_idx])
        parent_hess_sum = float(parent_hess[node_idx])

        right_grad_choice = parent_grad_sum - left_grad_choice
        right_hess_choice = parent_hess_sum - left_hess_choice

        left_value = -left_grad_choice / (left_hess_choice + lambda_l2) if left_count_choice else 0.0
        right_count_choice = parent_count[node_idx] - left_count_choice
        right_value = (
            -right_grad_choice / (right_hess_choice + lambda_l2)
            if right_count_choice
            else 0.0
        )

        base_val = float(base_values[node_idx])

        decisions.append(
            FrontierDecision(
                feature=feature_choice,
                threshold=threshold_choice,
                score=score_choice,
                agreement=agreement_choice,
                left_value=float(left_value),
                right_value=float(right_value),
                base_value=base_val,
                left_grad=left_grad_choice,
                left_hess=left_hess_choice,
                left_count=left_count_choice,
            )
        )

    return decisions
