"""FAST PATH frontier evaluator implementing era tiling + Welford DES."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..config import PackBoostConfig
from ..backends import cpu_frontier_evaluate, cuda_frontier_evaluate
from .shards import EraShard, FrontierDecision, NodeState


def _allocate_workspace(
    n_nodes: int, n_features: int, max_bins: int, era_tile: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (n_nodes, n_features, max_bins, era_tile)
    grad = np.zeros(shape, dtype=np.float32)
    hess = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int32)
    return grad, hess, count


def _build_frontier_layout(
    shards: Sequence[EraShard],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened row indices and offsets for backend kernels."""

    n_nodes = len(shards)
    if n_nodes == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    n_eras = shards[0].n_eras
    base_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    era_offsets = np.zeros((n_nodes, n_eras + 1), dtype=np.int32)
    parts: list[np.ndarray] = []
    cursor = 0

    for node_idx, shard in enumerate(shards):
        base_offsets[node_idx] = cursor
        offset = 0
        for era_idx in range(n_eras):
            rows = np.asarray(shard.rows_per_era[era_idx], dtype=np.int32)
            parts.append(rows)
            offset += int(rows.size)
            era_offsets[node_idx, era_idx + 1] = offset
        cursor += offset

    base_offsets[-1] = cursor
    indices = (
        np.concatenate(parts).astype(np.int32, copy=False)
        if any(part.size for part in parts)
        else np.empty(0, dtype=np.int32)
    )
    return indices, base_offsets, era_offsets.reshape(-1)


def _evaluate_frontier_backend(
    *,
    device: str,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    shards: Sequence[EraShard],
    features: np.ndarray,
    base_values: np.ndarray,
    parent_grad: np.ndarray,
    parent_hess: np.ndarray,
    parent_count: np.ndarray,
    lambda_l2: float,
    lambda_dro: float,
    direction_weight: float,
    min_leaf: int,
    max_bins: int,
) -> list[FrontierDecision] | None:
    if device == "cuda":
        backend = cuda_frontier_evaluate or cpu_frontier_evaluate
    else:
        backend = cpu_frontier_evaluate

    if backend is None:
        return None

    if not shards:
        return []

    n_nodes = len(shards)
    n_eras = shards[0].n_eras
    node_indices, node_base_offsets, node_era_offsets = _build_frontier_layout(shards)

    result = backend(
        X_binned,
        gradients,
        hessians,
        node_indices,
        node_base_offsets,
        node_era_offsets,
        features,
        parent_grad.astype(np.float32, copy=False),
        parent_hess.astype(np.float32, copy=False),
        parent_count.astype(np.int32, copy=False),
        max_bins,
        n_eras,
        float(lambda_l2),
        float(lambda_dro),
        int(min_leaf),
        float(direction_weight),
    )

    (
        best_feature_idx,
        best_threshold,
        scores,
        agreements,
        left_grad_arr,
        left_hess_arr,
        left_count_arr,
    ) = result

    decisions: list[FrontierDecision] = []
    for node_idx in range(n_nodes):
        feature_pos = int(best_feature_idx[node_idx])
        score = float(scores[node_idx])

        if feature_pos < 0:
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

        threshold = int(best_threshold[node_idx])
        feature_choice = int(features[feature_pos])
        left_grad = float(left_grad_arr[node_idx])
        left_hess = float(left_hess_arr[node_idx])
        left_count = int(left_count_arr[node_idx])
        parent_grad_sum = float(parent_grad[node_idx])
        parent_hess_sum = float(parent_hess[node_idx])
        parent_count_sum = int(parent_count[node_idx])
        right_grad = parent_grad_sum - left_grad
        right_hess = parent_hess_sum - left_hess
        right_count = parent_count_sum - left_count

        denom_left = left_hess + lambda_l2
        denom_right = right_hess + lambda_l2
        eps = 1e-12
        left_value = (
            -left_grad / max(denom_left, eps)
            if left_count >= min_leaf and left_count > 0
            else 0.0
        )
        right_value = (
            -right_grad / max(denom_right, eps)
            if right_count >= min_leaf and right_count > 0
            else 0.0
        )

        decisions.append(
            FrontierDecision(
                feature=feature_choice,
                threshold=threshold,
                score=score,
                agreement=float(agreements[node_idx]),
                left_value=float(left_value),
                right_value=float(right_value),
                base_value=float(base_values[node_idx]),
                left_grad=left_grad,
                left_hess=left_hess,
                left_count=left_count,
            )
        )

    return decisions


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

    X_binned_arr = np.asarray(X_binned, dtype=np.uint8)
    gradients_arr = np.asarray(gradients, dtype=np.float32)
    hessians_arr = np.asarray(hessians, dtype=np.float32)

    backend_device = "cuda" if config.device == "cuda" else "cpu"

    backend_decisions = _evaluate_frontier_backend(
        device=backend_device,
        X_binned=X_binned_arr,
        gradients=gradients_arr,
        hessians=hessians_arr,
        shards=shards,
        features=features_arr,
        base_values=base_values,
        parent_grad=parent_grad,
        parent_hess=parent_hess,
        parent_count=parent_count,
        lambda_l2=lambda_l2,
        lambda_dro=lambda_dro,
        direction_weight=direction_weight,
        min_leaf=min_leaf,
        max_bins=max_bins,
    )
    if backend_decisions is not None:
        return backend_decisions

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

    thresholds_shape = (n_nodes, n_features, thresholds)
    gain_sum = np.zeros(thresholds_shape, dtype=np.float64)
    gain_sumsq = np.zeros_like(gain_sum)
    era_counts = np.zeros(thresholds_shape, dtype=np.int32)
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

        mask = parent_count_tile > 0
        mask_float = mask.astype(np.float32, copy=False)

        gain_sum += np.sum(gains * mask_float, axis=3, dtype=np.float64)
        gain_sumsq += np.sum((gains * gains) * mask_float, axis=3, dtype=np.float64)
        era_counts += np.sum(mask, axis=3, dtype=np.int32)

        left_grad_tot += np.sum(left_grad_tile, axis=3)
        left_hess_tot += np.sum(left_hess_tile, axis=3)
        left_count_tot += np.sum(left_count_tile, axis=3, dtype=np.int32)

        left_pred = -left_grad_tile / np.maximum(safe_left_hess, 1e-12)
        right_pred = -right_grad_tile / np.maximum(safe_right_hess, 1e-12)
        direction = np.where(left_pred >= right_pred, 1.0, -1.0) * mask_float
        direction_sum += np.sum(direction, axis=3)
        direction_count += np.sum(mask, axis=3, dtype=np.int32)

    valid_counts = era_counts > 0
    mean_gain = np.zeros_like(gain_sum, dtype=np.float32)
    mean_gain[valid_counts] = (
        gain_sum[valid_counts] / era_counts[valid_counts]
    ).astype(np.float32)
    variance = np.zeros_like(mean_gain)
    variance[valid_counts] = (
        gain_sumsq[valid_counts] / era_counts[valid_counts]
    ).astype(np.float32) - mean_gain[valid_counts] ** 2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance, dtype=np.float32)
    dro_score = mean_gain - lambda_dro * std

    agreement = np.zeros_like(mean_gain)
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
