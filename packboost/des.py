"""Directional era splitting (DES) utilities using the FAST PATH evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .config import PackBoostConfig
from .core import EraShard, NodeState, evaluate_frontier_fastpath, split_shard


@dataclass
class SplitDecision:
    """Container describing the best split for a node."""

    feature: int | None
    threshold: int | None
    score: float
    direction_agreement: float
    left_value: float
    right_value: float
    left_indices: np.ndarray
    right_indices: np.ndarray


def _flatten_shard(shard: EraShard) -> np.ndarray:
    if not shard.rows_per_era:
        return np.empty(0, dtype=np.int32)
    parts = [rows.astype(np.int32, copy=False) for rows in shard.rows_per_era if rows.size]
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(parts).astype(np.int32, copy=False)


def _shard_from_indices(indices: np.ndarray, era_ids: np.ndarray, n_eras: int) -> EraShard:
    if indices.size == 0 or n_eras == 0:
        return EraShard.empty(n_eras)
    node_eras = era_ids[indices]
    order = np.argsort(node_eras, kind="stable")
    sorted_indices = indices[order]
    sorted_eras = node_eras[order]
    counts = np.bincount(sorted_eras, minlength=n_eras)
    offsets = np.empty(n_eras + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts[:n_eras], out=offsets[1:])
    grouped = [sorted_indices[offsets[e] : offsets[e + 1]] for e in range(n_eras)]
    return EraShard.from_grouped_indices(grouped)


def evaluate_frontier(
    *,
    X_binned: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    era_ids: np.ndarray,
    node_indices_list: Iterable[np.ndarray],
    features: Iterable[int],
    config: PackBoostConfig,
) -> List[SplitDecision]:
    node_arrays = [np.asarray(indices, dtype=np.int32) for indices in node_indices_list]
    if not node_arrays:
        return []

    features_arr = np.asarray(list(features), dtype=np.int32)
    era_ids_arr = np.asarray(era_ids, dtype=np.int32)
    n_eras = int(era_ids_arr.max() + 1) if era_ids_arr.size else 0

    node_states: list[NodeState] = []
    for node_id, indices in enumerate(node_arrays):
        shard = _shard_from_indices(indices, era_ids_arr, n_eras)
        grad_sum = float(np.sum(gradients[indices], dtype=np.float64)) if indices.size else 0.0
        hess_sum = float(np.sum(hessians[indices], dtype=np.float64)) if indices.size else 0.0
        node_states.append(
            NodeState(
                node_id=node_id,
                shard=shard,
                grad_sum=grad_sum,
                hess_sum=hess_sum,
                sample_count=int(indices.size),
                depth=0,
            )
        )

    frontier_decisions = evaluate_frontier_fastpath(
        X_binned=X_binned,
        gradients=gradients,
        hessians=hessians,
        node_states=node_states,
        features=features_arr,
        config=config,
    )

    split_decisions: list[SplitDecision] = []
    for node_state, decision in zip(node_states, frontier_decisions):
        if decision.feature is None:
            left_indices = node_arrays[node_state.node_id]
            right_indices = np.empty(0, dtype=np.int32)
            split_decisions.append(
                SplitDecision(
                    feature=None,
                    threshold=None,
                    score=decision.score,
                    direction_agreement=decision.agreement,
                    left_value=decision.base_value,
                    right_value=decision.base_value,
                    left_indices=left_indices.astype(np.int32, copy=False),
                    right_indices=right_indices,
                )
            )
            continue

        left_shard, right_shard = split_shard(
            node_state.shard, X_binned, decision.feature, decision.threshold
        )
        left_indices = _flatten_shard(left_shard)
        right_indices = _flatten_shard(right_shard)

        split_decisions.append(
            SplitDecision(
                feature=decision.feature,
                threshold=decision.threshold,
                score=decision.score,
                direction_agreement=decision.agreement,
                left_value=decision.left_value,
                right_value=decision.right_value,
                left_indices=left_indices,
                right_indices=right_indices,
            )
        )

    return split_decisions


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
