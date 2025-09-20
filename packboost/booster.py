"""PackBoost implementation following the FAST PATH design."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, List, Sequence

import numpy as np

from .config import PackBoostConfig
from .core import (
    EraShard,
    NodeState,
    evaluate_frontier_fastpath,
    group_rows_by_era,
    split_shard,
)
from .des import SplitDecision, evaluate_frontier
from .model import PackBoostModel, Tree, TreeNode
from .utils.binning import apply_binning, ensure_prebinned, quantile_binning


class PackBoost:
    """Gradient boosting with directional era splitting."""

    def __init__(self, config: PackBoostConfig) -> None:
        self.config = config
        self._model: PackBoostModel | None = None
        self._auto_prebinned = False
        # Backwards-compatibility attributes relied upon by the tests and
        # existing integration code. They are no longer used by the FAST PATH
        # trainer but keeping them avoids AttributeErrors.
        self._num_eras: int | None = None

    @property
    def model(self) -> PackBoostModel:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        return self._model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        era_ids: Sequence[int] | None,
        *,
        num_rounds: int = 10,
        eval_set: tuple[np.ndarray, np.ndarray, Sequence[int] | None] | None = None,
        callbacks: Sequence[Any] | None = None,
        log_evaluation: int | None = None,
    ) -> "PackBoost":
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X_array.shape
        y = np.asarray(y, dtype=np.float32)
        if era_ids is None:
            era_ids = np.zeros(n_samples, dtype=np.int64)
        else:
            era_ids = np.asarray(era_ids, dtype=np.int64)

        if y.shape[0] != n_samples or era_ids.shape[0] != n_samples:
            raise ValueError("X, y, and era_ids must share the same number of rows")

        self.config.validate(n_features)

        prebinned_candidate: np.ndarray | None = None
        auto_prebinned = False
        if not self.config.prebinned and np.issubdtype(X_array.dtype, np.integer):
            if X_array.size == 0:
                auto_prebinned = True
                prebinned_candidate = X_array.astype(np.uint8, copy=False)
            else:
                min_val = int(X_array.min())
                max_val = int(X_array.max())
                if min_val >= 0 and max_val < self.config.max_bins:
                    auto_prebinned = True
                    prebinned_candidate = X_array.astype(np.uint8, copy=False)
        if auto_prebinned:
            self.config = replace(self.config, prebinned=True)
        self._auto_prebinned = auto_prebinned

        if self.config.prebinned:
            if prebinned_candidate is not None:
                X_binned = prebinned_candidate
            else:
                X_binned = ensure_prebinned(X_array, self.config.max_bins)
            bin_edges: np.ndarray | None = None
        else:
            X_float = X_array.astype(np.float32, copy=False)
            X_binned, bin_edges = quantile_binning(
                X_float, self.config.max_bins, random_state=self.config.random_state
            )

        unique_eras, era_inverse = np.unique(era_ids, return_inverse=True)
        era_encoded = era_inverse.astype(np.int32)
        n_eras = int(unique_eras.size)
        self._num_eras = n_eras

        rows_by_era = group_rows_by_era(era_encoded, n_eras)
        root_shard = EraShard.from_grouped_indices(rows_by_era)

        initial_prediction = float(np.mean(y)) if y.size else 0.0
        predictions = np.full(n_samples, initial_prediction, dtype=np.float32)
        gradients = predictions - y
        hessians = np.ones_like(gradients)

        eval_data: tuple[np.ndarray, np.ndarray] | None = None
        predictions_valid: np.ndarray | None = None
        if eval_set is not None:
            X_val, y_val, era_val = eval_set
            X_val_array = np.asarray(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)
            if era_val is None:
                era_val = np.zeros(X_val_array.shape[0], dtype=np.int64)
            else:
                era_val = np.asarray(era_val, dtype=np.int64)
            if X_val_array.shape[0] != y_val.shape[0] or X_val_array.shape[0] != era_val.shape[0]:
                raise ValueError("eval_set components must have the same number of rows")
            if self.config.prebinned:
                X_val_binned = ensure_prebinned(X_val_array, self.config.max_bins)
            else:
                X_val_binned = apply_binning(X_val_array.astype(np.float32, copy=False), bin_edges)
            eval_data = (X_val_binned, y_val)
            predictions_valid = np.full(X_val_array.shape[0], initial_prediction, dtype=np.float32)

        trees: List[Tree] = []
        rng = np.random.default_rng(self.config.random_state)
        tree_weight = self.config.learning_rate / self.config.pack_size

        def _nan_safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            if y_true.size == 0:
                return float("nan")
            corr_matrix = np.corrcoef(y_true, y_pred)
            corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else float("nan")
            if not np.isfinite(corr):
                return 0.0
            return corr

        for round_idx in range(num_rounds):
            np.subtract(predictions, y, out=gradients)
            grad_sum_total = float(np.sum(gradients, dtype=np.float64))
            hess_sum_total = float(np.sum(hessians, dtype=np.float64))

            pack_trees: List[Tree] = []
            pack_frontiers: List[List[NodeState]] = []

            for _ in range(self.config.pack_size):
                tree = Tree()
                root_id = tree.add_node(TreeNode(is_leaf=False, prediction=0.0, depth=0))
                pack_trees.append(tree)
                pack_frontiers.append(
                    [
                        NodeState(
                            node_id=root_id,
                            shard=root_shard,
                            grad_sum=grad_sum_total,
                            hess_sum=hess_sum_total,
                            sample_count=n_samples,
                            depth=0,
                        )
                    ]
                )

            for depth in range(self.config.max_depth):
                feature_subset = self._sample_features(rng, n_features)
                if feature_subset.size == 0:
                    break

                pack_frontiers = self._grow_depth(
                    feature_subset=feature_subset,
                    pack_frontiers=pack_frontiers,
                    pack_trees=pack_trees,
                    X_binned=X_binned,
                    gradients=gradients,
                    hessians=hessians,
                )

                if not any(pack_frontiers):
                    break

            for tree_idx, tree in enumerate(pack_trees):
                for state in pack_frontiers[tree_idx]:
                    denom = state.hess_sum + self.config.lambda_l2
                    value = 0.0 if denom == 0 else float(-state.grad_sum / denom)
                    tree.ensure_leaf(state.node_id, value)

            for tree in pack_trees:
                tree_pred = tree.predict_binned(X_binned)
                predictions += tree_weight * tree_pred
                if eval_data is not None and predictions_valid is not None:
                    X_val_binned, _ = eval_data
                    predictions_valid += tree_weight * tree.predict_binned(X_val_binned)
                trees.append(tree)

            train_corr = _nan_safe_corr(y, predictions)
            valid_corr = float("nan")
            if eval_data is not None and predictions_valid is not None:
                _, y_val = eval_data
                valid_corr = _nan_safe_corr(y_val, predictions_valid)

            if log_evaluation is not None and log_evaluation > 0 and (round_idx + 1) % log_evaluation == 0:
                parts = []
                if np.isfinite(train_corr):
                    parts.append(f"train corr = {train_corr:.4f}")
                if eval_data is not None:
                    if np.isfinite(valid_corr):
                        parts.append(f"valid corr = {valid_corr:.4f}")
                    else:
                        parts.append("valid corr = nan")
                message = ", ".join(parts) if parts else "(no metrics)"
                print(f"Round {round_idx + 1}: {message}")

            if callbacks:
                info_valid_corr = float(valid_corr) if np.isfinite(valid_corr) else float("nan")
                info = {
                    "round": round_idx + 1,
                    "train_corr": float(train_corr),
                    "valid_corr": info_valid_corr,
                    "predictions_train": predictions.copy(),
                    "train_targets": y,
                }
                if predictions_valid is not None:
                    info["predictions_valid"] = predictions_valid.copy()
                if eval_data is not None:
                    _, y_val = eval_data
                    info["validation_targets"] = y_val
                for callback in callbacks:
                    if hasattr(callback, "on_round"):
                        callback.on_round(self, info)
                    else:
                        try:
                            callback(self, info)
                        except TypeError:
                            callback(self)

        self._model = PackBoostModel(
            config=self.config,
            bin_edges=bin_edges,
            initial_prediction=initial_prediction,
            trees=trees,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = self.model
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array")
        if model.config.prebinned:
            X_binned = ensure_prebinned(X_array, model.config.max_bins)
        else:
            if model.bin_edges is None:
                raise RuntimeError("Model missing bin edges for float input")
            X_binned = apply_binning(X_array.astype(np.float32, copy=False), model.bin_edges)
        return model.predict_binned(X_binned)

    def _sample_features(self, rng: np.random.Generator, n_features: int) -> np.ndarray:
        n_candidates = max(1, int(np.ceil(self.config.layer_feature_fraction * n_features)))
        return np.sort(rng.choice(n_features, size=n_candidates, replace=False))

    def _grow_depth(
        self,
        *,
        feature_subset: np.ndarray,
        pack_frontiers: List[List[NodeState]],
        pack_trees: List[Tree],
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> List[List[NodeState]]:
        """Expand all trees in the current pack at the given depth."""

        total_states = sum(len(states) for states in pack_frontiers)
        if total_states == 0:
            return [[] for _ in pack_frontiers]

        node_states: list[NodeState] = []
        slices: list[tuple[int, int, int]] = []
        for tree_idx, states in enumerate(pack_frontiers):
            if not states:
                slices.append((tree_idx, len(node_states), 0))
                continue
            start = len(node_states)
            node_states.extend(states)
            slices.append((tree_idx, start, len(states)))

        decisions = evaluate_frontier_fastpath(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            node_states=node_states,
            features=feature_subset,
            config=self.config,
        )

        next_frontiers: List[List[NodeState]] = [[] for _ in pack_frontiers]

        for tree_idx, start, count in slices:
            if count == 0:
                continue
            tree = pack_trees[tree_idx]
            states = pack_frontiers[tree_idx]
            tree_decisions = decisions[start : start + count]

            for state, decision in zip(states, tree_decisions):
                node = tree.nodes[state.node_id]
                right_count = state.sample_count - decision.left_count
                if (
                    decision.feature is None
                    or decision.score <= 0
                    or decision.left_count < self.config.min_samples_leaf
                    or right_count < self.config.min_samples_leaf
                ):
                    tree.ensure_leaf(state.node_id, decision.base_value)
                    continue

                node.feature = decision.feature
                node.threshold = decision.threshold
                node.is_leaf = False

                left_node = TreeNode(
                    is_leaf=False,
                    prediction=decision.left_value,
                    depth=state.depth + 1,
                )
                right_node = TreeNode(
                    is_leaf=False,
                    prediction=decision.right_value,
                    depth=state.depth + 1,
                )
                left_id = tree.add_node(left_node)
                right_id = tree.add_node(right_node)
                node.left = left_id
                node.right = right_id

                left_shard, right_shard = split_shard(
                    state.shard, X_binned, decision.feature, decision.threshold
                )

                left_state = NodeState(
                    node_id=left_id,
                    shard=left_shard,
                    grad_sum=decision.left_grad,
                    hess_sum=decision.left_hess,
                    sample_count=decision.left_count,
                    depth=state.depth + 1,
                )
                right_state = NodeState(
                    node_id=right_id,
                    shard=right_shard,
                    grad_sum=state.grad_sum - decision.left_grad,
                    hess_sum=state.hess_sum - decision.left_hess,
                    sample_count=right_count,
                    depth=state.depth + 1,
                )

                next_frontiers[tree_idx].extend((left_state, right_state))

        return next_frontiers

    # ------------------------------------------------------------------
    # Compatibility helpers for the legacy native bindings interface.
    # The reworked FAST PATH trainer no longer relies on the native CPU/CUDA
    # frontiers, but the public tests (and some downstream tooling) still call
    # these helpers directly. They now delegate to the unified Python DES
    # implementation to keep behaviour identical without rebuilding the native
    # packing structures.

    @staticmethod
    def _prepare_frontier_batch(
        node_samples: List[np.ndarray],
        era_ids: np.ndarray,
    ) -> dict[str, list[np.ndarray]]:
        """Return a minimal batch descriptor compatible with old callers."""

        _ = era_ids  # era_ids were previously used to pre-group rows; unused now.
        samples = [np.asarray(samples, dtype=np.int32) for samples in node_samples]
        return {"samples": samples}

    def _evaluate_frontier_batch(
        self,
        *,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        node_samples: list[np.ndarray] | None = None,
        batch: dict[str, list[np.ndarray]] | None = None,
        features: Iterable[int],
    ) -> List[SplitDecision]:
        """Mimic the legacy native frontier evaluation interface."""

        if node_samples is None:
            if batch is None:
                raise ValueError("node_samples or batch must be provided")
            node_samples = batch.get("samples")
            if node_samples is None:
                raise ValueError("batch is missing 'samples' entry")

        sample_arrays = [np.asarray(arr, dtype=np.int32) for arr in node_samples]

        return evaluate_frontier(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            node_indices_list=sample_arrays,
            features=features,
            config=self.config,
        )
