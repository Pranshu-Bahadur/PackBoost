"""CPU reference implementation of PackBoost with DES scoring."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .config import PackBoostConfig
from .des import SplitDecision, evaluate_frontier, evaluate_node_split, evaluate_node_split_from_hist
from .model import PackBoostModel, Tree, TreeNode
from .utils.binning import apply_binning, ensure_prebinned, quantile_binning
from .backends import (
    cpu_available,
    cpu_frontier_evaluate,
    cuda_available,
    cuda_frontier_evaluate,
    cuda_histogram,
)


class PackBoost:
    """Gradient boosting with directional era splitting on the CPU."""

    def __init__(self, config: PackBoostConfig) -> None:
        self.config = config
        self._model: PackBoostModel | None = None
        self._use_gpu: bool = False
        self._num_eras: int | None = None
        self._auto_prebinned: bool = False

    @property
    def model(self) -> PackBoostModel:
        """Return the fitted model."""
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
        """Fit PackBoost to the provided data."""
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array")

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

        y = np.asarray(y, dtype=np.float32)
        if era_ids is None:
            era_ids = np.zeros(X_array.shape[0], dtype=np.int64)
        else:
            era_ids = np.asarray(era_ids, dtype=np.int64)

        if X_array.shape[0] != y.shape[0] or X_array.shape[0] != era_ids.shape[0]:
            raise ValueError("X, y, and era_ids must have the same number of rows")

        n_samples, n_features = X_array.shape
        self.config.validate(n_features)

        unique_eras, era_ids = np.unique(era_ids, return_inverse=True)
        era_ids = era_ids.astype(np.int16)
        self._num_eras = int(unique_eras.size)

        if self.config.device == "cuda" and not self._should_use_gpu():
            raise RuntimeError(
                "CUDA device requested but the native CUDA backend is not available."
            )
        self._use_gpu = self._should_use_gpu()

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

        initial_prediction = float(np.mean(y))
        predictions = np.full(n_samples, initial_prediction, dtype=np.float32)
        hessians = np.ones(n_samples, dtype=np.float32)

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

        X_val_binned: np.ndarray | None = None
        y_val_array: np.ndarray | None = None
        if eval_data is not None:
            X_val_binned, y_val_array = eval_data

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
            gradients = predictions - y
            pack_trees, pack_frontiers, pack_samples = self._initialise_pack(n_samples)

            for depth in range(self.config.max_depth):
                feature_subset = self._sample_features(rng, n_features)
                next_frontiers = self._grow_depth(
                    trees=pack_trees,
                    frontiers=pack_frontiers,
                    samples=pack_samples,
                    feature_subset=feature_subset,
                    gradients=gradients,
                    hessians=hessians,
                    era_ids=era_ids,
                    X_binned=X_binned,
                    depth=depth,
                )
                if not any(next_frontiers):
                    break
                pack_frontiers = next_frontiers

            self._finalise_trees(pack_trees)

            for tree in pack_trees:
                tree_pred = tree.predict_binned(X_binned)
                predictions += tree_weight * tree_pred
                if X_val_binned is not None and predictions_valid is not None:
                    predictions_valid += tree_weight * tree.predict_binned(X_val_binned)
                trees.append(tree)

            train_corr = _nan_safe_corr(y, predictions)
            valid_corr = float("nan")
            if y_val_array is not None and predictions_valid is not None:
                valid_corr = _nan_safe_corr(y_val_array, predictions_valid)

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
                if y_val_array is not None:
                    info["validation_targets"] = y_val_array
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
        """Predict targets for ``X`` using the fitted model."""
        model = self.model
        X_array = np.asarray(X)
        if model.config.prebinned:
            X_binned = ensure_prebinned(X_array, model.config.max_bins)
        else:
            X_binned = apply_binning(X_array.astype(np.float32, copy=False), model.bin_edges)
        return model.predict_binned(X_binned)

    # ------------------------------------------------------------------
    # Internal helpers

    def _sample_features(self, rng: np.random.Generator, n_features: int) -> np.ndarray:
        """Sample a layer-wise feature subset shared across the pack."""
        n_candidates = max(1, int(np.ceil(self.config.layer_feature_fraction * n_features)))
        return np.sort(rng.choice(n_features, size=n_candidates, replace=False))

    def _initialise_pack(
        self, n_samples: int
    ) -> tuple[List[Tree], List[List[int]], List[Dict[int, np.ndarray]]]:
        """Initialise pack structures for a new boosting round."""
        root_indices = np.arange(n_samples, dtype=np.int32)
        trees: List[Tree] = []
        frontiers: List[List[int]] = []
        sample_maps: List[Dict[int, np.ndarray]] = []

        for _ in range(self.config.pack_size):
            tree = Tree()
            root_id = tree.add_node(TreeNode(is_leaf=False, prediction=0.0, depth=0))
            trees.append(tree)
            frontiers.append([root_id])
            sample_maps.append({root_id: root_indices})

        return trees, frontiers, sample_maps

    def _grow_depth(
        self,
        *,
        trees: List[Tree],
        frontiers: List[List[int]],
        samples: List[Dict[int, np.ndarray]],
        feature_subset: Iterable[int],
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        X_binned: np.ndarray,
        depth: int,
    ) -> List[List[int]]:
        """Process one depth across the pack and return the next frontier."""
        next_frontiers: List[List[int]] = [[] for _ in range(len(trees))]

        for tree_idx, tree in enumerate(trees):
            node_map = samples[tree_idx]
            frontier_nodes = frontiers[tree_idx]
            if not frontier_nodes:
                continue

            node_ids: list[int] = []
            node_samples: list[np.ndarray] = []
            for node_id in frontier_nodes:
                indices = node_map.get(node_id)
                if indices is None or indices.size == 0:
                    tree.ensure_leaf(node_id, 0.0)
                    node_map.pop(node_id, None)
                    continue
                node_ids.append(node_id)
                node_samples.append(indices)

            if not node_samples:
                continue

            batch = self._prepare_frontier_batch(node_samples, era_ids)
            decisions = self._evaluate_frontier_batch(
                X_binned=X_binned,
                gradients=gradients,
                hessians=hessians,
                era_ids=era_ids,
                batch=batch,
                features=feature_subset,
            )

            for node_id, decision in zip(node_ids, decisions):
                samples_arr = node_map.pop(node_id, None)
                if samples_arr is None:
                    continue

                if (
                    decision.feature is None
                    or decision.score <= 0
                    or decision.left_indices.size == 0
                    or decision.right_indices.size == 0
                ):
                    tree.ensure_leaf(node_id, decision.left_value)
                    continue

                node = tree.nodes[node_id]
                node.feature = decision.feature
                node.threshold = decision.threshold
                node.is_leaf = False

                left_node = TreeNode(
                    is_leaf=False,
                    prediction=decision.left_value,
                    depth=depth + 1,
                )
                right_node = TreeNode(
                    is_leaf=False,
                    prediction=decision.right_value,
                    depth=depth + 1,
                )
                left_id = tree.add_node(left_node)
                right_id = tree.add_node(right_node)
                node.left = left_id
                node.right = right_id

                samples[tree_idx][left_id] = decision.left_indices
                samples[tree_idx][right_id] = decision.right_indices

                next_frontiers[tree_idx].extend([left_id, right_id])

        return next_frontiers

    def _finalise_trees(self, trees: List[Tree]) -> None:
        """Mark dangling nodes as leaves to ensure consistency."""
        for tree in trees:
            for node_id, node in enumerate(tree.nodes):
                if node.is_leaf or (node.left is None and node.right is None):
                    tree.ensure_leaf(node_id, node.prediction)

    @staticmethod
    def _prepare_frontier_batch(
        node_samples: List[np.ndarray],
        era_ids: np.ndarray,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        node_offsets = np.zeros(len(node_samples) + 1, dtype=np.int32)
        node_era_offsets = np.zeros(len(node_samples) + 1, dtype=np.int32)

        concatenated_parts: list[np.ndarray] = []
        era_group_eras: list[int] = []
        era_group_offsets: list[int] = [0]

        running_index = 0
        running_era_group = 0

        for node_idx, samples in enumerate(node_samples):
            node_offsets[node_idx + 1] = running_index
            node_era_offsets[node_idx + 1] = running_era_group

            if samples.size == 0:
                continue

            sample_era_ids = era_ids[samples]
            order = np.argsort(sample_era_ids, kind="stable")
            sorted_samples = samples[order]
            sorted_eras = sample_era_ids[order]

            concatenated_parts.append(sorted_samples.astype(np.int32, copy=False))
            running_index += sorted_samples.size
            node_offsets[node_idx + 1] = running_index

            unique_eras, counts = np.unique(sorted_eras, return_counts=True)
            if unique_eras.size == 0:
                node_era_offsets[node_idx + 1] = running_era_group
                continue

            era_group_eras.extend(int(e) for e in unique_eras)
            running_era_group += unique_eras.size
            node_era_offsets[node_idx + 1] = running_era_group

            for count in counts:
                era_group_offsets.append(era_group_offsets[-1] + int(count))

        indices = (
            np.concatenate(concatenated_parts)
            if concatenated_parts
            else np.empty(0, dtype=np.int32)
        )
        era_group_eras_arr = np.asarray(era_group_eras, dtype=np.int32)
        era_group_offsets_arr = np.asarray(era_group_offsets, dtype=np.int32)

        return {
            "node_offsets": node_offsets,
            "node_era_offsets": node_era_offsets,
            "era_group_eras": era_group_eras_arr,
            "era_group_offsets": era_group_offsets_arr,
            "indices": indices,
            "samples": node_samples,
        }

    def _evaluate_frontier_batch(
        self,
        *,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        batch: dict[str, np.ndarray | list[np.ndarray]],
        features: Iterable[int],
    ) -> List[SplitDecision]:
        node_samples = batch["samples"]  # type: ignore[index]
        indices = batch["indices"]  # type: ignore[index]
        node_offsets = batch["node_offsets"]  # type: ignore[index]
        node_era_offsets = batch["node_era_offsets"]  # type: ignore[index]
        era_group_eras = batch["era_group_eras"]  # type: ignore[index]
        era_group_offsets = batch["era_group_offsets"]  # type: ignore[index]

        features_arr = np.asarray(list(features), dtype=np.int32)
        if features_arr.size == 0:
            return evaluate_frontier(
                X_binned=X_binned,
                gradients=gradients,
                hessians=hessians,
                era_ids=era_ids,
                node_indices_list=node_samples,
                features=features,
                config=self.config,
            )

        if era_ids.size:
            n_total_eras = int(self._num_eras or (int(era_ids.max()) + 1))
        else:
            n_total_eras = int(self._num_eras or 0)

        if self._use_gpu and cuda_frontier_evaluate is not None:
            (
                features_native,
                thresholds,
                scores,
                agreements,
                left_vals,
                right_vals,
                base_vals,
                left_offsets,
                right_offsets,
                left_indices_flat,
                right_indices_flat,
            ) = cuda_frontier_evaluate(
                X_binned,
                indices,
                node_offsets,
                node_era_offsets,
                era_group_eras,
                era_group_offsets,
                features_arr,
                gradients,
                hessians,
                self.config.max_bins,
                n_total_eras,
                self.config.lambda_l2,
                self.config.lambda_dro,
                self.config.min_samples_leaf,
                self.config.direction_weight,
                self.config.era_tile_size,
                self.config.cuda_threads_per_block,
                self.config.cuda_rows_per_thread,
            )
        elif cpu_frontier_evaluate is not None:
            (
                features_native,
                thresholds,
                scores,
                agreements,
                left_vals,
                right_vals,
                base_vals,
                left_offsets,
                right_offsets,
                left_indices_flat,
                right_indices_flat,
            ) = cpu_frontier_evaluate(
                X_binned,
                indices,
                node_offsets,
                node_era_offsets,
                era_group_eras,
                era_group_offsets,
                features_arr,
                gradients,
                hessians,
                self.config.max_bins,
                n_total_eras,
                self.config.lambda_l2,
                self.config.lambda_dro,
                self.config.min_samples_leaf,
                self.config.direction_weight,
                self.config.era_tile_size,
            )
        else:
            return evaluate_frontier(
                X_binned=X_binned,
                gradients=gradients,
                hessians=hessians,
                era_ids=era_ids,
                node_indices_list=node_samples,
                features=features,
                config=self.config,
            )

        features_arr = features_arr.astype(np.int32, copy=False)

        decisions: List[SplitDecision] = []
        for idx, samples_arr in enumerate(node_samples):
            best_feature = int(features_native[idx])
            threshold = int(thresholds[idx])
            score = float(scores[idx])
            agreement = float(agreements[idx])
            left_value = float(left_vals[idx])
            right_value = float(right_vals[idx])
            base_value = float(base_vals[idx])

            left_start = int(left_offsets[idx])
            left_end = int(left_offsets[idx + 1])
            right_start = int(right_offsets[idx])
            right_end = int(right_offsets[idx + 1])

            left_indices = left_indices_flat[left_start:left_end].astype(np.int32, copy=False)
            right_indices = right_indices_flat[right_start:right_end].astype(np.int32, copy=False)

            if (
                best_feature < 0
                or score <= 0
                or left_indices.size == 0
                or right_indices.size == 0
            ):
                decisions.append(
                    SplitDecision(
                        feature=None,
                        threshold=None,
                        score=score,
                        direction_agreement=0.0,
                        left_value=base_value,
                        right_value=base_value,
                        left_indices=samples_arr,
                        right_indices=np.empty(0, dtype=np.int32),
                    )
                )
                continue

            if (
                left_indices.size < self.config.min_samples_leaf
                or right_indices.size < self.config.min_samples_leaf
            ):
                decisions.append(
                    SplitDecision(
                        feature=None,
                        threshold=None,
                        score=score,
                        direction_agreement=0.0,
                        left_value=base_value,
                        right_value=base_value,
                        left_indices=samples_arr,
                        right_indices=np.empty(0, dtype=np.int32),
                    )
                )
            else:
                decisions.append(
                    SplitDecision(
                        feature=best_feature,
                        threshold=threshold,
                        score=score,
                        direction_agreement=agreement,
                        left_value=left_value,
                        right_value=right_value,
                        left_indices=left_indices.astype(np.int32, copy=False),
                        right_indices=right_indices.astype(np.int32, copy=False),
                    )
                )

        return decisions

    def _evaluate_node(
        self,
        *,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        node_indices: np.ndarray,
        features: Iterable[int],
    ) -> SplitDecision:
        if self._use_gpu:
            return self._evaluate_node_gpu(
                X_binned=X_binned,
                gradients=gradients,
                hessians=hessians,
                era_ids=era_ids,
                node_indices=node_indices,
                features=features,
            )
        return evaluate_node_split(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            node_indices=node_indices,
            features=features,
            config=self.config,
        )

    def _evaluate_node_gpu(
        self,
        *,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        era_ids: np.ndarray,
        node_indices: np.ndarray,
        features: Iterable[int],
    ) -> SplitDecision:
        if cuda_histogram is None:
            raise RuntimeError("CUDA backend is not available")

        features_arr = np.array(list(features), dtype=np.int32)
        node_bins = X_binned[np.ix_(node_indices, features_arr)].astype(np.uint8, copy=False)
        gradients_node = gradients[node_indices]
        hessians_node = hessians[node_indices]
        unique_eras, era_inverse = np.unique(era_ids[node_indices], return_inverse=True)
        n_eras = unique_eras.size
        if n_eras == 0:
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

        era_inverse = era_inverse.astype(np.int16, copy=False)
        hist_grad, hist_hess, hist_count = cuda_histogram(
            node_bins,
            gradients_node,
            hessians_node,
            era_inverse,
            self.config.max_bins,
            n_eras,
        )

        return evaluate_node_split_from_hist(
            features=features_arr,
            node_indices=node_indices,
            node_bins=node_bins,
            gradients=gradients,
            hessians=hessians,
            config=self.config,
            hist_grad=hist_grad,
            hist_hess=hist_hess,
            hist_count=hist_count,
        )

    def _should_use_gpu(self) -> bool:
        return self.config.device == "cuda" and cuda_frontier_evaluate is not None and cuda_available()
