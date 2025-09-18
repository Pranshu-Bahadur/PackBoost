"""CPU reference implementation of PackBoost with DES scoring."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from .config import PackBoostConfig
from .des import SplitDecision, evaluate_node_split, evaluate_node_split_from_hist
from .model import PackBoostModel, Tree, TreeNode
from .utils.binning import apply_binning, quantile_binning
from .backends import cuda_available, cuda_histogram


class PackBoost:
    """Gradient boosting with directional era splitting on the CPU."""

    def __init__(self, config: PackBoostConfig) -> None:
        self.config = config
        self._model: PackBoostModel | None = None
        self._use_gpu: bool = False

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
        era_ids: Sequence[int],
        *,
        num_rounds: int = 10,
    ) -> "PackBoost":
        """Fit PackBoost to the provided data."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        era_ids = np.asarray(era_ids, dtype=np.int16)

        if X.shape[0] != y.shape[0] or X.shape[0] != era_ids.shape[0]:
            raise ValueError("X, y, and era_ids must have the same number of rows")

        n_samples, n_features = X.shape
        self.config.validate(n_features)

        if self.config.device == "cuda" and not self._should_use_gpu():
            raise RuntimeError(
                "CUDA device requested but the native CUDA backend is not available."
            )
        self._use_gpu = self._should_use_gpu()

        X_binned, bin_edges = quantile_binning(
            X, self.config.max_bins, random_state=self.config.random_state
        )

        initial_prediction = float(np.mean(y))
        predictions = np.full(n_samples, initial_prediction, dtype=np.float32)
        hessians = np.ones(n_samples, dtype=np.float32)

        trees: List[Tree] = []
        rng = np.random.default_rng(self.config.random_state)
        tree_weight = self.config.learning_rate / self.config.pack_size

        for _ in range(num_rounds):
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
                predictions += tree_weight * tree.predict_binned(X_binned)
                trees.append(tree)

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
        X = np.asarray(X, dtype=np.float32)
        X_binned = apply_binning(X, model.bin_edges)
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
            for node_id in frontiers[tree_idx]:
                node_indices = node_map.pop(node_id, None)
                if node_indices is None or node_indices.size == 0:
                    continue

                decision = self._evaluate_node(
                    X_binned=X_binned,
                    gradients=gradients,
                    hessians=hessians,
                    era_ids=era_ids,
                    node_indices=node_indices,
                    features=feature_subset,
                )

                if decision.feature is None or decision.score <= 0:
                    tree.ensure_leaf(node_id, decision.left_value)
                    continue

                node = tree.nodes[node_id]
                node.feature = decision.feature
                node.threshold = decision.threshold
                node.is_leaf = False

                left_id = tree.add_node(
                    TreeNode(is_leaf=False, prediction=decision.left_value, depth=depth + 1)
                )
                right_id = tree.add_node(
                    TreeNode(is_leaf=False, prediction=decision.right_value, depth=depth + 1)
                )
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
        return self.config.device == "cuda" and cuda_available()
