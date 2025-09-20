"""Pure-Python PackBoost baseline backed by vectorised torch ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .config import PackBoostConfig
from .data import BinningResult, apply_bins, build_era_index, preprocess_features


@dataclass(slots=True)
class TreeNode:
    """Single node inside a PackBoost tree."""

    feature: int = -1
    threshold: int = 0
    left: int = -1
    right: int = -1
    value: float = 0.0
    is_leaf: bool = True


@dataclass(slots=True)
class Tree:
    """A complete binary tree represented by an array of :class:`TreeNode`."""

    nodes: List[TreeNode]

    def predict_bins(self, bins: torch.Tensor) -> torch.Tensor:
        """Apply the tree to binned features, returning per-row predictions."""

        num_rows = bins.shape[0]
        device = bins.device
        outputs = torch.zeros(num_rows, dtype=torch.float32, device=device)
        stack: list[tuple[int, torch.Tensor]] = [(0, torch.arange(num_rows, device=device, dtype=torch.int64))]
        while stack:
            node_id, row_idx = stack.pop()
            if row_idx.numel() == 0:
                continue
            node = self.nodes[node_id]
            if node.is_leaf:
                outputs[row_idx] = node.value
                continue
            feature_values = bins[row_idx, node.feature]
            left_mask = feature_values <= node.threshold
            if left_mask.any():
                stack.append((node.left, row_idx[left_mask]))
            if (~left_mask).any():
                stack.append((node.right, row_idx[~left_mask]))
        return outputs


class TreeBuilder:
    """Utility constructing :class:`Tree` objects incrementally."""

    def __init__(self) -> None:
        self.nodes: List[TreeNode] = [TreeNode()]

    def set_leaf(self, node_id: int, value: float) -> None:
        node = self.nodes[node_id]
        node.value = float(value)
        node.is_leaf = True
        node.feature = -1
        node.threshold = 0
        node.left = -1
        node.right = -1

    def split(self, node_id: int, feature: int, threshold: int) -> tuple[int, int]:
        node = self.nodes[node_id]
        node.feature = int(feature)
        node.threshold = int(threshold)
        node.is_leaf = False
        left_id = len(self.nodes)
        right_id = left_id + 1
        node.left = left_id
        node.right = right_id
        self.nodes.append(TreeNode())
        self.nodes.append(TreeNode())
        return left_id, right_id

    def build(self) -> Tree:
        return Tree(nodes=self.nodes)


@dataclass(slots=True)
class NodeShard:
    """Frontier node view storing per-era row indices."""

    tree_id: int
    node_id: int
    depth: int
    era_rows: list[torch.Tensor]


@dataclass(slots=True)
class SplitDecision:
    """Best split metadata for a node."""

    feature: int
    threshold: int
    score: float
    left_grad: float
    left_hess: float
    right_grad: float
    right_hess: float
    left_count: int
    right_count: int
    left_rows: list[torch.Tensor]
    right_rows: list[torch.Tensor]


class PackBoost:
    """Pack-parallel gradient booster baseline."""

    def __init__(self, config: PackBoostConfig) -> None:
        self.config = config
        self._device = torch.device(config.device)
        self._rng = torch.Generator(device="cpu")
        if config.random_state is not None:
            self._rng.manual_seed(int(config.random_state))
        self._binner: BinningResult | None = None
        self._trees: list[Tree] = []
        self._era_unique: np.ndarray | None = None
        self._feature_names: list[str] | None = None

    @property
    def trees(self) -> Sequence[Tree]:
        return self._trees

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        era: Iterable[int],
        *,
        num_rounds: int,
        feature_names: Sequence[str] | None = None,
    ) -> "PackBoost":
        """Fit the model on prebinned or raw features using squared loss."""

        X_np = np.asarray(X)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 1:
            raise ValueError("y must be a 1-D array")
        num_rows, num_features = X_np.shape
        if num_rows != y_np.shape[0]:
            raise ValueError("X and y must share the same number of rows")
        era_np = np.asarray(list(era))
        if era_np.shape[0] != num_rows:
            raise ValueError("era must align with X rows")
        unique_eras, era_inverse = np.unique(era_np, return_inverse=True)
        self._era_unique = unique_eras.astype(np.int16)
        era_encoded = era_inverse.astype(np.int64)
        num_eras = int(unique_eras.shape[0])
        if feature_names is not None:
            if len(feature_names) != num_features:
                raise ValueError("feature_names length mismatch")
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"f{i}" for i in range(num_features)]

        self._binner = preprocess_features(X_np, self.config.max_bins)
        bins = torch.from_numpy(self._binner.bins).to(device=self._device)
        y_t = torch.from_numpy(y_np).to(device=self._device)
        era_t = torch.from_numpy(era_encoded).to(device=self._device)
        grad = torch.zeros(num_rows, dtype=torch.float32, device=self._device)
        hess = torch.ones(num_rows, dtype=torch.float32, device=self._device)
        predictions = torch.zeros(num_rows, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            era_index = build_era_index(era_encoded, num_eras)
            base_era_rows = [rows.to(device=self._device) for rows in era_index]
            self._trees = []

            for round_idx in range(num_rounds):
                grad.copy_(predictions - y_t)
                hess.fill_(1.0)
                pack_trees = [TreeBuilder() for _ in range(self.config.pack_size)]
                frontier: list[NodeShard] = []
                for tree_id in range(self.config.pack_size):
                    shard_rows = [rows.clone() for rows in base_era_rows]
                    frontier.append(
                        NodeShard(tree_id=tree_id, node_id=0, depth=0, era_rows=shard_rows)
                    )
                    root_rows = torch.cat([rows for rows in shard_rows if rows.numel() > 0], dim=0)
                    if root_rows.numel() == 0:
                        pack_trees[tree_id].set_leaf(0, 0.0)
                    else:
                        grad_sum = float(grad[root_rows].sum().item())
                        hess_sum = float(hess[root_rows].sum().item())
                        leaf_value = -grad_sum / (hess_sum + self.config.lambda_l2)
                        pack_trees[tree_id].set_leaf(0, leaf_value)

                for depth in range(self.config.max_depth):
                    active_nodes = [node for node in frontier if self._node_has_capacity(node, grad)]
                    if not active_nodes:
                        break
                    feat_subset = self._sample_features(num_features)
                    next_frontier: list[NodeShard] = []
                    for node in active_nodes:
                        decision = self._find_best_split(node, grad, hess, bins, feat_subset)
                        tree_builder = pack_trees[node.tree_id]
                        if decision is None:
                            rows = torch.cat([r for r in node.era_rows if r.numel() > 0], dim=0)
                            if rows.numel() == 0:
                                tree_builder.set_leaf(node.node_id, 0.0)
                            else:
                                grad_sum = float(grad[rows].sum().item())
                                hess_sum = float(hess[rows].sum().item())
                                value = -grad_sum / (hess_sum + self.config.lambda_l2)
                                tree_builder.set_leaf(node.node_id, value)
                            continue
                        left_id, right_id = tree_builder.split(
                            node.node_id, decision.feature, decision.threshold
                        )
                        left_value = -decision.left_grad / (decision.left_hess + self.config.lambda_l2)
                        right_value = -decision.right_grad / (decision.right_hess + self.config.lambda_l2)
                        tree_builder.set_leaf(left_id, left_value)
                        tree_builder.set_leaf(right_id, right_value)
                        next_frontier.append(
                            NodeShard(
                                tree_id=node.tree_id,
                                node_id=left_id,
                                depth=node.depth + 1,
                                era_rows=[rows.clone() for rows in decision.left_rows],
                            )
                        )
                        next_frontier.append(
                            NodeShard(
                                tree_id=node.tree_id,
                                node_id=right_id,
                                depth=node.depth + 1,
                                era_rows=[rows.clone() for rows in decision.right_rows],
                            )
                        )
                    frontier = next_frontier

                pack_predictions = torch.zeros_like(predictions)
                for tree_builder in pack_trees:
                    tree = tree_builder.build()
                    self._trees.append(tree)
                    pack_predictions += tree.predict_bins(bins)
                predictions += self.config.learning_rate * pack_predictions

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted trees."""

        if self._binner is None:
            raise RuntimeError("PackBoost model must be fitted before predicting")
        bins_np = apply_bins(X, self._binner.bin_edges, self.config.max_bins)
        with torch.no_grad():
            bins = torch.from_numpy(bins_np).to(device=self._device)
            if not self._trees:
                return np.zeros(bins.shape[0], dtype=np.float32)
            predictions = torch.zeros(bins.shape[0], dtype=torch.float32, device=self._device)
            for tree in self._trees:
                predictions += self.config.learning_rate * tree.predict_bins(bins)
        return predictions.cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers

    def _sample_features(self, num_features: int) -> torch.Tensor:
        frac = float(self.config.layer_feature_fraction)
        k = max(1, int(round(frac * num_features)))
        if k > num_features:
            k = num_features
        perm = torch.randperm(num_features, generator=self._rng)
        return perm[:k]

    def _node_has_capacity(self, node: NodeShard, grad: torch.Tensor) -> bool:
        rows = torch.cat([r for r in node.era_rows if r.numel() > 0], dim=0)
        return rows.numel() >= 2 * self.config.min_samples_leaf

    def _find_best_split(
        self,
        node: NodeShard,
        grad: torch.Tensor,
        hess: torch.Tensor,
        bins: torch.Tensor,
        feature_subset: torch.Tensor,
    ) -> SplitDecision | None:
        num_bins = self.config.max_bins
        num_thresholds = num_bins - 1
        if num_thresholds <= 0:
            return None
        era_rows = node.era_rows
        all_rows = torch.cat([rows for rows in era_rows if rows.numel() > 0], dim=0)
        if all_rows.numel() < 2 * self.config.min_samples_leaf:
            return None
        total_grad = grad[all_rows].sum()
        total_hess = hess[all_rows].sum()
        total_count = all_rows.numel()
        best_decision: SplitDecision | None = None
        lambda_l2 = self.config.lambda_l2
        lambda_dro = self.config.lambda_dro
        direction_weight = self.config.direction_weight
        for feature in feature_subset.tolist():
            agg_grad_hist = torch.zeros(num_bins, dtype=torch.float32, device=self._device)
            agg_hess_hist = torch.zeros(num_bins, dtype=torch.float32, device=self._device)
            agg_count_hist = torch.zeros(num_bins, dtype=torch.float32, device=self._device)
            n = torch.zeros(num_thresholds, dtype=torch.float32, device=self._device)
            mean = torch.zeros_like(n)
            m2 = torch.zeros_like(n)
            agree = torch.zeros_like(n)
            for rows in era_rows:
                if rows.numel() == 0:
                    continue
                feature_values = bins[rows, feature]
                grad_rows = grad[rows]
                hess_rows = hess[rows]
                counts = torch.bincount(feature_values, minlength=num_bins).to(torch.float32)
                grad_hist = torch.bincount(feature_values, weights=grad_rows, minlength=num_bins)
                hess_hist = torch.bincount(feature_values, weights=hess_rows, minlength=num_bins)
                agg_count_hist += counts
                agg_grad_hist += grad_hist
                agg_hess_hist += hess_hist
                if counts.sum() == 0:
                    continue
                prefix_counts = torch.cumsum(counts, dim=0)[:-1]
                prefix_grad = torch.cumsum(grad_hist, dim=0)[:-1]
                prefix_hess = torch.cumsum(hess_hist, dim=0)[:-1]
                total_grad_e = grad_hist.sum()
                total_hess_e = hess_hist.sum()
                left_count = prefix_counts
                right_count = counts.sum() - prefix_counts
                valid = (left_count > 0) & (right_count > 0)
                if not torch.any(valid):
                    continue
                left_grad = prefix_grad
                right_grad = total_grad_e - left_grad
                left_hess = prefix_hess
                right_hess = total_hess_e - left_hess
                parent_gain = 0.5 * (total_grad_e ** 2) / (total_hess_e + lambda_l2)
                gain_left = 0.5 * (left_grad ** 2) / (left_hess + lambda_l2)
                gain_right = 0.5 * (right_grad ** 2) / (right_hess + lambda_l2)
                era_gain = gain_left + gain_right - parent_gain
                parent_value = -total_grad_e / (total_hess_e + lambda_l2)
                left_value = -left_grad / (left_hess + lambda_l2)
                right_value = -right_grad / (right_hess + lambda_l2)
                parent_dir = torch.sign(parent_value)
                agree_left = (torch.sign(left_value) == parent_dir).to(torch.float32)
                agree_right = (torch.sign(right_value) == parent_dir).to(torch.float32)
                era_agreement = 0.5 * (agree_left + agree_right)
                valid_f = valid.to(torch.float32)
                delta = era_gain - mean
                new_n = n + valid_f
                inv_new_n = torch.where(new_n > 0, 1.0 / new_n, torch.zeros_like(new_n))
                updated_mean = mean + delta * inv_new_n
                updated_m2 = m2 + (era_gain - updated_mean) * delta
                mean = torch.where(valid, updated_mean, mean)
                m2 = torch.where(valid, updated_m2, m2)
                n = new_n
                agree = agree + era_agreement * valid_f
            if agg_count_hist.sum() < 2 * self.config.min_samples_leaf:
                continue
            prefix_counts = torch.cumsum(agg_count_hist, dim=0)[:-1]
            prefix_grad = torch.cumsum(agg_grad_hist, dim=0)[:-1]
            prefix_hess = torch.cumsum(agg_hess_hist, dim=0)[:-1]
            left_count = prefix_counts
            right_count = total_count - prefix_counts
            valid_global = (left_count >= self.config.min_samples_leaf) & (
                right_count >= self.config.min_samples_leaf
            )
            if not torch.any(valid_global):
                continue
            variance = torch.where(n > 1.0, m2 / (n - 1.0), torch.zeros_like(m2))
            std = torch.sqrt(torch.clamp(variance, min=0.0))
            mean_score = mean
            agreement_mean = torch.where(n > 0, agree / torch.clamp_min(n, 1.0), torch.zeros_like(agree))
            score = mean_score - lambda_dro * std + direction_weight * agreement_mean
            score = torch.where(valid_global, score, torch.full_like(score, float("-inf")))
            best_score, best_idx = torch.max(score, dim=0)
            if not torch.isfinite(best_score):
                continue
            left_grad = prefix_grad[best_idx]
            left_hess = prefix_hess[best_idx]
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            chosen_threshold = int(best_idx.item())
            left_rows: list[torch.Tensor] = []
            right_rows: list[torch.Tensor] = []
            for rows in era_rows:
                if rows.numel() == 0:
                    left_rows.append(torch.empty(0, dtype=torch.int64, device=self._device))
                    right_rows.append(torch.empty(0, dtype=torch.int64, device=self._device))
                    continue
                feature_values = bins[rows, feature]
                mask = feature_values <= chosen_threshold
                left_rows.append(rows[mask])
                right_rows.append(rows[~mask])
            decision = SplitDecision(
                feature=int(feature),
                threshold=chosen_threshold,
                score=float(best_score.item()),
                left_grad=float(left_grad.item()),
                left_hess=float(left_hess.item()),
                right_grad=float(right_grad.item()),
                right_hess=float(right_hess.item()),
                left_count=int(prefix_counts[best_idx].item()),
                right_count=int(right_count[best_idx].item()),
                left_rows=left_rows,
                right_rows=right_rows,
            )
            if best_decision is None or decision.score > best_decision.score:
                best_decision = decision
        return best_decision
