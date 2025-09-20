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
                self._update_gradients(predictions, y_t, grad)
                self._update_hessians(hess)
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
                    active_nodes = [node for node in frontier if self._node_has_capacity(node)]
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

    def _node_has_capacity(self, node: NodeShard) -> bool:
        total = sum(int(rows.numel()) for rows in node.era_rows)
        return total >= 2 * self.config.min_samples_leaf

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
        all_rows, era_ids = self._stack_node_rows(era_rows)
        if all_rows.numel() < 2 * self.config.min_samples_leaf:
            return None

        grad_all = grad[all_rows]
        hess_all = hess[all_rows]
        total_grad = grad_all.sum()
        total_hess = hess_all.sum()
        total_count = all_rows.numel()

        lambda_l2 = self.config.lambda_l2
        lambda_dro = self.config.lambda_dro
        direction_weight = self.config.direction_weight
        num_eras = len(era_rows)

        best_decision: SplitDecision | None = None
        thresholds = torch.arange(num_bins - 1, device=self._device, dtype=torch.int64)

        for feature in feature_subset.tolist():
            feature_values = bins[all_rows, feature].to(dtype=torch.int64)
            counts, grad_hist, hess_hist = self._compute_histograms(
                feature_values, grad_all, hess_all, era_ids, num_eras, num_bins
            )

            if counts.sum() < 2 * self.config.min_samples_leaf:
                continue

            # Per-era prefix stats for Welford aggregates
            prefix_counts = counts[:, :-1].cumsum(dim=1)
            prefix_grad = grad_hist[:, :-1].cumsum(dim=1)
            prefix_hess = hess_hist[:, :-1].cumsum(dim=1)

            total_count_e = counts.sum(dim=1, keepdim=True)
            total_grad_e = grad_hist.sum(dim=1, keepdim=True)
            total_hess_e = hess_hist.sum(dim=1, keepdim=True)

            left_count = prefix_counts
            right_count = total_count_e - left_count
            valid = (left_count > 0) & (right_count > 0)
            if not torch.any(valid):
                continue

            left_grad = prefix_grad
            right_grad = total_grad_e - left_grad
            left_hess = prefix_hess
            right_hess = total_hess_e - left_hess

            parent_gain = 0.5 * (total_grad_e**2) / (total_hess_e + lambda_l2)
            gain_left = 0.5 * (left_grad**2) / (left_hess + lambda_l2)
            gain_right = 0.5 * (right_grad**2) / (right_hess + lambda_l2)
            era_gain = gain_left + gain_right - parent_gain

            parent_value = -total_grad_e / (total_hess_e + lambda_l2)
            left_value = -left_grad / (left_hess + lambda_l2)
            right_value = -right_grad / (right_hess + lambda_l2)
            parent_dir = torch.sign(parent_value)
            agree_left = (torch.sign(left_value) == parent_dir).to(torch.float32)
            agree_right = (torch.sign(right_value) == parent_dir).to(torch.float32)
            era_agreement = 0.5 * (agree_left + agree_right)

            valid_float = valid.to(torch.float32)
            n = valid_float.sum(dim=0)
            mean_gain = torch.where(
                n > 0,
                (era_gain * valid_float).sum(dim=0) / torch.clamp_min(n, 1.0),
                torch.zeros(
                    num_thresholds, device=self._device, dtype=torch.float32
                ),
            )

            diff = era_gain - mean_gain.unsqueeze(0)
            sum_sq = (diff.pow(2) * valid_float).sum(dim=0)
            variance = torch.where(
                n > 1,
                sum_sq / torch.clamp_min(n - 1.0, 1.0),
                torch.zeros_like(sum_sq),
            )
            std = torch.sqrt(torch.clamp_min(variance, 0.0))

            agreement_mean = torch.where(
                n > 0,
                (era_agreement * valid_float).sum(dim=0) / torch.clamp_min(n, 1.0),
                torch.zeros_like(mean_gain),
            )

            agg_count = counts.sum(dim=0)
            agg_grad = grad_hist.sum(dim=0)
            agg_hess = hess_hist.sum(dim=0)

            prefix_count_total = agg_count.cumsum(dim=0)[:-1]
            prefix_grad_total = agg_grad.cumsum(dim=0)[:-1]
            prefix_hess_total = agg_hess.cumsum(dim=0)[:-1]

            left_count_total = prefix_count_total
            right_count_total = total_count - left_count_total
            valid_global = (left_count_total >= self.config.min_samples_leaf) & (
                right_count_total >= self.config.min_samples_leaf
            )
            if not torch.any(valid_global):
                continue

            score = mean_gain - lambda_dro * std + direction_weight * agreement_mean
            score = torch.where(
                valid_global,
                score,
                torch.full_like(score, float("-inf")),
            )

            best_score, best_idx = torch.max(score, dim=0)
            if not torch.isfinite(best_score):
                continue

            chosen_idx = int(best_idx.item())
            chosen_threshold = int(thresholds[chosen_idx].item())

            left_grad_total = prefix_grad_total[chosen_idx]
            left_hess_total = prefix_hess_total[chosen_idx]
            right_grad_total = total_grad - left_grad_total
            right_hess_total = total_hess - left_hess_total

            left_rows, right_rows = self._partition_rows(
                bins, era_rows, feature, chosen_threshold
            )

            decision = SplitDecision(
                feature=int(feature),
                threshold=chosen_threshold,
                score=float(best_score.item()),
                left_grad=float(left_grad_total.item()),
                left_hess=float(left_hess_total.item()),
                right_grad=float(right_grad_total.item()),
                right_hess=float(right_hess_total.item()),
                left_count=int(left_count_total[chosen_idx].item()),
                right_count=int(right_count_total[chosen_idx].item()),
                left_rows=left_rows,
                right_rows=right_rows,
            )

            if best_decision is None or decision.score > best_decision.score:
                best_decision = decision

        return best_decision

    def _update_gradients(
        self, predictions: torch.Tensor, targets: torch.Tensor, out: torch.Tensor
    ) -> None:
        out.copy_(predictions - targets)

    def _update_hessians(self, out: torch.Tensor) -> None:
        out.fill_(1.0)

    def _stack_node_rows(
        self, era_rows: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_chunks: list[torch.Tensor] = []
        era_chunks: list[torch.Tensor] = []
        for era_idx, rows in enumerate(era_rows):
            if rows.numel() == 0:
                continue
            row_chunks.append(rows)
            era_chunks.append(
                torch.full((rows.numel(),), era_idx, dtype=torch.int64, device=rows.device)
            )
        if not row_chunks:
            empty = torch.empty(0, dtype=torch.int64, device=self._device)
            return empty, empty.clone()
        return torch.cat(row_chunks, dim=0), torch.cat(era_chunks, dim=0)

    def _compute_histograms(
        self,
        feature_values: torch.Tensor,
        grad_rows: torch.Tensor,
        hess_rows: torch.Tensor,
        era_ids: torch.Tensor,
        num_eras: int,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_index = era_ids * num_bins + feature_values
        hist_size = num_eras * num_bins
        counts = torch.bincount(flat_index, minlength=hist_size).reshape(num_eras, num_bins)
        grad_hist = torch.bincount(
            flat_index, weights=grad_rows, minlength=hist_size
        ).reshape(num_eras, num_bins)
        hess_hist = torch.bincount(
            flat_index, weights=hess_rows, minlength=hist_size
        ).reshape(num_eras, num_bins)
        return counts.to(torch.float32), grad_hist, hess_hist

    def _partition_rows(
        self,
        bins: torch.Tensor,
        era_rows: Sequence[torch.Tensor],
        feature: int,
        threshold: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feature_column = bins[:, feature]
        left_rows: list[torch.Tensor] = []
        right_rows: list[torch.Tensor] = []
        for rows in era_rows:
            if rows.numel() == 0:
                empty = rows.new_empty(0)
                left_rows.append(empty)
                right_rows.append(rows.new_empty(0))
                continue
            feature_values = feature_column.index_select(0, rows)
            mask = feature_values <= threshold
            left_rows.append(rows[mask])
            right_rows.append(rows[~mask])
        return left_rows, right_rows
