"""Pure-Python PackBoost baseline backed by vectorised torch ops."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import perf_counter
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


@dataclass(slots=True)
class DepthInstrumentation:
    """Per-depth counters emitted as JSON logs."""

    nodes_processed: int = 0
    nodes_skipped: int = 0
    rows_total: int = 0
    feature_blocks: int = 0
    bincount_calls: int = 0
    hist_ms: float = 0.0
    scan_ms: float = 0.0
    score_ms: float = 0.0
    partition_ms: float = 0.0
    nodes_subtract_ok: int = 0
    nodes_subtract_fallback: int = 0
    nodes_rebuild: int = 0
    block_size: int = 0

    def __iadd__(self, other: "DepthInstrumentation") -> "DepthInstrumentation":
        self.nodes_processed += other.nodes_processed
        self.nodes_skipped += other.nodes_skipped
        self.rows_total += other.rows_total
        self.feature_blocks += other.feature_blocks
        self.bincount_calls += other.bincount_calls
        self.hist_ms += other.hist_ms
        self.scan_ms += other.scan_ms
        self.score_ms += other.score_ms
        self.partition_ms += other.partition_ms
        self.nodes_subtract_ok += other.nodes_subtract_ok
        self.nodes_subtract_fallback += other.nodes_subtract_fallback
        self.nodes_rebuild += other.nodes_rebuild
        self.block_size = max(self.block_size, other.block_size)
        return self

    def to_dict(self) -> dict[str, int | float]:
        return {
            "nodes_processed": self.nodes_processed,
            "nodes_skipped": self.nodes_skipped,
            "rows_total": self.rows_total,
            "feature_blocks": self.feature_blocks,
            "bincount_calls": self.bincount_calls,
            "hist_ms": self.hist_ms,
            "scan_ms": self.scan_ms,
            "score_ms": self.score_ms,
            "partition_ms": self.partition_ms,
            "nodes_subtract_ok": self.nodes_subtract_ok,
            "nodes_subtract_fallback": self.nodes_subtract_fallback,
            "nodes_rebuild": self.nodes_rebuild,
            "feature_block_size": self.block_size,
        }


@dataclass(slots=True)
class _NodeContext:
    """Precomputed tensors for a frontier node processed in a batch."""

    shard: NodeShard
    all_rows: torch.Tensor
    era_ids: torch.Tensor
    grad_rows: torch.Tensor
    hess_rows: torch.Tensor
    bins_all: torch.Tensor
    era_weights: torch.Tensor
    parent_dir: torch.Tensor
    total_grad: float
    total_hess: float
    total_count: int


class PackBoost:
    """Pack-parallel gradient booster baseline."""

    def __init__(self, config: PackBoostConfig) -> None:
        self.config = config
        self._device = torch.device(config.device)
        self._rng = torch.Generator(device="cpu")
        if config.random_state is not None:
            self._rng.manual_seed(int(config.random_state))
        env_mode = os.getenv("PACKBOOST_HIST_MODE")
        mode = (env_mode or config.histogram_mode).lower()
        if mode not in {"rebuild", "subtract", "auto"}:
            raise ValueError(f"Unsupported histogram_mode: {mode}")
        self._histogram_mode = mode
        self._histogram_l2_budget = 256 * 1024  # bytes
        self._logger = logging.getLogger(__name__)
        self._depth_logs: list[dict[str, object]] = []
        self._binner: BinningResult | None = None
        self._trees: list[Tree] = []
        self._era_unique: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._tree_weight: float | None = None
        self._trained_pack_size: int | None = None

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
            self._depth_logs = []
            self._tree_weight = None
            self._trained_pack_size = None
            era_index = build_era_index(era_encoded, num_eras)
            base_era_rows = [rows.to(device=self._device) for rows in era_index]
            self._trees = []

            for round_idx in range(num_rounds):
                self._update_gradients(predictions, y_t, grad)
                self._update_hessians(hess)
                pack_trees = [TreeBuilder() for _ in range(self.config.pack_size)]
                frontier: list[NodeShard] = []
                for tree_id in range(self.config.pack_size):
                    shard_rows = list(base_era_rows)
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
                    stats_depth = DepthInstrumentation()
                    decisions: list[SplitDecision | None] = []
                    if self.config.enable_node_batching:
                        decisions, stats_batch = self._find_best_splits_batched(
                            active_nodes, grad, hess, bins, feat_subset
                        )
                        stats_depth += stats_batch
                    else:
                        for node in active_nodes:
                            partial, stats_batch = self._find_best_splits_batched(
                                [node], grad, hess, bins, feat_subset
                            )
                            decisions.extend(partial)
                            stats_depth += stats_batch
                    next_frontier: list[NodeShard] = []
                    for node, decision in zip(active_nodes, decisions):
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
                                era_rows=list(decision.left_rows),
                            )
                        )
                        next_frontier.append(
                            NodeShard(
                                tree_id=node.tree_id,
                                node_id=right_id,
                                depth=node.depth + 1,
                                era_rows=list(decision.right_rows),
                            )
                        )
                    frontier = next_frontier

                    depth_log = stats_depth.to_dict()
                    depth_log.update(
                        {
                            "depth": depth,
                            "feature_subset_size": int(feat_subset.numel()),
                            "pack_size": self.config.pack_size,
                            "layer_feature_fraction": float(self.config.layer_feature_fraction),
                            "histogram_mode": self._histogram_mode,
                            "seed": self.config.random_state,
                            "torch_threads": torch.get_num_threads(),
                        }
                    )
                    if depth_log["feature_block_size"] == 0:
                        depth_log["feature_block_size"] = self._resolve_feature_block_size(
                            int(feat_subset.numel())
                        )
                    self._depth_logs.append(depth_log)
                    if self._logger.isEnabledFor(logging.INFO):
                        self._logger.info(json.dumps(depth_log))

                pack_predictions = torch.zeros_like(predictions)
                for tree_builder in pack_trees:
                    tree = tree_builder.build()
                    self._trees.append(tree)
                    pack_predictions += tree.predict_bins(bins)
                per_tree_weight = float(self.config.learning_rate) / float(
                    self.config.pack_size
                )
                predictions += per_tree_weight * pack_predictions
                self._tree_weight = per_tree_weight
                self._trained_pack_size = int(self.config.pack_size)

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
            if self._tree_weight is None:
                raise RuntimeError("Model missing _tree_weight; call fit() first.")
            if (
                self._trained_pack_size is not None
                and int(self.config.pack_size) != self._trained_pack_size
            ):
                raise RuntimeError(
                    "Config pack_size differs from training; please keep pack_size constant."
                )
            predictions = torch.zeros(bins.shape[0], dtype=torch.float32, device=self._device)
            for tree in self._trees:
                predictions += self._tree_weight * tree.predict_bins(bins)
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

    def _resolve_feature_block_size(self, subset_size: int) -> int:
        block = int(self.config.feature_block_size)
        if block <= 0 or block > subset_size:
            return subset_size
        return block

    def _find_best_splits_batched(
        self,
        nodes: Sequence[NodeShard],
        grad: torch.Tensor,
        hess: torch.Tensor,
        bins: torch.Tensor,
        feature_subset: torch.Tensor,
    ) -> tuple[list[SplitDecision | None], DepthInstrumentation]:
        stats = DepthInstrumentation()
        num_nodes_total = len(nodes)
        decisions: list[SplitDecision | None] = [None] * num_nodes_total
        if num_nodes_total == 0:
            return decisions, stats

        num_bins = self.config.max_bins
        num_thresholds = num_bins - 1
        if num_thresholds <= 0:
            stats.nodes_skipped += num_nodes_total
            return decisions, stats

        feat_ids = feature_subset.to(self._device, dtype=torch.int64)
        if feat_ids.numel() == 0:
            stats.nodes_skipped += num_nodes_total
            return decisions, stats

        contexts: list[_NodeContext] = []
        index_map: list[int] = []
        thresholds = torch.arange(num_thresholds, device=self._device, dtype=torch.int64)

        lambda_l2 = self.config.lambda_l2
        lambda_dro = self.config.lambda_dro
        direction_weight = self.config.direction_weight

        for idx, node in enumerate(nodes):
            all_rows, era_ids = self._stack_node_rows(node.era_rows)
            total_count = int(all_rows.numel())
            if total_count < 2 * self.config.min_samples_leaf:
                stats.nodes_skipped += 1
                continue
            grad_rows = grad.index_select(0, all_rows)
            hess_rows = hess.index_select(0, all_rows)
            total_grad = float(grad_rows.sum().item())
            total_hess = float(hess_rows.sum().item())
            parent_value_total = -total_grad / (total_hess + lambda_l2)
            parent_dir = torch.sign(torch.tensor(parent_value_total, device=self._device))
            era_counts = torch.tensor(
                [float(rows.numel()) for rows in node.era_rows],
                dtype=torch.float32,
                device=self._device,
            )
            era_weights = self._compute_era_weights(era_counts)
            bins_all = bins.index_select(0, all_rows)
            contexts.append(
                _NodeContext(
                    shard=node,
                    all_rows=all_rows,
                    era_ids=era_ids,
                    grad_rows=grad_rows,
                    hess_rows=hess_rows,
                    bins_all=bins_all,
                    era_weights=era_weights,
                    parent_dir=parent_dir,
                    total_grad=total_grad,
                    total_hess=total_hess,
                    total_count=total_count,
                )
            )
            index_map.append(idx)
            stats.rows_total += total_count

        if not contexts:
            return decisions, stats

        stats.nodes_processed += len(contexts)
        num_nodes = len(contexts)
        num_eras = len(contexts[0].shard.era_rows)
        block_size = self._resolve_feature_block_size(int(feat_ids.numel()))
        stats.block_size = max(stats.block_size, block_size)

        era_weights_tensor = torch.stack([ctx.era_weights for ctx in contexts], dim=0)
        total_grad_nodes = torch.tensor(
            [ctx.total_grad for ctx in contexts], dtype=torch.float32, device=self._device
        )
        total_hess_nodes = torch.tensor(
            [ctx.total_hess for ctx in contexts], dtype=torch.float32, device=self._device
        )
        total_count_nodes = torch.tensor(
            [ctx.total_count for ctx in contexts], dtype=torch.float32, device=self._device
        )
        total_count_nodes_int = torch.tensor(
            [ctx.total_count for ctx in contexts], dtype=torch.int64, device=self._device
        )
        parent_dirs = torch.stack([ctx.parent_dir.to(torch.float32) for ctx in contexts], dim=0)

        best_scores = torch.full((num_nodes,), float("-inf"), device=self._device)
        best_features = torch.full((num_nodes,), -1, dtype=torch.int64, device=self._device)
        best_thresholds = torch.zeros(num_nodes, dtype=torch.int64, device=self._device)
        best_left_grad = torch.zeros(num_nodes, dtype=torch.float32, device=self._device)
        best_left_hess = torch.zeros_like(best_left_grad)
        best_left_count = torch.zeros(num_nodes, dtype=torch.float32, device=self._device)

        for start in range(0, feat_ids.numel(), block_size):
            block = feat_ids[start : start + block_size]
            if block.numel() == 0:
                continue

            bins_block_chunks: list[torch.Tensor] = []
            grad_chunks: list[torch.Tensor] = []
            hess_chunks: list[torch.Tensor] = []
            era_chunks: list[torch.Tensor] = []
            node_chunks: list[torch.Tensor] = []
            for node_idx, ctx in enumerate(contexts):
                bins_chunk = ctx.bins_all.index_select(1, block).to(dtype=torch.int64)
                bins_block_chunks.append(bins_chunk)
                grad_chunks.append(ctx.grad_rows)
                hess_chunks.append(ctx.hess_rows)
                era_chunks.append(ctx.era_ids)
                node_chunks.append(
                    torch.full(
                        (ctx.all_rows.numel(),),
                        node_idx,
                        dtype=torch.int64,
                        device=self._device,
                    )
                )

            bins_block = torch.cat(bins_block_chunks, dim=0)
            grad_concat = torch.cat(grad_chunks, dim=0)
            hess_concat = torch.cat(hess_chunks, dim=0)
            era_concat = torch.cat(era_chunks, dim=0)
            node_concat = torch.cat(node_chunks, dim=0)

            t0 = perf_counter()
            counts, grad_hist, hess_hist = self._batched_histograms_nodes(
                bins_block,
                grad_concat,
                hess_concat,
                era_concat,
                node_concat,
                num_nodes,
                num_eras,
                num_bins,
            )
            stats.hist_ms += (perf_counter() - t0) * 1000.0
            stats.feature_blocks += 1
            stats.bincount_calls += 3

            if counts.numel() == 0:
                continue

            scan_start = perf_counter()
            left_count = counts[:, :, :, :-1].cumsum(dim=3)
            left_grad = grad_hist[:, :, :, :-1].cumsum(dim=3)
            left_hess = hess_hist[:, :, :, :-1].cumsum(dim=3)
            total_count_e = counts.sum(dim=3, keepdim=True)
            total_grad_e = grad_hist.sum(dim=3, keepdim=True)
            total_hess_e = hess_hist.sum(dim=3, keepdim=True)

            agg_count = counts.sum(dim=2)
            agg_grad = grad_hist.sum(dim=2)
            agg_hess = hess_hist.sum(dim=2)
            prefix_count_total = agg_count.cumsum(dim=2)[:, :, :-1]
            prefix_grad_total = agg_grad.cumsum(dim=2)[:, :, :-1]
            prefix_hess_total = agg_hess.cumsum(dim=2)[:, :, :-1]
            stats.scan_ms += (perf_counter() - scan_start) * 1000.0

            left_count_total = prefix_count_total
            right_count_total = (
                total_count_nodes.view(1, num_nodes, 1) - left_count_total
            )

            workspace_bytes = 8 * 2 * num_eras * num_bins * block.numel()
            if self._histogram_mode == "subtract":
                subtract_mask = torch.ones(num_nodes, dtype=torch.bool, device=self._device)
            elif self._histogram_mode == "rebuild":
                subtract_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self._device)
            else:
                workspace_ok = workspace_bytes <= self._histogram_l2_budget
                if workspace_ok:
                    ratio = torch.minimum(left_count_total, right_count_total)
                    ratio = ratio / torch.clamp_min(
                        total_count_nodes.view(1, num_nodes, 1), 1.0
                    )
                    ratio_perm = ratio.permute(1, 0, 2).reshape(num_nodes, -1)
                    ratio_ok = ratio_perm.ge(0.1).any(dim=1)
                    subtract_mask = ratio_ok
                else:
                    subtract_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self._device)
            score_start = perf_counter()
            right_count = total_count_e - left_count
            right_grad = total_grad_e - left_grad
            right_hess = total_hess_e - left_hess

            subtract_valid = torch.zeros(num_nodes, dtype=torch.bool, device=self._device)
            if subtract_mask.any():
                subtract_valid = self._validate_histogram_subtraction(
                    left_count,
                    right_count,
                    total_count_e,
                    left_grad,
                    right_grad,
                    total_grad_e,
                    left_hess,
                    right_hess,
                    total_hess_e,
                )

            final_subtract = subtract_mask & subtract_valid
            rebuild_mask = ~final_subtract
            fallback_mask = subtract_mask & ~subtract_valid

            if rebuild_mask.any():
                right_count_rebuild = counts[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                right_grad_rebuild = grad_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                right_hess_rebuild = hess_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                idx = rebuild_mask.nonzero(as_tuple=True)[0]
                right_count[:, idx, :, :] = right_count_rebuild[:, idx, :, :]
                right_grad[:, idx, :, :] = right_grad_rebuild[:, idx, :, :]
                right_hess[:, idx, :, :] = right_hess_rebuild[:, idx, :, :]

            stats.nodes_subtract_ok += int(final_subtract.sum().item())
            stats.nodes_subtract_fallback += int(fallback_mask.sum().item())
            stats.nodes_rebuild += int(rebuild_mask.sum().item())

            valid = (left_count > 0) & (right_count > 0)
            if not torch.any(valid):
                stats.score_ms += (perf_counter() - score_start) * 1000.0
                continue

            parent_gain = 0.5 * (total_grad_e**2) / (total_hess_e + lambda_l2)
            gain_left = 0.5 * (left_grad**2) / (left_hess + lambda_l2)
            gain_right = 0.5 * (right_grad**2) / (right_hess + lambda_l2)
            era_gain = gain_left + gain_right - parent_gain

            weights_eff = (
                era_weights_tensor.view(1, num_nodes, num_eras, 1).to(era_gain.dtype)
                * valid.to(era_gain.dtype)
            )
            weight_sum = weights_eff.sum(dim=2)
            safe_weight_sum = torch.clamp_min(weight_sum, 1e-12)

            mean_gain = torch.where(
                weight_sum > 0,
                (weights_eff * era_gain).sum(dim=2) / safe_weight_sum,
                torch.zeros_like(weight_sum),
            )
            diff = era_gain - mean_gain.unsqueeze(2)
            var = torch.where(
                weight_sum > 0,
                (weights_eff * diff * diff).sum(dim=2) / safe_weight_sum,
                torch.zeros_like(weight_sum),
            )
            std_gain = torch.sqrt(torch.clamp_min(var, 0.0))

            agreement_mean = torch.zeros_like(mean_gain)
            if direction_weight != 0.0 and torch.any(parent_dirs != 0):
                active_mask = parent_dirs != 0
                if active_mask.any():
                    parent_active = parent_dirs[active_mask].view(1, -1, 1, 1)
                    left_active = left_grad[:, active_mask, :, :] / (
                        left_hess[:, active_mask, :, :] + lambda_l2
                    )
                    right_active = right_grad[:, active_mask, :, :] / (
                        right_hess[:, active_mask, :, :] + lambda_l2
                    )
                    left_sign = torch.sign(-left_active)
                    right_sign = torch.sign(-right_active)
                    agree = 0.5 * (
                        (left_sign == parent_active).to(left_sign.dtype)
                        + (right_sign == parent_active).to(right_sign.dtype)
                    )
                    weights_active = weights_eff[:, active_mask, :, :]
                    safe_weight_active = safe_weight_sum[:, active_mask, :]
                    raw = torch.where(
                        safe_weight_active > 0,
                        (weights_active * agree).sum(dim=2) / safe_weight_active,
                        torch.zeros_like(safe_weight_active),
                    )
                    agreement_mean[:, active_mask, :] = raw

            score = mean_gain - lambda_dro * std_gain + direction_weight * agreement_mean

            valid_global = (
                (left_count_total >= self.config.min_samples_leaf)
                & (right_count_total >= self.config.min_samples_leaf)
            )
            score = torch.where(
                valid_global,
                score,
                torch.full_like(score, float("-inf")),
            )

            stats.score_ms += (perf_counter() - score_start) * 1000.0

            score_perm = score.permute(1, 0, 2).reshape(num_nodes, -1)
            block_best, block_idx = score_perm.max(dim=1)
            update_mask = torch.isfinite(block_best) & (block_best > best_scores)
            if not torch.any(update_mask):
                continue

            num_thresholds_block = num_thresholds
            feat_idx = block_idx // num_thresholds_block
            thresh_idx = block_idx % num_thresholds_block

            node_indices = torch.arange(num_nodes, device=self._device)
            left_total_perm = left_count_total.permute(1, 0, 2)
            grad_total_perm = prefix_grad_total.permute(1, 0, 2)
            hess_total_perm = prefix_hess_total.permute(1, 0, 2)

            chosen_left_count = left_total_perm[node_indices, feat_idx, thresh_idx]
            chosen_left_grad = grad_total_perm[node_indices, feat_idx, thresh_idx]
            chosen_left_hess = hess_total_perm[node_indices, feat_idx, thresh_idx]

            best_scores = torch.where(update_mask, block_best, best_scores)
            best_features = torch.where(update_mask, block[feat_idx], best_features)
            best_thresholds = torch.where(update_mask, thresholds[thresh_idx], best_thresholds)
            best_left_grad = torch.where(update_mask, chosen_left_grad, best_left_grad)
            best_left_hess = torch.where(update_mask, chosen_left_hess, best_left_hess)
            best_left_count = torch.where(update_mask, chosen_left_count, best_left_count)

        for ctx_idx, ctx in enumerate(contexts):
            node_idx = index_map[ctx_idx]
            feature = int(best_features[ctx_idx].item())
            if feature < 0:
                continue
            threshold = int(best_thresholds[ctx_idx].item())
            score_val = float(best_scores[ctx_idx].item())
            left_grad_val = float(best_left_grad[ctx_idx].item())
            left_hess_val = float(best_left_hess[ctx_idx].item())
            left_count_val = int(round(best_left_count[ctx_idx].item()))
            right_grad_val = float(total_grad_nodes[ctx_idx].item() - left_grad_val)
            right_hess_val = float(total_hess_nodes[ctx_idx].item() - left_hess_val)
            right_count_val = int(total_count_nodes_int[ctx_idx].item() - left_count_val)

            part_start = perf_counter()
            left_rows, right_rows = self._partition_rows(
                bins, ctx.shard.era_rows, feature, threshold
            )
            stats.partition_ms += (perf_counter() - part_start) * 1000.0

            decisions[node_idx] = SplitDecision(
                feature=feature,
                threshold=threshold,
                score=score_val,
                left_grad=left_grad_val,
                left_hess=left_hess_val,
                right_grad=right_grad_val,
                right_hess=right_hess_val,
                left_count=left_count_val,
                right_count=right_count_val,
                left_rows=left_rows,
                right_rows=right_rows,
            )

        return decisions, stats

    def _find_best_split(
        self,
        node: NodeShard,
        grad: torch.Tensor,
        hess: torch.Tensor,
        bins: torch.Tensor,
        feature_subset: torch.Tensor,
    ) -> SplitDecision | None:
        decisions, _ = self._find_best_splits_batched(
            [node], grad, hess, bins, feature_subset
        )
        return decisions[0]

    def _compute_era_weights(self, counts: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(counts, dtype=torch.float32)
        positive = counts > 0
        if not torch.any(positive):
            return weights
        era_alpha = max(0.0, float(self.config.era_alpha))
        if era_alpha > 0.0:
            weights[positive] = counts[positive] + era_alpha
        else:
            weights[positive] = 1.0
        return weights

    def _weighted_welford(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        era_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self._device)
            return empty, empty.clone(), empty.clone()

        dtype = values.dtype
        weights = era_weights.to(dtype).unsqueeze(1)
        mask = valid_mask.to(dtype)
        effective_weights = weights * mask

        weight_sum = effective_weights.sum(dim=0)
        safe_weight_sum = torch.clamp_min(weight_sum, 1e-12)

        weighted_values = effective_weights * values
        mean = weighted_values.sum(dim=0) / safe_weight_sum
        mean = torch.where(weight_sum > 0, mean, torch.zeros_like(mean))

        diff = values - mean.unsqueeze(0)
        variance = (effective_weights * diff * diff).sum(dim=0) / safe_weight_sum
        variance = torch.where(weight_sum > 0, variance, torch.zeros_like(variance))

        std = torch.sqrt(torch.clamp_min(variance, 0.0))
        return mean.to(torch.float32), std.to(torch.float32), weight_sum.to(torch.float32)

    def _directional_agreement(
        self,
        left_values: torch.Tensor,
        right_values: torch.Tensor,
        valid_mask: torch.Tensor,
        era_weights: torch.Tensor,
        parent_dir: torch.Tensor,
        weight_sum: torch.Tensor,
    ) -> torch.Tensor:
        if parent_dir.item() == 0:
            return torch.zeros(
                left_values.shape[1], dtype=torch.float32, device=left_values.device
            )
        dtype = left_values.dtype
        parent = parent_dir.to(dtype)
        weights = era_weights.to(dtype).unsqueeze(1)
        mask = valid_mask.to(dtype)
        left_sign = torch.sign(left_values)
        right_sign = torch.sign(right_values)
        agree = 0.5 * (
            (left_sign == parent).to(dtype) + (right_sign == parent).to(dtype)
        )
        weighted = (agree * weights * mask).sum(dim=0)
        denom = torch.clamp_min(weight_sum.to(dtype), 1e-12)
        return torch.where(weight_sum > 0, weighted / denom, torch.zeros_like(weight_sum))

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

    def _batched_histograms_nodes(
        self,
        bins_matrix: torch.Tensor,
        grad_rows: torch.Tensor,
        hess_rows: torch.Tensor,
        era_ids: torch.Tensor,
        node_ids: torch.Tensor,
        num_nodes: int,
        num_eras: int,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bins_matrix.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=bins_matrix.device)
            return empty, empty.clone(), empty.clone()

        rows, num_features_block = bins_matrix.shape
        device = bins_matrix.device
        stride_era = num_eras * num_bins
        stride_node = stride_era
        block_stride = num_nodes * stride_era
        base = (
            torch.arange(num_features_block, device=device, dtype=torch.int64) * block_stride
        ).view(1, num_features_block)
        key = (
            base
            + node_ids.view(rows, 1) * stride_node
            + era_ids.view(rows, 1) * num_bins
            + bins_matrix
        )
        key_flat = key.reshape(-1)

        hist_size = int(num_features_block * num_nodes * stride_era)
        counts = torch.bincount(key_flat, minlength=hist_size).reshape(
            num_features_block, num_nodes, num_eras, num_bins
        )

        grad_weights = grad_rows.view(rows, 1).expand(rows, num_features_block).reshape(-1)
        hess_weights = hess_rows.view(rows, 1).expand(rows, num_features_block).reshape(-1)
        grad_hist = torch.bincount(
            key_flat, weights=grad_weights, minlength=hist_size
        ).reshape(num_features_block, num_nodes, num_eras, num_bins)
        hess_hist = torch.bincount(
            key_flat, weights=hess_weights, minlength=hist_size
        ).reshape(num_features_block, num_nodes, num_eras, num_bins)

        return counts.to(torch.float32), grad_hist, hess_hist

    def _batched_histograms(
        self,
        bins_matrix: torch.Tensor,
        grad_rows: torch.Tensor,
        hess_rows: torch.Tensor,
        era_ids: torch.Tensor,
        num_eras: int,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build histograms for a block of features with shared bincounts."""

        if bins_matrix.numel() == 0:
            empty = torch.empty(0, device=bins_matrix.device, dtype=torch.float32)
            return empty, empty.clone(), empty.clone()

        rows, num_features_block = bins_matrix.shape
        device = bins_matrix.device
        stride = num_eras * num_bins
        base = (
            torch.arange(num_features_block, device=device, dtype=torch.int64) * stride
        ).view(1, num_features_block)
        key = base + era_ids.view(rows, 1) * num_bins + bins_matrix
        key_flat = key.reshape(-1)

        hist_size = int(num_features_block * stride)
        counts = torch.bincount(key_flat, minlength=hist_size).reshape(
            num_features_block, num_eras, num_bins
        )

        grad_weights = grad_rows.view(rows, 1).expand(rows, num_features_block).reshape(-1)
        hess_weights = hess_rows.view(rows, 1).expand(rows, num_features_block).reshape(-1)
        grad_hist = torch.bincount(
            key_flat, weights=grad_weights, minlength=hist_size
        ).reshape(num_features_block, num_eras, num_bins)
        hess_hist = torch.bincount(
            key_flat, weights=hess_weights, minlength=hist_size
        ).reshape(num_features_block, num_eras, num_bins)

        return counts.to(torch.float32), grad_hist, hess_hist

    def _validate_histogram_subtraction(
        self,
        left_count: torch.Tensor,
        right_count: torch.Tensor,
        total_count: torch.Tensor,
        left_grad: torch.Tensor,
        right_grad: torch.Tensor,
        total_grad: torch.Tensor,
        left_hess: torch.Tensor,
        right_hess: torch.Tensor,
        total_hess: torch.Tensor,
    ) -> torch.Tensor:
        eps_count = 1e-3
        eps_grad = 1e-6
        eps_hess = 1e-6

        total_count_expand = total_count.expand_as(left_count)
        total_grad_expand = total_grad.expand_as(left_grad)
        total_hess_expand = total_hess.expand_as(left_hess)

        count_ok = torch.abs(left_count + right_count - total_count_expand) <= eps_count
        grad_ok = torch.abs(left_grad + right_grad - total_grad_expand) <= eps_grad
        hess_ok = torch.abs(left_hess + right_hess - total_hess_expand) <= eps_hess
        finite_ok = (
            torch.isfinite(left_count)
            & torch.isfinite(right_count)
            & torch.isfinite(left_grad)
            & torch.isfinite(right_grad)
            & torch.isfinite(left_hess)
            & torch.isfinite(right_hess)
        )
        nonneg_ok = (right_count >= -eps_count) & (right_hess >= -eps_hess)

        combined = count_ok & grad_ok & hess_ok & finite_ok & nonneg_ok
        per_node = combined.permute(1, 0, 2, 3).reshape(combined.shape[1], -1)
        return per_node.all(dim=1)

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
