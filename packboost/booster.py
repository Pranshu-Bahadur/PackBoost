"""Pure-Python PackBoost baseline backed by vectorised torch ops."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
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
    _compiled: dict[str, dict[str, torch.Tensor]] = field(
        init=False, repr=False, default_factory=dict
    )

    def _ensure_compiled(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Materialise SoA tensors for ``device`` and cache them."""

        key = str(device)
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled

        features = torch.tensor(
            [node.feature for node in self.nodes],
            dtype=torch.int32,
            device=device,
        )
        thresholds = torch.tensor(
            [node.threshold for node in self.nodes],
            dtype=torch.int32,
            device=device,
        )
        left = torch.tensor(
            [node.left for node in self.nodes],
            dtype=torch.int32,
            device=device,
        )
        right = torch.tensor(
            [node.right for node in self.nodes],
            dtype=torch.int32,
            device=device,
        )
        values = torch.tensor(
            [node.value for node in self.nodes],
            dtype=torch.float32,
            device=device,
        )
        is_leaf = torch.tensor(
            [node.is_leaf for node in self.nodes],
            dtype=torch.bool,
            device=device,
        )

        compiled = {
            "feature": features,
            "threshold": thresholds,
            "left": left,
            "right": right,
            "value": values,
            "is_leaf": is_leaf,
        }
        self._compiled[key] = compiled
        return compiled

    def predict_bins(self, bins: torch.Tensor) -> torch.Tensor:
        """Apply the tree to binned features using vectorised routing."""

        num_rows = bins.shape[0]
        device = bins.device
        if num_rows == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        compiled = self._ensure_compiled(device)
        node_indices = torch.zeros(num_rows, dtype=torch.int32, device=device)
        outputs = torch.zeros(num_rows, dtype=torch.float32, device=device)
        active_rows = torch.arange(num_rows, dtype=torch.int64, device=device)

        while active_rows.numel() > 0:
            nodes_active = node_indices.index_select(0, active_rows).to(torch.int64)
            leaf_mask = compiled["is_leaf"].index_select(0, nodes_active)

            if leaf_mask.all():
                outputs[active_rows] = compiled["value"].index_select(0, nodes_active)
                break

            if leaf_mask.any():
                leaf_rows = active_rows[leaf_mask]
                leaf_nodes = nodes_active[leaf_mask]
                outputs[leaf_rows] = compiled["value"].index_select(0, leaf_nodes)
                active_rows = active_rows[~leaf_mask]
                nodes_active = nodes_active[~leaf_mask]
                if active_rows.numel() == 0:
                    break
            features = compiled["feature"].index_select(0, nodes_active).to(torch.int64)
            thresholds = compiled["threshold"].index_select(0, nodes_active).to(
                torch.int64
            )
            left = compiled["left"].index_select(0, nodes_active)
            right = compiled["right"].index_select(0, nodes_active)

            row_features = bins.index_select(0, active_rows)
            feature_values = row_features.gather(
                1, features.view(-1, 1)
            ).squeeze(1).to(torch.int64)
            go_left = feature_values <= thresholds
            next_nodes = torch.where(go_left, left, right)
            node_indices.index_copy_(
                0, active_rows, next_nodes.to(torch.int32)
            )

        # Assign remaining active rows that ended on leaves in the last iteration.
        if active_rows.numel() > 0:
            final_nodes = node_indices.index_select(0, active_rows).to(torch.int64)
            outputs[active_rows] = compiled["value"].index_select(0, final_nodes)
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
    era_weights: torch.Tensor
    parent_dir: torch.Tensor
    total_grad: float
    total_hess: float
    total_count: int
    row_start: int
    row_count: int


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

    def predict_packwise(
        self, X: np.ndarray, block_size_trees: int = 16
    ) -> np.ndarray:
        """Predict using tree blocks with cached vectorised traversals."""

        if block_size_trees <= 0:
            raise ValueError("block_size_trees must be positive")
        if self._binner is None:
            raise RuntimeError("PackBoost model must be fitted before predicting")
        bins_np = apply_bins(X, self._binner.bin_edges, self.config.max_bins)
        with torch.no_grad():
            bins = torch.from_numpy(bins_np).to(device=self._device)
            num_rows = bins.shape[0]
            if not self._trees:
                return np.zeros(num_rows, dtype=np.float32)
            if self._tree_weight is None:
                raise RuntimeError("Model missing _tree_weight; call fit() first.")
            if (
                self._trained_pack_size is not None
                and int(self.config.pack_size) != self._trained_pack_size
            ):
                raise RuntimeError(
                    "Config pack_size differs from training; please keep pack_size constant."
                )
            predictions = torch.zeros(num_rows, dtype=torch.float32, device=self._device)
            tree_weight = float(self._tree_weight)
            for tree in self._trees:
                tree._ensure_compiled(self._device)
            block = int(block_size_trees)
            for start in range(0, len(self._trees), block):
                block_trees = self._trees[start : start + block]
                if not block_trees:
                    continue
                block_sum = torch.zeros_like(predictions)
                for tree in block_trees:
                    block_sum += tree.predict_bins(bins)
                predictions += tree_weight * block_sum
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

            # One-time casts
            feat_ids = feature_subset.to(self._device, dtype=torch.int32)
            if feat_ids.numel() == 0:
                stats.nodes_skipped += num_nodes_total
                return decisions, stats
            feat_ids_i64 = feat_ids.to(torch.int64)
            thresholds = torch.arange(num_thresholds, device=self._device, dtype=torch.int32)

            lambda_l2 = self.config.lambda_l2
            lambda_dro = self.config.lambda_dro
            direction_weight = self.config.direction_weight

            # Build node contexts + depth-concatenated row buffers
            contexts: list[_NodeContext] = []
            index_map: list[int] = []
            row_chunks: list[torch.Tensor] = []
            era_chunks: list[torch.Tensor] = []
            grad_chunks: list[torch.Tensor] = []
            hess_chunks: list[torch.Tensor] = []
            node_chunks: list[torch.Tensor] = []
            row_start = 0

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
                parent_dir = torch.sign(
                    torch.tensor(parent_value_total, dtype=torch.float32, device=self._device)
                )
                era_counts = torch.tensor(
                    [float(rows.numel()) for rows in node.era_rows],
                    dtype=torch.float32,
                    device=self._device,
                )
                era_weights = self._compute_era_weights(era_counts)

                context_id = len(contexts)
                contexts.append(
                    _NodeContext(
                        shard=node,
                        era_weights=era_weights,
                        parent_dir=parent_dir,
                        total_grad=total_grad,
                        total_hess=total_hess,
                        total_count=total_count,
                        row_start=row_start,
                        row_count=total_count,
                    )
                )
                index_map.append(idx)

                row_chunks.append(all_rows)
                era_chunks.append(era_ids)
                grad_chunks.append(grad_rows)
                hess_chunks.append(hess_rows)
                node_chunks.append(
                    torch.full((total_count,), context_id, dtype=torch.int64, device=self._device)
                )
                row_start += total_count
                stats.rows_total += total_count

            if not contexts:
                return decisions, stats

            stats.nodes_processed += len(contexts)
            num_nodes = len(contexts)
            num_eras = len(contexts[0].shard.era_rows)
            block_size = self._resolve_feature_block_size(int(feat_ids.numel()))
            stats.block_size = max(stats.block_size, block_size)

            all_rows_concat = torch.cat(row_chunks, dim=0)
            era_ids_concat = torch.cat(era_chunks, dim=0)
            grad_concat = torch.cat(grad_chunks, dim=0)
            hess_concat = torch.cat(hess_chunks, dim=0)
            node_concat = torch.cat(node_chunks, dim=0)
            bins_concat = bins.index_select(0, all_rows_concat)

            era_weights_tensor = torch.stack([ctx.era_weights for ctx in contexts], dim=0).contiguous()
            total_grad_nodes = torch.tensor(
                [ctx.total_grad for ctx in contexts], dtype=torch.float32, device=self._device
            )
            total_hess_nodes = torch.tensor(
                [ctx.total_hess for ctx in contexts], dtype=torch.float32, device=self._device
            )
            total_count_nodes_int = torch.tensor(
                [ctx.total_count for ctx in contexts], dtype=torch.int64, device=self._device
            )
            parent_dirs = torch.stack([ctx.parent_dir.to(torch.float32) for ctx in contexts], dim=0)

            best_scores = torch.full((num_nodes,), float("-inf"), device=self._device)
            best_features = torch.full((num_nodes,), -1, dtype=torch.int32, device=self._device)
            best_thresholds = torch.full((num_nodes,), -1, dtype=torch.int32, device=self._device)
            best_left_grad = torch.zeros(num_nodes, dtype=torch.float32, device=self._device)
            best_left_hess = torch.zeros(num_nodes, dtype=torch.float32, device=self._device)
            best_left_count = torch.full((num_nodes,), -1, dtype=torch.int64, device=self._device)

            subtract_success = torch.ones(num_nodes, dtype=torch.bool, device=self._device)
            fallback_any = torch.zeros(num_nodes, dtype=torch.bool, device=self._device)
            if self._histogram_mode == "rebuild":
                subtract_success.zero_()

            for start in range(0, feat_ids.numel(), block_size):
                block = feat_ids[start : start + block_size]                 # int32
                block_i64 = feat_ids_i64[start : start + block_size]         # int64
                if block.numel() == 0:
                    continue

                # Histograms for this feature block
                t0 = perf_counter()
                counts, grad_hist, hess_hist = self._batched_histograms_nodes(
                    bins_concat.index_select(1, block_i64),
                    grad_concat,
                    hess_concat,
                    era_ids_concat,
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

                # Prefix scans (era-wise)
                scan_start = perf_counter()
                left_count = counts[:, :, :, :-1].cumsum(dim=3)  # int64 [B,N,E,T]
                left_grad = grad_hist[:, :, :, :-1].cumsum(dim=3)  # fp32
                left_hess = hess_hist[:, :, :, :-1].cumsum(dim=3)  # fp32
                total_count_e = counts.sum(dim=3, keepdim=True)     # int64 [B,N,E,1]
                total_grad_e = grad_hist.sum(dim=3, keepdim=True)   # fp32
                total_hess_e = hess_hist.sum(dim=3, keepdim=True)   # fp32

                # Aggregate across eras (global guards and left stats)
                left_count_total = left_count.sum(dim=2)  # int64 [B,N,T]
                left_grad_total  = left_grad.sum(dim=2)   # fp32
                left_hess_total  = left_hess.sum(dim=2)   # fp32
                stats.scan_ms += (perf_counter() - scan_start) * 1000.0

                # Right by subtraction (or rebuild for failures)
                if self._histogram_mode == "rebuild":
                    right_count = counts[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                    right_grad  = grad_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                    right_hess  = hess_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                else:
                    right_count = total_count_e.expand_as(left_count) - left_count
                    right_grad  = total_grad_e.expand_as(left_grad)   - left_grad
                    right_hess  = total_hess_e.expand_as(left_hess)   - left_hess

                    '''
                    subtract_valid = self._validate_histogram_subtraction(
                        left_count, right_count, total_count_e,
                        left_grad, right_grad, total_grad_e,
                        left_hess, right_hess, total_hess_e,
                    )
                    rebuild_mask = ~subtract_valid
                    fallback_any |= rebuild_mask
                    subtract_success &= ~rebuild_mask
                    if rebuild_mask.any():
                        rc = counts[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                        rg = grad_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                        rh = hess_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                        idx = rebuild_mask.nonzero(as_tuple=True)[0]
                        right_count[:, idx, :, :] = rc[:, idx, :, :]
                        right_grad[:,  idx, :, :] = rg[:, idx, :, :]
                        right_hess[:,  idx, :, :] = rh[:, idx, :, :]
                    '''

                # Score (DES)
                score_start = perf_counter()
                valid = (left_count > 0) & (right_count > 0)  # era-wise
                if not torch.any(valid):
                    stats.score_ms += (perf_counter() - score_start) * 1000.0
                    continue

                parent_gain = 0.5 * (total_grad_e**2) / (total_hess_e + lambda_l2)
                gain_left   = 0.5 * (left_grad**2) / (left_hess + lambda_l2)
                gain_right  = 0.5 * (right_grad**2) / (right_hess + lambda_l2)
                era_gain = gain_left + gain_right - parent_gain  # [B,N,E,T]

                weights_eff = (
                    era_weights_tensor.view(1, num_nodes, num_eras, 1).to(era_gain.dtype)
                    * valid.to(era_gain.dtype)
                )
                weight_sum = weights_eff.sum(dim=2).clamp_min_(1e-12)  # [B,N,T]
                mean_gain = (weights_eff * era_gain).sum(dim=2) / weight_sum
                diff = era_gain - mean_gain.unsqueeze(2)
                var = (weights_eff * diff * diff).sum(dim=2) / weight_sum
                std_gain = torch.sqrt(var.clamp_min_(0.0))

                score = mean_gain - lambda_dro * std_gain  # [B,N,T]

                # Directional agreement (optional)
                if direction_weight != 0.0 and (parent_dirs != 0).any():
                    active = (parent_dirs != 0)
                    if active.any():
                        pa = parent_dirs[active].view(1, -1, 1, 1)
                        lg = left_grad[:, active, :, :]  / (left_hess[:, active, :, :]  + lambda_l2)
                        rg = right_grad[:, active, :, :] / (right_hess[:, active, :, :] + lambda_l2)
                        agree = 0.5 * ((torch.sign(-lg) == pa).float() + (torch.sign(-rg) == pa).float())
                        w_eff_a = weights_eff[:, active, :, :]
                        w_sum_a = w_eff_a.sum(dim=2).clamp_min_(1e-12)
                        agr = (w_eff_a * agree).sum(dim=2) / w_sum_a
                        score[:, active, :] += direction_weight * agr

                # Global min_samples_leaf (aggregate across eras)
                lc = left_count_total.permute(1, 0, 2)  # [N,B,T] int64
                rc = (total_count_nodes_int.view(num_nodes, 1, 1) - lc)
                valid_global = (lc >= self.config.min_samples_leaf) & (rc >= self.config.min_samples_leaf)

                # Prepare flattened views for selection (N, M) where M = B*T
                score_perm = score.permute(1, 0, 2)                          # [N,B,T] fp32
                score_perm = torch.where(valid_global, score_perm, torch.full_like(score_perm, float("-inf")))
                score_perm = score_perm.reshape(num_nodes, -1)                # [N, M]
                left_total_perm = lc.reshape(num_nodes, -1)                   # [N, M] int64
                grad_total_perm = left_grad_total.permute(1, 0, 2).reshape(num_nodes, -1)  # [N,M]
                hess_total_perm = left_hess_total.permute(1, 0, 2).reshape(num_nodes, -1)  # [N,M]

                # ----- O(N) argmax with flattened-index tiebreak -----
                block_best = score_perm.max(dim=1).values                     # [N]
                mask1 = (score_perm == block_best.unsqueeze(1))

                NEG_I64 = torch.iinfo(torch.int64).min
                count_masked = torch.where(mask1, left_total_perm, torch.full_like(left_total_perm, NEG_I64))
                best_count = count_masked.max(dim=1).values                   # [N]

                mask2 = mask1 & (left_total_perm == best_count.unsqueeze(1))
                M = score_perm.shape[1]
                flat_idx = torch.arange(M, device=self._device, dtype=torch.int64).view(1, -1)
                selected_idx = torch.where(mask2, flat_idx, torch.full_like(flat_idx, -1)).max(dim=1).values  # [N]
                valid_mask = (selected_idx >= 0)

                gather_idx = selected_idx.clamp_min(0).view(-1, 1)
                candidate_score      = score_perm.gather(1, gather_idx).squeeze(1)
                candidate_left_count = left_total_perm.gather(1, gather_idx).squeeze(1)
                candidate_left_grad  = grad_total_perm.gather(1, gather_idx).squeeze(1)
                candidate_left_hess  = hess_total_perm.gather(1, gather_idx).squeeze(1)

                # Unravel feature/threshold within the **current block**
                num_thresh = num_thresholds
                f_idx_in_block = (selected_idx // num_thresh).to(torch.int64)  # [N]
                t_idx_in_block = (selected_idx %  num_thresh).to(torch.int64)  # [N]

                # map to actual feature ids in this feature block
                candidate_feature   = block_i64.gather(0, f_idx_in_block).to(torch.int32)
                candidate_threshold = thresholds.gather(0, t_idx_in_block).to(torch.int32)

                # Cross-block update with full tie-break order
                better_score   = candidate_score > best_scores
                score_equal    = candidate_score == best_scores
                better_count   = candidate_left_count > best_left_count
                count_equal    = candidate_left_count == best_left_count
                better_feature = candidate_feature > best_features
                feature_equal  = candidate_feature == best_features
                better_thresh  = candidate_threshold > best_thresholds

                update_mask = valid_mask & (
                    better_score
                    | (score_equal & (better_count
                    | (count_equal & (better_feature
                        | (feature_equal & better_thresh)))))
                )

                best_scores     = torch.where(update_mask, candidate_score, best_scores)
                best_left_count = torch.where(update_mask, candidate_left_count.to(torch.int64), best_left_count)
                best_left_grad  = torch.where(update_mask, candidate_left_grad,  best_left_grad)
                best_left_hess  = torch.where(update_mask, candidate_left_hess,  best_left_hess)
                best_features   = torch.where(update_mask, candidate_feature,    best_features)
                best_thresholds = torch.where(update_mask, candidate_threshold,  best_thresholds)
                # -----------------------------------------------------

                stats.score_ms += (perf_counter() - score_start) * 1000.0

            # Validator accounting
            if self._histogram_mode == "rebuild":
                rebuild_count = num_nodes
            else:
                rebuild_count = int((~subtract_success).sum().item())
            stats.nodes_subtract_ok += int(subtract_success.sum().item())
            stats.nodes_rebuild += rebuild_count
            stats.nodes_subtract_fallback += int(fallback_any.sum().item())

            # Emit decisions
            for ctx_idx, ctx in enumerate(contexts):
                node_idx = index_map[ctx_idx]
                feature = int(best_features[ctx_idx].item())
                if feature < 0:
                    continue
                threshold = int(best_thresholds[ctx_idx].item())
                score_val = float(best_scores[ctx_idx].item())
                left_grad_val = float(best_left_grad[ctx_idx].item())
                left_hess_val = float(best_left_hess[ctx_idx].item())
                left_count_val = int(best_left_count[ctx_idx].item())
                right_grad_val = float(total_grad_nodes[ctx_idx].item() - left_grad_val)
                right_hess_val = float(total_hess_nodes[ctx_idx].item() - left_hess_val)
                right_count_val = int(total_count_nodes_int[ctx_idx].item() - best_left_count[ctx_idx].item())

                part_start = perf_counter()
                left_rows, right_rows = self._partition_rows(bins, ctx.shard.era_rows, feature, threshold)
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
            empty = torch.empty(0, dtype=torch.int64, device=bins_matrix.device)
            return empty, empty.clone(), empty.clone()

        rows, num_features_block = bins_matrix.shape
        device = bins_matrix.device
        bins_int = bins_matrix.to(torch.int64)
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
            + bins_int
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

        return counts, grad_hist, hess_hist

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
            empty = torch.empty(0, device=bins_matrix.device, dtype=torch.int64)
            return empty, empty.clone(), empty.clone()

        rows, num_features_block = bins_matrix.shape
        device = bins_matrix.device
        stride = num_eras * num_bins
        base = (
            torch.arange(num_features_block, device=device, dtype=torch.int64) * stride
        ).view(1, num_features_block)
        key = base + era_ids.view(rows, 1) * num_bins + bins_matrix.to(torch.int64)
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

        return counts, grad_hist, hess_hist

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

        left_count_f = left_count.to(torch.float64)
        right_count_f = right_count.to(torch.float64)
        total_count_f = total_count.to(torch.float64)
        left_grad_f = left_grad.to(torch.float64)
        right_grad_f = right_grad.to(torch.float64)
        total_grad_f = total_grad.to(torch.float64)
        left_hess_f = left_hess.to(torch.float64)
        right_hess_f = right_hess.to(torch.float64)
        total_hess_f = total_hess.to(torch.float64)

        total_count_expand = total_count_f.expand_as(left_count_f)
        total_grad_expand = total_grad_f.expand_as(left_grad_f)
        total_hess_expand = total_hess_f.expand_as(left_hess_f)

        count_ok = (
            torch.abs(left_count_f + right_count_f - total_count_expand) <= eps_count
        )
        grad_ok = (
            torch.abs(left_grad_f + right_grad_f - total_grad_expand) <= eps_grad
        )
        hess_ok = (
            torch.abs(left_hess_f + right_hess_f - total_hess_expand) <= eps_hess
        )
        finite_ok = (
            torch.isfinite(left_count_f)
            & torch.isfinite(right_count_f)
            & torch.isfinite(left_grad_f)
            & torch.isfinite(right_grad_f)
            & torch.isfinite(left_hess_f)
            & torch.isfinite(right_hess_f)
        )
        nonneg_ok = (right_count_f >= -eps_count) & (right_hess_f >= -eps_hess)

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
