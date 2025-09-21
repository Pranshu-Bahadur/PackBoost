"""Pure-Python PackBoost (Torch) with on-the-fly K-cuts and DES (Welford)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .config import PackBoostConfig
from .data import BinningResult, apply_bins, build_era_index, preprocess_features


# ------------------------------
# Tree structures (unchanged API)
# ------------------------------

@dataclass(slots=True)
class TreeNode:
    feature: int = -1
    threshold: int = 0
    left: int = -1
    right: int = -1
    value: float = 0.0
    is_leaf: bool = True


@dataclass(slots=True)
class Tree:
    nodes: List[TreeNode]
    _compiled: dict[str, dict[str, torch.Tensor]] = field(
        init=False, repr=False, default_factory=dict
    )

    def _ensure_compiled(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled

        features = torch.tensor([n.feature for n in self.nodes], dtype=torch.int32, device=device)
        thresholds = torch.tensor([n.threshold for n in self.nodes], dtype=torch.int32, device=device)
        left = torch.tensor([n.left for n in self.nodes], dtype=torch.int32, device=device)
        right = torch.tensor([n.right for n in self.nodes], dtype=torch.int32, device=device)
        values = torch.tensor([n.value for n in self.nodes], dtype=torch.float32, device=device)
        is_leaf = torch.tensor([n.is_leaf for n in self.nodes], dtype=torch.bool, device=device)

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
        """Vectorised routing for integer-binned features."""
        N = bins.shape[0]
        device = bins.device
        if N == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        c = self._ensure_compiled(device)
        node_idx = torch.zeros(N, dtype=torch.int32, device=device)
        out = torch.zeros(N, dtype=torch.float32, device=device)
        active = torch.arange(N, dtype=torch.int64, device=device)

        while active.numel() > 0:
            nodes = node_idx.index_select(0, active).to(torch.int64)
            is_leaf = c["is_leaf"].index_select(0, nodes)
            if is_leaf.all():
                out[active] = c["value"].index_select(0, nodes)
                break
            if is_leaf.any():
                leaf_rows = active[is_leaf]
                leaf_nodes = nodes[is_leaf]
                out[leaf_rows] = c["value"].index_select(0, leaf_nodes)
                active = active[~is_leaf]
                nodes = nodes[~is_leaf]
                if active.numel() == 0:
                    break

            feat = c["feature"].index_select(0, nodes).to(torch.int64)
            thr = c["threshold"].index_select(0, nodes).to(torch.int64)
            row_feat = bins.index_select(0, active)
            val = row_feat.gather(1, feat.view(-1, 1)).squeeze(1).to(torch.int64)
            go_left = val <= thr
            next_idx = torch.where(go_left, c["left"].index_select(0, nodes), c["right"].index_select(0, nodes))
            node_idx.index_copy_(0, active, next_idx.to(torch.int32))

        if active.numel() > 0:
            nodes = node_idx.index_select(0, active).to(torch.int64)
            out[active] = c["value"].index_select(0, nodes)
        return out


class TreeBuilder:
    def __init__(self) -> None:
        self.nodes: List[TreeNode] = [TreeNode()]

    def set_leaf(self, node_id: int, value: float) -> None:
        n = self.nodes[node_id]
        n.value = float(value); n.is_leaf = True
        n.feature = -1; n.threshold = 0; n.left = -1; n.right = -1

    def split(self, node_id: int, feature: int, threshold: int) -> Tuple[int, int]:
        n = self.nodes[node_id]
        n.feature = int(feature); n.threshold = int(threshold); n.is_leaf = False
        l = len(self.nodes); r = l + 1
        n.left = l; n.right = r
        self.nodes.append(TreeNode()); self.nodes.append(TreeNode())
        return l, r

    def build(self) -> Tree:
        return Tree(nodes=self.nodes)


# ------------------------------
# Frontier / decisions / metrics
# ------------------------------

@dataclass(slots=True)
class NodeShard:
    tree_id: int
    node_id: int
    depth: int
    era_rows: list[torch.Tensor]  # per-era row indices


@dataclass(slots=True)
class SplitDecision:
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
            "nodes_rebuild": self.nodes_rebuild,
            "nodes_subtract_fallback": self.nodes_subtract_fallback,
            "feature_block_size": self.block_size,
        }


@dataclass(slots=True)
class _NodeContext:
    shard: NodeShard
    era_weights: torch.Tensor        # [E]
    parent_dir: torch.Tensor         # scalar tensor {-1,0,1}
    total_grad: float
    total_hess: float
    total_count: int
    row_start: int
    row_count: int


# ------------------------------
# PackBoost (Torch)
# ------------------------------

class PackBoost:
    """Pack-parallel gradient booster with on-the-fly K cuts and DES (Welford)."""

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
        self._logger = logging.getLogger(__name__)

        # Runtime state
        self._depth_logs: list[dict[str, object]] = []
        self._binner: BinningResult | None = None
        self._trees: list[Tree] = []
        self._era_unique: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._tree_weight: float | None = None
        self._trained_pack_size: int | None = None

    # Public -------------------------------------------------------------

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
        """Squared-loss booster with pack-synchronous growth and DES."""
        X_np = np.asarray(X)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 1:
            raise ValueError("y must be 1-D")
        N, F = X_np.shape
        if N != y_np.shape[0]:
            raise ValueError("X and y row mismatch")
        era_np = np.asarray(list(era))
        if era_np.shape[0] != N:
            raise ValueError("era must align with X rows")

        uniq_era, era_inv = np.unique(era_np, return_inverse=True)
        self._era_unique = uniq_era.astype(np.int16)
        era_encoded = era_inv.astype(np.int64)
        E = int(uniq_era.shape[0])

        self._feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(F)]

        # Bin once
        self._binner = preprocess_features(X_np, self.config.max_bins)
        bins = torch.from_numpy(self._binner.bins).to(device=self._device)  # [N,F], int64/uint8 okay
        y_t = torch.from_numpy(y_np).to(device=self._device)

        # Init gradient state
        grad = torch.zeros(N, dtype=torch.float32, device=self._device)
        hess = torch.ones(N, dtype=torch.float32, device=self._device)  # squared loss
        preds = torch.zeros(N, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            self._depth_logs = []
            self._tree_weight = None
            self._trained_pack_size = None
            era_index = build_era_index(era_encoded, E)  # list[np.ndarray] of row ids per era
            base_era_rows = [r for r in era_index]
            self._trees = []

            for round_idx in range(num_rounds):
                self._update_gradients(preds, y_t, grad)
                self._update_hessians(hess)

                # New pack (B trees) with root initialisation
                pack_builders = [TreeBuilder() for _ in range(self.config.pack_size)]
                frontier: list[NodeShard] = []

                for t_id in range(self.config.pack_size):
                    shard_rows = list(base_era_rows)
                    frontier.append(NodeShard(tree_id=t_id, node_id=0, depth=0, era_rows=shard_rows))
                    rows_root = torch.cat([r for r in shard_rows if r.numel() > 0], dim=0)
                    if rows_root.numel() == 0:
                        pack_builders[t_id].set_leaf(0, 0.0)
                    else:
                        g = float(grad[rows_root].sum().item())
                        h = float(hess[rows_root].sum().item())
                        val = -g / (h + self.config.lambda_l2)
                        pack_builders[t_id].set_leaf(0, val)

                # Depth loop (shared feature subset per depth)
                for depth in range(self.config.max_depth):
                    active_nodes = [n for n in frontier if self._node_has_capacity(n)]
                    if not active_nodes:
                        break

                    feat_subset = self._sample_features(F)  # shared across pack
                    stats_depth = DepthInstrumentation()

                    # Batched node processing
                    decisions, stats_batch = self._find_best_splits_batched(
                        active_nodes, grad, hess, bins, feat_subset
                    )
                    stats_depth += stats_batch

                    # Apply decisions → grow frontier
                    next_frontier: list[NodeShard] = []
                    for node, dec in zip(active_nodes, decisions):
                        tb = pack_builders[node.tree_id]
                        if dec is None:
                            rows_all = torch.cat([r for r in node.era_rows if r.numel() > 0], 0)
                            leaf_val = 0.0
                            if rows_all.numel() > 0:
                                g = float(grad[rows_all].sum().item())
                                h = float(hess[rows_all].sum().item())
                                leaf_val = -g / (h + self.config.lambda_l2)
                            tb.set_leaf(node.node_id, leaf_val)
                            continue

                        left_id, right_id = tb.split(node.node_id, dec.feature, dec.threshold)
                        lv = -dec.left_grad / (dec.left_hess + self.config.lambda_l2)
                        rv = -dec.right_grad / (dec.right_hess + self.config.lambda_l2)
                        tb.set_leaf(left_id, lv)
                        tb.set_leaf(right_id, rv)

                        next_frontier.append(NodeShard(node.tree_id, left_id, node.depth + 1, list(dec.left_rows)))
                        next_frontier.append(NodeShard(node.tree_id, right_id, node.depth + 1, list(dec.right_rows)))

                    frontier = next_frontier

                    # Logs
                    depth_log = stats_depth.to_dict()
                    depth_log.update({
                        "depth": depth,
                        "feature_subset_size": int(feat_subset.numel()),
                        "pack_size": self.config.pack_size,
                        "layer_feature_fraction": float(self.config.layer_feature_fraction),
                        "histogram_mode": self._histogram_mode,
                        "seed": self.config.random_state,
                        "torch_threads": torch.get_num_threads(),
                    })
                    if depth_log["feature_block_size"] == 0:
                        depth_log["feature_block_size"] = self._resolve_feature_block_size(int(feat_subset.numel()))
                    self._depth_logs.append(depth_log)
                    if self._logger.isEnabledFor(logging.INFO):
                        self._logger.info(json.dumps(depth_log))

                # Pack prediction and averaging (lr / B)
                pack_sum = torch.zeros_like(preds)
                for tb in pack_builders:
                    tr = tb.build()
                    self._trees.append(tr)
                    pack_sum += tr.predict_bins(bins)
                per_tree_w = float(self.config.learning_rate) / float(self.config.pack_size)
                preds += per_tree_w * pack_sum
                self._tree_weight = per_tree_w
                self._trained_pack_size = int(self.config.pack_size)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._binner is None:
            raise RuntimeError("Model must be fitted before predict()")
        bins_np = apply_bins(X, self._binner.bin_edges, self.config.max_bins)
        with torch.no_grad():
            bins = torch.from_numpy(bins_np).to(device=self._device)
            if not self._trees:
                return np.zeros(bins.shape[0], dtype=np.float32)
            if self._tree_weight is None:
                raise RuntimeError("Missing _tree_weight; call fit() first.")
            if self._trained_pack_size is not None and int(self.config.pack_size) != self._trained_pack_size:
                raise RuntimeError("pack_size differs from training; keep it constant.")
            pred = torch.zeros(bins.shape[0], dtype=torch.float32, device=self._device)
            for tr in self._trees:
                pred += self._tree_weight * tr.predict_bins(bins)
        return pred.cpu().numpy()

    def predict_packwise(self, X: np.ndarray, block_size_trees: int = 800 // 8) -> np.ndarray:
        if block_size_trees <= 0:
            raise ValueError("block_size_trees must be positive")
        if self._binner is None:
            raise RuntimeError("Model must be fitted before predict_packwise()")
        bins_np = apply_bins(X, self._binner.bin_edges, self.config.max_bins)
        with torch.no_grad():
            bins = torch.from_numpy(bins_np).to(device=self._device)
            N = bins.shape[0]
            if not self._trees:
                return np.zeros(N, dtype=np.float32)
            if self._tree_weight is None:
                raise RuntimeError("Missing _tree_weight; call fit() first.")
            if self._trained_pack_size is not None and int(self.config.pack_size) != self._trained_pack_size:
                raise RuntimeError("pack_size differs from training; keep it constant.")
            pred = torch.zeros(N, dtype=torch.float32, device=self._device)
            w = float(self._tree_weight)
            for tr in self._trees:
                tr._ensure_compiled(self._device)
            B = int(block_size_trees)
            for s in range(0, len(self._trees), B):
                blk = self._trees[s : s + B]
                if not blk:
                    continue
                acc = torch.zeros_like(pred)
                for tr in blk:
                    acc += tr.predict_bins(bins)
                pred += w * acc
        return pred.cpu().numpy()

    # Internals ----------------------------------------------------------

    def _sample_features(self, num_features: int) -> torch.Tensor:
        frac = float(self.config.layer_feature_fraction)
        k = max(1, int(round(frac * num_features)))
        k = min(k, num_features)
        perm = torch.randperm(num_features, generator=self._rng)
        return perm[:k]

    def _node_has_capacity(self, node: NodeShard) -> bool:
        total = sum(int(r.numel()) for r in node.era_rows)
        return total >= 2 * self.config.min_samples_leaf

    def _resolve_feature_block_size(self, subset_size: int) -> int:
        blk = int(self.config.feature_block_size)
        if blk <= 0 or blk > subset_size:
            return subset_size
        return blk

    # --- DES utilities ---

    @staticmethod
    def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        if a.size < 2 or b.size < 2:
            return float("nan")
        c = np.corrcoef(a, b)[0, 1]
        return float(c)

    def _compute_era_weights(self, counts: torch.Tensor) -> torch.Tensor:
        """Equal-era weights for present eras; optional +alpha on counts."""
        w = torch.zeros_like(counts, dtype=torch.float32, device=counts.device)
        pos = counts > 0
        if not torch.any(pos):
            return w
        alpha = max(0.0, float(self.config.era_alpha))
        if alpha > 0.0:
            w[pos] = counts[pos].to(torch.float32) + alpha
        else:
            w[pos] = 1.0
        return w

    # --- Histogram builders (Torch) ---

    def _batched_histograms_nodes(
        self,
        bins_block: torch.Tensor,    # [R, B]   (features in this block)
        grad_rows: torch.Tensor,     # [R]
        hess_rows: torch.Tensor,     # [R]
        era_ids: torch.Tensor,       # [R]
        node_ids: torch.Tensor,      # [R]
        num_nodes: int,
        num_eras: int,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared bincount across (node, era, bin) for a feature block."""
        if bins_block.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=bins_block.device)
            return empty, empty.clone(), empty.clone()

        R, B = bins_block.shape
        dev = bins_block.device
        bins_i64 = bins_block.to(torch.int64)
        stride_era = num_eras * num_bins
        block_stride = num_nodes * stride_era

        base = (torch.arange(B, device=dev, dtype=torch.int64) * block_stride).view(1, B)
        key = base + node_ids.view(R, 1) * stride_era + era_ids.view(R, 1) * num_bins + bins_i64
        key_flat = key.reshape(-1)

        hist_size = int(B * num_nodes * stride_era)
        counts = torch.bincount(key_flat, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)

        gw = grad_rows.view(R, 1).expand(R, B).reshape(-1)
        hw = hess_rows.view(R, 1).expand(R, B).reshape(-1)
        grad_hist = torch.bincount(key_flat, weights=gw, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)
        hess_hist = torch.bincount(key_flat, weights=hw, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)

        return counts, grad_hist, hess_hist

    # --- K-cut selection helpers (on-the-fly thermometer lanes) ---

    def _even_cut_indices(self, num_bins: int, k: int) -> torch.Tensor:
        k = max(1, min(k, max(1, num_bins - 1)))
        idx = torch.linspace(0, num_bins - 2, k, device=self._device)
        idx = torch.round(idx).to(torch.int64)
        return torch.unique(idx, sorted=True).clamp_(0, num_bins - 2)

    def _mass_cut_indices(self, counts_block: torch.Tensor, k: int) -> torch.Tensor:
        """Per-feature cut selection by mass quantiles over (nodes, eras)."""
        B, N, E, BINS = counts_block.shape
        k = max(1, min(k, max(1, BINS - 1)))
        outs = []
        for b in range(B):
            m = counts_block[b].sum(dim=(0, 1))  # [bins]
            total = m.sum()
            if total <= 0:
                outs.append(self._even_cut_indices(BINS, k))
                continue
            cdf = torch.cumsum(m.to(torch.float64), dim=0)
            q = torch.linspace(0, float(total.item()) * (1 - 1e-12), k, device=m.device)
            idx = torch.searchsorted(cdf, q).clamp_(max=BINS - 1)
            thr = (idx - 1).clamp_(0, BINS - 2).to(torch.int64)
            thr = torch.unique(thr, sorted=True)
            if thr.numel() < k:
                pad = self._even_cut_indices(BINS, k - thr.numel())
                thr = torch.unique(torch.cat([thr, pad]), sorted=True).clamp_(0, BINS - 2)
            outs.append(thr)
        K = max(k, max(t.numel() for t in outs))
        ret = torch.empty((B, K), dtype=torch.int64, device=counts_block.device)
        for i, t in enumerate(outs):
            if t.numel() < K:
                pad = self._even_cut_indices(BINS, K - t.numel())
                t = torch.unique(torch.cat([t, pad]), sorted=True).clamp_(0, BINS - 2)
            ret[i] = t
        return ret  # [B,K]

    # --- Core: find best splits for batched nodes ---

    def _find_best_splits_batched(
        self,
        nodes: Sequence[NodeShard],
        grad: torch.Tensor,
        hess: torch.Tensor,
        bins: torch.Tensor,
        feature_subset: torch.Tensor,
    ) -> tuple[list[SplitDecision | None], DepthInstrumentation]:
        stats = DepthInstrumentation()
        M = len(nodes)
        decisions: list[SplitDecision | None] = [None] * M
        if M == 0:
            return decisions, stats

        num_bins = int(self.config.max_bins)
        full_T = num_bins - 1
        if full_T <= 0:
            stats.nodes_skipped += M
            return decisions, stats

        feat_ids = feature_subset.to(self._device, dtype=torch.int32)
        if feat_ids.numel() == 0:
            stats.nodes_skipped += M
            return decisions, stats
        feat_ids_i64 = feat_ids.to(torch.int64)

        lam_l2 = float(self.config.lambda_l2)
        lam_dro = float(self.config.lambda_dro)
        dir_w = float(self.config.direction_weight)

        # Build contexts & concat rows across nodes for one-pass histograms
        contexts: list[_NodeContext] = []
        index_map: list[int] = []
        row_chunks: list[torch.Tensor] = []
        era_chunks: list[torch.Tensor] = []
        grad_chunks: list[torch.Tensor] = []
        hess_chunks: list[torch.Tensor] = []
        node_chunks: list[torch.Tensor] = []
        row_start = 0

        # Build era weights per node
        for idx, node in enumerate(nodes):
            all_rows, era_ids = self._stack_node_rows(node.era_rows)
            total_count = int(all_rows.numel())
            if total_count < 2 * self.config.min_samples_leaf:
                stats.nodes_skipped += 1
                continue

            g_rows = grad.index_select(0, all_rows)
            h_rows = hess.index_select(0, all_rows)
            g_tot = float(g_rows.sum().item())
            h_tot = float(h_rows.sum().item())

            parent_value_total = -g_tot / (h_tot + lam_l2)
            parent_dir = torch.sign(torch.tensor(parent_value_total, dtype=torch.float32, device=self._device))

            era_counts = torch.tensor(
                [float(r.numel()) for r in node.era_rows],
                dtype=torch.float32, device=self._device
            )
            era_w = self._compute_era_weights(era_counts)

            cid = len(contexts)
            contexts.append(
                _NodeContext(
                    shard=node, era_weights=era_w, parent_dir=parent_dir,
                    total_grad=g_tot, total_hess=h_tot, total_count=total_count,
                    row_start=row_start, row_count=total_count
                )
            )
            index_map.append(idx)

            row_chunks.append(all_rows)
            era_chunks.append(era_ids)
            grad_chunks.append(g_rows)
            hess_chunks.append(h_rows)
            node_chunks.append(torch.full((total_count,), cid, dtype=torch.int64, device=self._device))
            row_start += total_count
            stats.rows_total += total_count

        if not contexts:
            return decisions, stats

        stats.nodes_processed += len(contexts)
        Nnodes = len(contexts)
        Eras = len(contexts[0].shard.era_rows)
        block_size = self._resolve_feature_block_size(int(feat_ids.numel()))
        stats.block_size = max(stats.block_size, block_size)

        rows_cat = torch.cat(row_chunks, 0)
        era_cat = torch.cat(era_chunks, 0)
        grad_cat = torch.cat(grad_chunks, 0)
        hess_cat = torch.cat(hess_chunks, 0)
        node_cat = torch.cat(node_chunks, 0)
        bins_cat = bins.index_select(0, rows_cat)

        era_weights_tensor = torch.stack([c.era_weights for c in contexts], 0).contiguous()  # [Nnodes, E]
        total_grad_nodes = torch.tensor([c.total_grad for c in contexts], dtype=torch.float32, device=self._device)  # [Nnodes]
        total_hess_nodes = torch.tensor([c.total_hess for c in contexts], dtype=torch.float32, device=self._device)
        total_count_nodes = torch.tensor([c.total_count for c in contexts], dtype=torch.int64, device=self._device)
        parent_dirs = torch.stack([c.parent_dir.to(torch.float32) for c in contexts], 0)  # [Nnodes]

        # Best trackers
        best_scores = torch.full((Nnodes,), float("-inf"), device=self._device)
        best_features = torch.full((Nnodes,), -1, dtype=torch.int32, device=self._device)
        best_thresholds = torch.full((Nnodes,), -1, dtype=torch.int32, device=self._device)
        best_left_grad = torch.zeros(Nnodes, dtype=torch.float32, device=self._device)
        best_left_hess = torch.zeros(Nnodes, dtype=torch.float32, device=self._device)
        best_left_count = torch.full((Nnodes,), -1, dtype=torch.int64, device=self._device)

        subtract_success = torch.ones(Nnodes, dtype=torch.bool, device=self._device)
        fallback_any = torch.zeros(Nnodes, dtype=torch.bool, device=self._device)
        if self._histogram_mode == "rebuild":
            subtract_success.zero_()

        # Feature blocks
        for start in range(0, feat_ids.numel(), block_size):
            block = feat_ids[start : start + block_size]                 # int32
            block_i64 = feat_ids_i64[start : start + block_size]         # int64
            if block.numel() == 0:
                continue

            # Histograms for this block
            t0 = perf_counter()
            counts, g_hist, h_hist = self._batched_histograms_nodes(
                bins_cat.index_select(1, block_i64),
                grad_cat, hess_cat,
                era_cat, node_cat,
                Nnodes, Eras, num_bins,
            )
            stats.hist_ms += (perf_counter() - t0) * 1000.0
            stats.feature_blocks += 1
            stats.bincount_calls += 3
            if counts.numel() == 0:
                continue

            # Prefix scans
            scan_start = perf_counter()
            # Left stats up to edge (bins-1)
            left_c_full = counts[:, :, :, :-1].cumsum(3)
            left_g_full = g_hist[:, :, :, :-1].cumsum(3)
            left_h_full = h_hist[:, :, :, :-1].cumsum(3)

            tot_c_e = counts.sum(3, keepdim=True)
            tot_g_e = g_hist.sum(3, keepdim=True)
            tot_h_e = h_hist.sum(3, keepdim=True)
            stats.scan_ms += (perf_counter() - scan_start) * 1000.0

            # K-cut selection
            K = int(self.config.k_cuts)
            use_K = (K > 0 and K < full_T)
            #K = self._choose_k_for_depth(contexts, full_T)
            #use_K = (K > 0 and K < full_T)
            if use_K and self.config.cut_selection == "mass":
                cut_idx_bk = self._mass_cut_indices(counts, K)       # [B,K]
                Bblk, Nn, Ee, _ = left_c_full.shape
                sel = cut_idx_bk.view(Bblk, 1, 1, -1).expand(Bblk, Nn, Ee, -1)
                left_c = left_c_full.gather(3, sel)
                left_g = left_g_full.gather(3, sel)
                left_h = left_h_full.gather(3, sel)
                thr_k = cut_idx_bk.to(torch.int32)                    # [B,K]
                num_thresh = thr_k.shape[1]
            else:
                if use_K:
                    cut_idx = self._even_cut_indices(num_bins, K)     # [K]
                else:
                    cut_idx = torch.arange(full_T, device=self._device, dtype=torch.int64)
                sel = cut_idx.view(1, 1, 1, -1).expand_as(left_c_full[..., :cut_idx.numel()])
                left_c = left_c_full.gather(3, sel)
                left_g = left_g_full.gather(3, sel)
                left_h = left_h_full.gather(3, sel)
                thr_k = cut_idx.to(torch.int32)                       # [K]
                num_thresh = int(thr_k.numel())

            # Right by subtraction (validator optional)
            if self._histogram_mode == "rebuild":
                right_c_full = counts[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                right_g_full = g_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                right_h_full = h_hist[:, :, :, 1:].flip(3).cumsum(3).flip(3)
                # gather to K if needed
                if use_K:
                    if self.config.cut_selection == "mass":
                        sel = cut_idx_bk.view(Bblk, 1, 1, -1).expand_as(left_c)
                    else:
                        sel = cut_idx.view(1, 1, 1, -1).expand_as(left_c)
                    right_c = right_c_full.gather(3, sel)
                    right_g = right_g_full.gather(3, sel)
                    right_h = right_h_full.gather(3, sel)
                else:
                    right_c, right_g, right_h = right_c_full, right_g_full, right_h_full
            else:
                right_c = tot_c_e.expand_as(left_c) - left_c
                right_g = tot_g_e.expand_as(left_g) - left_g
                right_h = tot_h_e.expand_as(left_h) - left_h

            # DES scoring (pure: mean - λ_dro * std), Welford-style (vectorised)
            score_start = perf_counter()
            # validity: both sides have positive count (era-wise); also global min_samples_leaf later
            valid = (left_c > 0) & (right_c > 0)

            parent_gain = (0.5 * (total_grad_nodes**2) / (total_hess_nodes + lam_l2)).view(1, Nnodes, 1, 1)
            gain_left = 0.5 * (left_g**2) / (left_h + lam_l2)
            gain_right = 0.5 * (right_g**2) / (right_h + lam_l2)
            era_gain = gain_left + gain_right - parent_gain  # [B,N,E,K]

            w = era_weights_tensor.view(1, Nnodes, Eras, 1).to(era_gain.dtype) * valid.to(era_gain.dtype)
            wsum = w.sum(dim=2).clamp_min_(1e-12)  # [B,N,K]
            mean = (w * era_gain).sum(dim=2) / wsum
            diff = era_gain - mean.unsqueeze(2)
            var = (w * diff * diff).sum(dim=2) / wsum
            std = torch.sqrt(var.clamp_min_(0.0))

            score = mean - lam_dro * std  # [B,N,K]

            # optional directional term (+λ_dir * agreement), off by default
            if dir_w != 0.0 and (parent_dirs != 0).any():
                active = (parent_dirs != 0)
                if active.any():
                    pa = parent_dirs[active].view(1, -1, 1, 1).to(score.dtype)
                    lg = left_g[:, active, :, :] / (left_h[:, active, :, :] + lam_l2)
                    rg = right_g[:, active, :, :] / (right_h[:, active, :, :] + lam_l2)
                    agree = 0.5 * ((torch.sign(-lg) == pa).float() + (torch.sign(-rg) == pa).float())
                    w_a = w[:, active, :, :]
                    ws_a = w_a.sum(dim=2).clamp_min_(1e-12)
                    agr = (w_a * agree).sum(dim=2) / ws_a
                    score[:, active, :] += dir_w * agr

            # Global min_samples_leaf (aggregate across eras)
            left_total = left_c.sum(dim=2)  # [B,N,K] int64
            right_total = (total_count_nodes.view(1, Nnodes, 1) - left_total)  # [B,N,K]
            min_child = int(self.config.min_samples_leaf)
            valid_global = (left_total >= min_child) & (right_total >= min_child)

            # Prepare flattened selection per node: M = B * K candidates
            score_perm = score.permute(1, 0, 2)  # [N,B,K]
            score_perm = torch.where(valid_global.permute(1, 0, 2), score_perm, torch.full_like(score_perm, float("-inf")))
            score_flat = score_perm.reshape(Nnodes, -1)                    # [N, M]
            left_total_flat = left_total.permute(1, 0, 2).reshape(Nnodes, -1)

            best_block = score_flat.max(dim=1).values                      # [N]
            m1 = (score_flat == best_block.unsqueeze(1))
            NEG = torch.iinfo(torch.int64).min
            best_cnt = torch.where(m1, left_total_flat, torch.full_like(left_total_flat, NEG)).max(dim=1).values
            m2 = m1 & (left_total_flat == best_cnt.unsqueeze(1))
            Mflat = score_flat.shape[1]
            flat_idx = torch.arange(Mflat, device=self._device, dtype=torch.int64).view(1, -1)
            sel_idx = torch.where(m2, flat_idx, torch.full_like(flat_idx, -1)).max(dim=1).values  # [N], -1 invalid
            valid_mask = (sel_idx >= 0)
            if not torch.any(valid_mask):
                stats.score_ms += (perf_counter() - score_start) * 1000.0
                continue

            # Gather candidate stats for nodes with valid selection
            gather_idx = sel_idx.clamp_min(0).view(-1, 1)
            cand_score = score_flat.gather(1, gather_idx).squeeze(1)
            cand_left_total = left_total_flat.gather(1, gather_idx).squeeze(1)

            # Unravel (feature_in_block, k_in_block) within this block
            num_thresh = int(num_thresh)  # ensure Python int
            f_in_blk = (sel_idx // num_thresh).to(torch.int64)   # [N]
            t_in_blk = (sel_idx %  num_thresh).to(torch.int64)   # [N]

            # Map to actual feature id
            cand_feature = block_i64.gather(0, f_in_blk.clamp_(0, block_i64.numel()-1)).to(torch.int32)

            # Map to threshold (bin edge index)
            if use_K and self.config.cut_selection == "mass":
                # thr_k: [B,K]
                cand_threshold = thr_k.gather(0, f_in_blk.clamp_(0, thr_k.shape[0]-1)).gather(
                    1, t_in_blk.clamp_(0, num_thresh - 1).view(-1, 1)
                ).squeeze(1).to(torch.int32)
            else:
                # thr_k: [K]
                cand_threshold = thr_k.gather(0, t_in_blk.clamp_(0, num_thresh - 1)).to(torch.int32)

            # Gather left grad/hess totals to form leaf values after split
            left_g_tot = left_g.sum(dim=2).permute(1, 0, 2).reshape(Nnodes, -1).gather(1, gather_idx).squeeze(1)
            left_h_tot = left_h.sum(dim=2).permute(1, 0, 2).reshape(Nnodes, -1).gather(1, gather_idx).squeeze(1)

            # Cross-block update with tiebreak (score ↓, left_count ↓, feature ↑, threshold ↑)
            better_score = cand_score > best_scores
            score_equal = cand_score == best_scores
            better_cnt = cand_left_total > best_left_count
            cnt_equal = cand_left_total == best_left_count
            better_feat = cand_feature > best_features
            feat_equal = cand_feature == best_features
            better_thr = cand_threshold > best_thresholds

            upd = valid_mask & (
                better_score
                | (score_equal & (better_cnt | (cnt_equal & (better_feat | (feat_equal & better_thr)))))
            )

            best_scores = torch.where(upd, cand_score, best_scores)
            best_left_count = torch.where(upd, cand_left_total.to(torch.int64), best_left_count)
            best_left_grad = torch.where(upd, left_g_tot, best_left_grad)
            best_left_hess = torch.where(upd, left_h_tot, best_left_hess)
            best_features = torch.where(upd, cand_feature, best_features)
            best_thresholds = torch.where(upd, cand_threshold, best_thresholds)
            stats.score_ms += (perf_counter() - score_start) * 1000.0

        # Accounting
        if self._histogram_mode == "rebuild":
            rebuild_cnt = Nnodes
        else:
            rebuild_cnt = int((~subtract_success).sum().item())
        stats.nodes_subtract_ok += int(subtract_success.sum().item())
        stats.nodes_rebuild += rebuild_cnt
        stats.nodes_subtract_fallback += int(fallback_any.sum().item())

        # Emit decisions and partition rows for each node
        for cid, ctx in enumerate(contexts):
            node_idx = index_map[cid]
            feat = int(best_features[cid].item())
            if feat < 0:
                continue
            thr = int(best_thresholds[cid].item())
            score_val = float(best_scores[cid].item())
            l_g = float(best_left_grad[cid].item())
            l_h = float(best_left_hess[cid].item())
            l_cnt = int(best_left_count[cid].item())
            r_g = float(total_grad_nodes[cid].item() - l_g)
            r_h = float(total_hess_nodes[cid].item() - l_h)
            r_cnt = int(total_count_nodes[cid].item() - best_left_count[cid].item())

            part_start = perf_counter()
            left_rows, right_rows = self._partition_rows(bins, ctx.shard.era_rows, feat, thr)
            stats.partition_ms += (perf_counter() - part_start) * 1000.0

            decisions[node_idx] = SplitDecision(
                feature=feat, threshold=thr, score=score_val,
                left_grad=l_g, left_hess=l_h, right_grad=r_g, right_hess=r_h,
                left_count=l_cnt, right_count=r_cnt,
                left_rows=left_rows, right_rows=right_rows
            )

        return decisions, stats

    # --- misc helpers ---

    def _update_gradients(self, preds: torch.Tensor, targets: torch.Tensor, out: torch.Tensor) -> None:
        # squared loss: grad = preds - targets
        out.copy_(preds - targets)

    def _update_hessians(self, out: torch.Tensor) -> None:
        out.fill_(1.0)  # squared loss: constant 1

    def _stack_node_rows(self, era_rows: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rchunks: list[torch.Tensor] = []
        echunks: list[torch.Tensor] = []
        for e_idx, rows in enumerate(era_rows):
            if rows.numel() == 0:
                continue
            rchunks.append(rows)
            echunks.append(torch.full((rows.numel(),), e_idx, dtype=torch.int64, device=rows.device))
        if not rchunks:
            empty = torch.empty(0, dtype=torch.int64, device=self._device)
            return empty, empty.clone()
        return torch.cat(rchunks, 0), torch.cat(echunks, 0)

    def _partition_rows(
        self,
        bins: torch.Tensor,
        era_rows: Sequence[torch.Tensor],
        feature: int,
        threshold: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        col = bins[:, feature]
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        for rows in era_rows:
            if rows.numel() == 0:
                left.append(rows.new_empty(0)); right.append(rows.new_empty(0)); continue
            vals = col.index_select(0, rows)
            m = vals <= threshold
            left.append(rows[m]); right.append(rows[~m])
        return left, right

    def _choose_k_for_depth(self, contexts: list, full_T: int) -> int:
        """
        Decide how many thresholds K to evaluate this depth.
        Returns an integer in [1, full_T].

        Policy:
        - If config.k_cuts == 0: use full sweep (K = full_T).
        - Else choose K based on median node size, with optional depth decay.
        """

        # Full sweep if user asked for it
        k_cap = int(getattr(self.config, "k_cuts", 0))
        if k_cap == 0:
            return int(full_T)

        # Node sizes & (shared) depth for this batch
        if not contexts:
            return min(1, int(full_T))
        node_counts = torch.tensor([c.total_count for c in contexts], dtype=torch.int64)
        depth = int(contexts[0].shard.depth)  # depth-synchronous growth ⇒ same for all

        # Heuristic thresholds (you can tune or move into config later)
        thr_large = int(getattr(self.config, "k_adapt_thr_large", 100_000))
        thr_med   = int(getattr(self.config, "k_adapt_thr_medium",  20_000))

        # K tiers (bounded by user cap). Defaults are sensible if not present in config.
        k_large = int(min(k_cap, getattr(self.config, "k_cuts_large", k_cap)))
        k_med   = int(min(k_large, getattr(self.config, "k_cuts_medium", min(15, k_cap))))
        k_small = int(min(k_med,   getattr(self.config, "k_cuts_small",  max(3,  min(7, k_cap)))))

        # Choose tier by median node size (robust to outliers)
        med = int(torch.median(node_counts).item()) if node_counts.numel() else 0
        if med >= thr_large:
            K = k_large
        elif med >= thr_med:
            K = k_med
        else:
            K = k_small

        # Optional depth decay: e.g., 0.7 shrinks K deeper in the tree
        decay = float(getattr(self.config, "k_depth_decay", 1.0))
        if decay != 1.0 and depth > 0:
            K = max(1, int(round(K * (decay ** depth))))

        return int(max(1, min(K, full_T)))

