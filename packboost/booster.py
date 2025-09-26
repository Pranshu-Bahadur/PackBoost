"""Pure-Python PackBoost (Torch) with on-the-fly K-cuts and DES (Welford)."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .config import PackBoostConfig
from .data import BinningResult, apply_bins, build_era_index, preprocess_features
from . import backends as native_backends


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

        # Fast native CUDA path (per-tree) — drop-in
        if device.type == "cuda":
            try:
                from . import backends as native_backends
                if native_backends.cuda_available():
                    c = self._ensure_compiled(device)
                    # Ensure compact dtypes; no copies if already correct
                    feat = c["feature"].to(torch.int32)
                    thr  = c["threshold"].to(torch.int32)
                    left = c["left"].to(torch.int32)
                    right= c["right"].to(torch.int32)
                    val  = c["value"].to(torch.float32)
                    leaf = c["is_leaf"].to(torch.bool)

                    # bins can be int8 or uint8; backend expects feature-major [F,N]
                    bins_fm = bins.to(torch.uint8).t().contiguous()
                    return native_backends.predict_bins_cuda(
                        bins_fm, feat, thr, left, right, val, leaf
                    )
            except Exception:
                # fall back if anything goes wrong
                pass

        # CPU fallback (NumPy) remains as-is
        if device.type == "cpu":
            return self._predict_bins_cpu_numpy(bins)

        # Generic Torch fallback (kept for completeness)
        c = self._ensure_compiled(device)
        node_idx = torch.zeros(N, dtype=torch.int32, device=device)
        out = torch.zeros(N, dtype=torch.float32, device=device)
        active = torch.arange(N, dtype=torch.int64, device=device)
        while active.numel() > 0:
            nodes = node_idx.index_select(0, active).to(torch.int64)
            is_leaf = c["is_leaf"].index_select(0, nodes)
            if is_leaf.all():
                out[active] = c["value"].index_select(0, nodes); break
            if is_leaf.any():
                leaf_rows = active[is_leaf]
                leaf_nodes = nodes[is_leaf]
                out[leaf_rows] = c["value"].index_select(0, leaf_nodes)
                active = active[~is_leaf]
                nodes = nodes[~is_leaf]
                if active.numel() == 0: break
            feat = c["feature"].index_select(0, nodes).to(torch.int64)
            thr  = c["threshold"].index_select(0, nodes).to(torch.int64)
            row_feat = bins.index_select(0, active)
            val = row_feat.gather(1, feat.view(-1, 1)).squeeze(1).to(torch.int64)
            go_left = val <= thr
            next_idx = torch.where(go_left, c["left"].index_select(0, nodes), c["right"].index_select(0, nodes))
            node_idx.index_copy_(0, active, next_idx.to(torch.int32))
        if active.numel() > 0:
            nodes = node_idx.index_select(0, active).to(torch.int64)
            out[active] = c["value"].index_select(0, nodes)
        return out


    def _predict_bins_cpu_numpy(self, bins: torch.Tensor) -> torch.Tensor:
        bins_np = bins.detach().cpu().numpy()
        N = bins_np.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.float32, device=bins.device)

        compiled = self._ensure_compiled(torch.device("cpu"))
        feature = compiled["feature"].cpu().numpy()
        threshold = compiled["threshold"].cpu().numpy()
        left = compiled["left"].cpu().numpy()
        right = compiled["right"].cpu().numpy()
        value = compiled["value"].cpu().numpy()
        is_leaf = compiled["is_leaf"].cpu().numpy()

        node_idx = np.zeros(N, dtype=np.int32)
        out_np = np.zeros(N, dtype=np.float32)
        active = np.arange(N, dtype=np.int64)

        while active.size > 0:
            nodes = node_idx[active]
            leaf_mask = is_leaf[nodes]
            if leaf_mask.any():
                leaf_rows = active[leaf_mask]
                out_np[leaf_rows] = value[nodes[leaf_mask]]
                active = active[~leaf_mask]
                nodes = nodes[~leaf_mask]
                if active.size == 0:
                    break

            feat = feature[nodes]
            thr = threshold[nodes]
            row_feat = bins_np[active, feat]
            go_left = row_feat <= thr

            next_nodes = np.where(go_left, left[nodes], right[nodes])
            node_idx[active] = next_nodes

        out_tensor = torch.from_numpy(out_np)
        return out_tensor.to(device=bins.device)


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
    nodes_collapsed: int = 0
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
        self.nodes_collapsed += other.nodes_collapsed
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
            "nodes_collapsed": self.nodes_collapsed,
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
        self._use_native_cpu = False
        self._cpu_backend = None
        self._use_native_cuda = False
        self._cuda_backend = None
        self._native_validate = os.getenv("PACKBOOST_NATIVE_VALIDATE") == "1"
        self._round_metrics: list[dict[str, float]] = []
        # __init__
        self._bins_fm: torch.Tensor | None = None


        # Packed-forest cache for fast inference/eval (flattened per-pack)
        self._packed_forest: list[dict[str, torch.Tensor]] = []

        # Optional: allow disabling pack predictor with an env var
        self._disable_pack_predict = os.getenv("PACKBOOST_DISABLE_PACK_PREDICT") == "1"

        disable_native = os.getenv("PACKBOOST_DISABLE_NATIVE_CPU") == "1"
        if not disable_native and self._device.type == "cpu":
            try:
                if native_backends.cpu_available():
                    self._use_native_cpu = True
                    self._cpu_backend = native_backends
            except Exception:  # pragma: no cover - defensive guard
                self._use_native_cpu = False

        disable_native_cuda = os.getenv("PACKBOOST_DISABLE_NATIVE_CUDA") == "1"
        if not disable_native_cuda and self._device.type == "cuda":
            try:
                if native_backends.cuda_available():
                    self._use_native_cuda = True
                    self._cuda_backend = native_backends
            except Exception:  # pragma: no cover - defensive guard
                self._use_native_cuda = False

    
    def _assert_feature_major_i8(self, fm: torch.Tensor, rows_dataset: int, feat_max: int, row_max: int) -> None:
        if fm is None:
            raise RuntimeError("_bins_fm_i8 is None")
        if fm.dim() != 2:
            raise RuntimeError("bins_fm_i8 must be 2D [F,N]")
        F, N = fm.shape
        if fm.dtype != torch.int8:
            raise RuntimeError(f"bins_fm_i8 must be int8, got {fm.dtype}")
        if not fm.is_contiguous():
            raise RuntimeError("bins_fm_i8 must be contiguous")
        s0, s1 = fm.stride()
        if s0 != N or s1 != 1:
            raise RuntimeError(f"bins_fm_i8 stride must be (N,1)=({N},1), got {fm.stride()}")
        if rows_dataset != N:
            raise RuntimeError(f"rows_dataset ({rows_dataset}) != bins_fm_i8.shape[1] ({N})")
        if feat_max >= F:
            raise RuntimeError(f"feature id {feat_max} out of range F={F}")
        if row_max >= N:
            raise RuntimeError(f"row id {row_max} out of range N={N}")

    # Public -------------------------------------------------------------

    @property
    def trees(self) -> Sequence[Tree]:
        return self._trees

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        era: Iterable[int] | None,
        *,
        num_rounds: int,
        feature_names: Sequence[str] | None = None,
        eval_sets: Sequence[tuple[str, np.ndarray, np.ndarray, Iterable[int] | None]] | None = None,
        round_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> "PackBoost":
        """Squared-loss booster with pack-synchronous growth and DES."""
        X_np = np.asarray(X)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 1:
            raise ValueError("y must be 1-D")
        N, F = X_np.shape
        if N != y_np.shape[0]:
            raise ValueError("X and y row mismatch")
        if self.config.max_bins > 128:
            raise ValueError("max_bins must be ≤ 128 when using int8 bin storage")
        if era is None:
            # DES-off path: collapse all rows into a single synthetic era
            era_np = np.zeros(N, dtype=np.int16)
            era_encoded = np.zeros(N, dtype=np.int16)
            uniq_era = np.array([0], dtype=np.int16)
        else:
            era_np = np.asarray(era, dtype=np.int16)
            if era_np.shape[0] != N:
                raise ValueError("era must align with X rows")
            uniq_era, era_inv = np.unique(era_np, return_inverse=True)
            era_encoded = era_inv.astype(np.int16)
        self._era_unique = uniq_era
        E = int(self._era_unique.shape[0])

        self._feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(F)]

        # Bin once
        self._binner = preprocess_features(
            X_np,
            self.config.max_bins,
            assume_prebinned=bool(self.config.prebinned),
        )
        
        bins_np = np.asarray(self._binner.bins, dtype=np.int8)
        bins = torch.from_numpy(bins_np).to(device=self._device, dtype=torch.int8)
        y_t = torch.from_numpy(y_np).to(device=self._device)

        # Build FM caches (once)
        if self._use_native_cuda:
            self._bins_fm_i8 = bins.to(torch.int8).t().contiguous()   # frontier kernels
            self._bins_fm_u8 = self._bins_fm_i8.to(torch.uint8)       # predict kernels
        else:
            self._bins_fm_i8 = None
            self._bins_fm_u8 = None

        # Init gradient state
        grad = torch.zeros(N, dtype=torch.float32, device=self._device)
        hess = torch.ones(N, dtype=torch.float32, device=self._device)
        preds = torch.zeros(N, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            self._depth_logs = []
            self._tree_weight = None
            self._trained_pack_size = None
            self._round_metrics = []
            self._packed_forest = []
            era_index = build_era_index(era_encoded, E)
            base_era_rows = [r.to(device=self._device, dtype=torch.int64) for r in era_index]
            self._trees = []

            # (Optional) eval state building unchanged, but we will *not* run eval if env disables it
            eval_states: list[dict[str, object]] = []
            if eval_sets:
                for name, X_eval, y_eval, era_eval in eval_sets:
                    if y_eval is None:
                        raise ValueError(f"eval set '{name}' requires y values for correlation logging")
                    X_eval_np = np.asarray(X_eval)
                    X_eval_bins = (
                        apply_bins(X_eval_np, self._binner.bin_edges, self.config.max_bins)
                        if not bool(self.config.prebinned) else X_eval_np.astype(np.int8, copy=False)
                    )
                    bins_eval = torch.from_numpy(X_eval_bins).to(self._device, dtype=torch.int8)
                    y_eval_np = np.asarray(y_eval, dtype=np.float32)
                    y_eval_tensor = torch.from_numpy(y_eval_np).to(self._device)
                    era_eval_np = (np.zeros(X_eval_bins.shape[0], dtype=np.int16)
                                  if era_eval is None
                                  else np.asarray(era_eval, dtype=np.int16))
                    if era_eval_np.shape[0] != X_eval_bins.shape[0]:
                        raise ValueError(f"eval set '{name}' era ids must align with X rows")
                    eval_states.append({
                        "name": name,
                        "bins": bins_eval,
                        "bins_fm_u8": (bins_eval.to(torch.uint8).t().contiguous() if self._use_native_cuda else None),
                        "y_tensor": y_eval_tensor,
                        "era_np": era_eval_np,
                        "preds": torch.zeros_like(y_eval_tensor),
                    })

            def log_metrics(round_idx: int, round_time: float) -> None:
                preds_np = preds.cpu().numpy()
                train_corr = self._era_correlation_np(era_encoded, y_np, preds_np)
                metrics: dict[str, float] = {
                    "round": round_idx + 1,
                    "train_corr": train_corr,
                    "trees_per_second": (self.config.pack_size / round_time) if round_time > 0 else float("inf"),
                    "round_seconds": round_time,
                }
                for state in eval_states:
                    name = state["name"]
                    y_eval_np = state["y_tensor"].cpu().numpy()
                    era_eval_np = state["era_np"]
                    preds_eval_np = state["preds"].cpu().numpy()
                    metrics[f"{name}_corr"] = self._era_correlation_np(era_eval_np, y_eval_np, preds_eval_np)
                self._round_metrics.append(metrics)
                if round_callback is not None:
                    round_callback(round_idx + 1, metrics)

            # ===== boosting rounds =====
            for round_idx in range(num_rounds):
                round_start = perf_counter()
                self._update_gradients(preds, y_t, grad)
                self._update_hessians(hess)

                pack_builders = [TreeBuilder() for _ in range(self.config.pack_size)]
                frontier: list[NodeShard] = []
                leaf_assignments: list[dict[int, torch.Tensor]] = [{} for _ in range(self.config.pack_size)]

                for t_id in range(self.config.pack_size):
                    shard_rows = list(base_era_rows)
                    frontier.append(NodeShard(tree_id=t_id, node_id=0, depth=0, era_rows=shard_rows))
                    rows_root = torch.cat([r for r in shard_rows if r.numel() > 0], 0)
                    leaf_assignments[t_id][0] = rows_root
                    if rows_root.numel() == 0:
                        pack_builders[t_id].set_leaf(0, 0.0)
                    else:
                        g = float(grad[rows_root].sum().item())
                        h = float(hess[rows_root].sum().item())
                        pack_builders[t_id].set_leaf(0, -g / (h + self.config.lambda_l2))

                for depth in range(self.config.max_depth):
                    active_nodes = [n for n in frontier if self._node_has_capacity(n)]
                    if not active_nodes:
                        break
                    feat_subset = self._sample_features(F)
                    stats_depth = DepthInstrumentation()

                    decisions, stats_batch = self._find_best_splits_batched(
                        active_nodes, grad, hess, bins, feat_subset
                    )
                    stats_depth += stats_batch

                    next_frontier: list[NodeShard] = []
                    flatten_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                    for node, dec in zip(active_nodes, decisions):
                        tb = pack_builders[node.tree_id]
                        leaf_map = leaf_assignments[node.tree_id]
                        if dec is None:
                            rows_all = torch.cat([r for r in node.era_rows if r.numel() > 0], 0)
                            leaf_val = 0.0
                            if rows_all.numel() > 0:
                                g = float(grad[rows_all].sum().item())
                                h = float(hess[rows_all].sum().item())
                                leaf_val = -g / (h + self.config.lambda_l2)
                            if node.node_id not in leaf_map:
                                leaf_map[node.node_id] = rows_all
                            tb.set_leaf(node.node_id, leaf_val)
                            continue

                        left_id, right_id = tb.split(node.node_id, dec.feature, dec.threshold)
                        tb.set_leaf(left_id, -dec.left_grad / (dec.left_hess + self.config.lambda_l2))
                        tb.set_leaf(right_id, -dec.right_grad / (dec.right_hess + self.config.lambda_l2))

                        ck = id(dec)
                        if ck in flatten_cache:
                            left_all, right_all = flatten_cache[ck]
                        else:
                            left_all, _ = self._stack_node_rows(dec.left_rows)
                            right_all, _ = self._stack_node_rows(dec.right_rows)
                            flatten_cache[ck] = (left_all, right_all)

                        leaf_map.pop(node.node_id, None)
                        leaf_map[left_id] = left_all
                        leaf_map[right_id] = right_all

                        next_frontier.append(NodeShard(node.tree_id, left_id, node.depth + 1, list(dec.left_rows)))
                        next_frontier.append(NodeShard(node.tree_id, right_id, node.depth + 1, list(dec.right_rows)))

                    frontier = next_frontier

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

                # Aggregate pack contribution
                pack_sum = torch.zeros_like(preds)
                per_tree_w = float(self.config.learning_rate) / float(self.config.pack_size)

                pack_trees: list[Tree] = []
                for t_id, tb in enumerate(pack_builders):
                    tr = tb.build()
                    pack_trees.append(tr)
                    self._trees.append(tr)
                    leaves = leaf_assignments[t_id]
                    if leaves:
                        for leaf_id, rows_all in leaves.items():
                            if rows_all.numel() == 0:
                                continue
                            pack_sum.index_add_(0, rows_all, torch.full((rows_all.numel(),), float(tr.nodes[leaf_id].value),
                                                                        dtype=torch.float32, device=self._device))
                preds += per_tree_w * pack_sum
                self._tree_weight = per_tree_w
                self._trained_pack_size = int(self.config.pack_size)

                if (self._device.type == "cuda") and (self._cuda_backend is not None) and (not self._disable_pack_predict):
                    self._packed_forest.append(self._pack_trees_cuda(pack_trees, self._device))

                # ===== evaluation gating (0 or negative disables completely) =====
                eval_every_env = os.getenv("PACKBOOST_EVAL_EVERY", "1")
                try:
                    eval_every = int(eval_every_env)
                except Exception:
                    eval_every = 1
                if eval_states and eval_every > 0:
                    evaluate_now = (((round_idx + 1) % eval_every) == 0) or ((round_idx + 1) == num_rounds)
                else:
                    evaluate_now = False

                if evaluate_now:
                    for state in eval_states:
                        y_eval_tensor: torch.Tensor = state["y_tensor"]  # type: ignore[assignment]
                        preds_eval = torch.zeros_like(y_eval_tensor)
                        if (self._packed_forest and self._device.type == "cuda"
                            and self._cuda_backend is not None and not self._disable_pack_predict):
                            bins_fm_u8 = state["bins_fm_u8"]  # type: ignore[index]
                            for p in self._packed_forest:
                                out_add = self._cuda_backend.predict_pack_cuda(
                                    bins_fm_u8 if bins_fm_u8 is not None
                                    else state["bins"].to(torch.uint8).t().contiguous(),  # type: ignore[index]
                                    p["feature"], p["threshold"],
                                    p["left_abs"], p["right_abs"],
                                    p["value"], p["is_leaf"], p["offsets"],
                                    float(self._tree_weight),
                                )
                                preds_eval.add_(out_add)
                        else:
                            # CPU or per-tree CUDA fallback
                            bins_eval: torch.Tensor = state["bins"]  # type: ignore[index]
                            if self._device.type == "cuda" and self._cuda_backend is not None:
                                bins_fm_u8 = bins_eval.to(torch.uint8).t().contiguous()
                                for tr in self._trees:
                                    c = tr._ensure_compiled(self._device)
                                    out = self._cuda_backend.predict_bins_cuda(
                                        bins_fm_u8,
                                        c["feature"].to(torch.int32),
                                        c["threshold"].to(torch.int32),
                                        c["left"].to(torch.int32),
                                        c["right"].to(torch.int32),
                                        c["value"].to(torch.float32),
                                        c["is_leaf"],
                                    )
                                    preds_eval.add_(self._tree_weight * out)
                            else:
                                for tr in self._trees:
                                    preds_eval.add_(self._tree_weight * tr.predict_bins(bins_eval))
                        state["preds"] = preds_eval
                round_elapsed = perf_counter() - round_start
                log_metrics(round_idx, round_elapsed)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._binner is None:
            raise RuntimeError("Model must be fitted before predict()")
        bins_np = apply_bins(X, self._binner.bin_edges, self.config.max_bins)
        with torch.no_grad():
            bins = torch.from_numpy(bins_np).to(device=self._device, dtype=torch.int8)
            if not self._trees:
                return np.zeros(bins.shape[0], dtype=np.float32)
            if self._tree_weight is None:
                raise RuntimeError("Missing _tree_weight; call fit() first.")
            if self._trained_pack_size is not None and int(self.config.pack_size) != self._trained_pack_size:
                raise RuntimeError("pack_size differs from training; keep it constant.")

            # Fast path: cached pack-forest + CUDA backend
            if (
                self._packed_forest
                and self._device.type == "cuda"
                and self._cuda_backend is not None
                and not self._disable_pack_predict
            ):
                pred = torch.zeros(bins.shape[0], dtype=torch.float32, device=self._device)
                bins_fm_pred = bins.to(torch.uint8).t().contiguous()
                for pack in self._packed_forest:
                    out_add = self._cuda_backend.predict_pack_cuda(
                        bins_fm_pred,
                        pack["feature"], pack["threshold"],
                        pack["left_abs"], pack["right_abs"],
                        pack["value"], pack["is_leaf"], pack["offsets"],
                        float(self._tree_weight),
                    )
                    pred.add_(out_add)
                return pred.cpu().numpy()

            # Fallback: per-tree — optimized CUDA path with single FM build
            pred = torch.zeros(bins.shape[0], dtype=torch.float32, device=self._device)
            if self._device.type == "cuda" and self._cuda_backend is not None:
                bins_fm_pred = bins.to(torch.uint8).t().contiguous()
                for tr in self._trees:
                    c = tr._ensure_compiled(self._device)
                    out = self._cuda_backend.predict_bins_cuda(
                        bins_fm_pred,
                        c["feature"].to(torch.int32),
                        c["threshold"].to(torch.int32),
                        c["left"].to(torch.int32),
                        c["right"].to(torch.int32),
                        c["value"].to(torch.float32),
                        c["is_leaf"],
                    )
                    pred.add_(self._tree_weight * out)
                return pred.cpu().numpy()
            # CPU fallback
            for tr in self._trees:
                pred += self._tree_weight * tr.predict_bins(bins)
            return pred.cpu().numpy()

    @property
    def round_metrics(self) -> Sequence[dict[str, float]]:
        """Per-round metrics recorded during the most recent ``fit`` call."""

        return self._round_metrics

    def predict_packwise(self, X: np.ndarray, block_size_trees: int = 800 // 8) -> np.ndarray:
        if block_size_trees <= 0:
            raise ValueError("block_size_trees must be positive")
        if self._binner is None:
            raise RuntimeError("Model must be fitted before predict_packwise()")
        # Delegate to predict(): we already cached per-pack flattenings during fit
        return self.predict(X)

    # Internals ----------------------------------------------------------

    def _pack_trees_cuda(self, trees: Sequence[Tree], device: torch.device) -> dict[str, torch.Tensor]:
        """
        Flatten a list of trees into concatenated arrays that a CUDA kernel can
        traverse in parallel. Child indices are converted to absolute indices
        within the pack. Leaves keep is_leaf=1 and their children are set to -1.
        Returns tensors on `device` with compact dtypes.
        """
        # Count total nodes and prepare offsets
        num_trees = len(trees)
        node_counts = [len(t.nodes) for t in trees]
        offsets = [0]
        for c in node_counts:
            offsets.append(offsets[-1] + c)
        total_nodes = offsets[-1]

        # Build host arrays
        feat = torch.empty(total_nodes, dtype=torch.int32)
        thr = torch.empty_like(feat)
        left_a = torch.empty_like(feat)
        right_a = torch.empty_like(feat)
        vals = torch.empty(total_nodes, dtype=torch.float32)
        leaf_u8 = torch.empty(total_nodes, dtype=torch.uint8)

        # Fill
        for ti, t in enumerate(trees):
            base = offsets[ti]
            for i, n in enumerate(t.nodes):
                j = base + i
                feat[j] = int(n.feature)
                thr[j] = int(n.threshold)
                vals[j] = float(n.value)
                if n.is_leaf:
                    leaf_u8[j] = 1
                    left_a[j] = -1
                    right_a[j] = -1
                else:
                    leaf_u8[j] = 0
                    left_a[j] = (base + n.left) if n.left >= 0 else -1
                    right_a[j] = (base + n.right) if n.right >= 0 else -1

        # Offsets tensor (prefix sum per tree; last item = total_nodes)
        off_t = torch.tensor(offsets, dtype=torch.int32)

        # Move to device once, make contiguous
        pack = {
            "feature": feat.to(device=device, non_blocking=True).contiguous(),
            "threshold": thr.to(device=device, non_blocking=True).contiguous(),
            "left_abs": left_a.to(device=device, non_blocking=True).contiguous(),
            "right_abs": right_a.to(device=device, non_blocking=True).contiguous(),
            "value": vals.to(device=device, non_blocking=True).contiguous(),
            "is_leaf": leaf_u8.to(device=device, non_blocking=True).contiguous(),
            "offsets": off_t.to(device=device, non_blocking=True).contiguous(),
        }
        return pack

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

    def _weighted_welford(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming weighted mean/std over ``values`` masking invalid entries."""

        if values.ndim != 2:
            raise ValueError("values must be 2D [E, T]")
        values = values.to(torch.float32)
        valid = valid.to(torch.bool)
        weights = weights.to(torch.float32)
        eras, thresholds = values.shape
        mean = torch.zeros(thresholds, dtype=torch.float32, device=values.device)
        M2 = torch.zeros_like(mean)
        wsum = torch.zeros_like(mean)

        for e in range(eras):
            w_e = float(weights[e].item())
            if w_e <= 0.0:
                continue
            mask = valid[e]
            if not torch.any(mask):
                continue
            v = values[e]
            weight_vec = torch.zeros_like(mean)
            weight_vec[mask] = w_e
            delta = v - mean
            wsum_new = wsum + weight_vec
            nz = wsum_new > 0
            factor = torch.zeros_like(wsum_new)
            factor[nz] = weight_vec[nz] / wsum_new[nz]
            mean = mean + factor * delta
            M2 = M2 + weight_vec * delta * (v - mean)
            wsum = wsum_new

        safe_w = torch.clamp_min(wsum, 1e-12)
        std = torch.sqrt(torch.clamp_min(M2 / safe_w, 0.0))
        return mean, std, wsum

    def _directional_agreement(
        self,
        left_values: torch.Tensor,
        right_values: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
        weight_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted directional coherence following the DES CUDA blueprint.

        ``direction`` per era/threshold is ``+1`` if the left leaf value exceeds the
        right leaf value, otherwise ``-1``. The result is a weighted average of
        these per-era directions over valid eras.
        """

        left = left_values.to(torch.float32)
        right = right_values.to(torch.float32)
        weights = weights.to(torch.float32)
        valid = valid.to(torch.bool)
        eras, thresholds = left.shape
        accum = torch.zeros(thresholds, dtype=torch.float32, device=left.device)

        for e in range(eras):
            w_e = float(weights[e].item())
            if w_e <= 0.0:
                continue
            mask = valid[e]
            if not torch.any(mask):
                continue
            lv = left[e]
            rv = right[e]
            direction = torch.where(lv > rv, 1.0, -1.0)
            accum[mask] += w_e * direction[mask]

        denom = torch.clamp_min(weight_sum.to(torch.float32), 1e-12)
        out = torch.zeros_like(accum)
        mask = denom > 0
        out[mask] = accum[mask] / denom[mask]
        return out

    @staticmethod
    def _era_correlation_np(era_ids: np.ndarray, target: np.ndarray, preds: np.ndarray) -> float:
        if era_ids.shape[0] == 0:
            return float("nan")
        unique_eras = np.unique(era_ids)
        cors: list[float] = []
        for era in unique_eras:
            mask = era_ids == era
            y = target[mask]
            p = preds[mask]
            if y.size < 2:
                continue
            std_y = y.std(ddof=0)
            std_p = p.std(ddof=0)
            if std_y <= 1e-12 or std_p <= 1e-12:
                continue
            cov = float(((y - y.mean()) * (p - p.mean())).mean())
            corr = cov / (std_y * std_p)
            if math.isfinite(corr):
                cors.append(corr)
        return float(np.mean(cors)) if cors else float("nan")

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
        hist_size = int(B * num_nodes * num_eras * num_bins)
        use_int32 = hist_size < (1 << 31)
        dtype = torch.int32 if use_int32 else torch.int64

        if bins_block.dtype != dtype:
            bins_block = bins_block.to(dtype)
        if node_ids.dtype != dtype:
            node_ids = node_ids.to(dtype)
        if era_ids.dtype != dtype:
            era_ids = era_ids.to(dtype)

        stride_era = torch.as_tensor(num_eras * num_bins, dtype=dtype, device=dev)
        node_stride = torch.as_tensor(num_nodes, dtype=dtype, device=dev) * stride_era
        bin_stride = torch.as_tensor(num_bins, dtype=dtype, device=dev)

        base = (torch.arange(B, device=dev, dtype=dtype).view(1, B) * node_stride)
        key = base + node_ids.view(R, 1) * stride_era
        key = key + era_ids.view(R, 1) * bin_stride
        key = key + bins_block
        key_flat = key.reshape(-1)
        counts = torch.bincount(key_flat, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)

        gw = grad_rows.view(R, 1).expand(R, B).reshape(-1)
        hw = hess_rows.view(R, 1).expand(R, B).reshape(-1)
        grad_hist = torch.bincount(key_flat, weights=gw, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)
        hess_hist = torch.bincount(key_flat, weights=hw, minlength=hist_size).reshape(B, num_nodes, num_eras, num_bins)
        return counts, grad_hist, hess_hist

    def _compute_histograms(
        self,
        feature_values: torch.Tensor,
        grad_rows: torch.Tensor,
        hess_rows: torch.Tensor,
        era_ids: torch.Tensor,
        *,
        num_eras: int,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature_values.numel() == 0:
            empty = torch.zeros((num_eras, num_bins), dtype=torch.int64, device=feature_values.device)
            return empty, empty.to(torch.float32), empty.to(torch.float32)
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        key = era_ids.to(torch.int64) * int(num_bins) + feature_values.to(torch.int64)
        hist_size = int(num_eras * num_bins)
        counts = torch.bincount(key, minlength=hist_size).reshape(num_eras, num_bins)
        grad_hist = torch.bincount(key, weights=grad_rows, minlength=hist_size).reshape(num_eras, num_bins)
        if hess_rows is None:
            hess_hist = counts.to(torch.float32)
        else:
            hess_hist = torch.bincount(key, weights=hess_rows, minlength=hist_size).reshape(num_eras, num_bins)
        return counts, grad_hist, hess_hist

    # --- K-cut selection helpers (on-the-fly thermometer lanes) ---

    def _even_cut_indices(self, num_bins: int, k: int) -> torch.Tensor:
        k = max(1, min(k, max(1, num_bins - 1)))
        idx = torch.linspace(0, num_bins - 2, k, device=self._device)
        idx = torch.round(idx).to(torch.int64)
        return torch.unique(idx, sorted=True).clamp_(0, num_bins - 2)

    def _find_best_splits_batched_native(
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

        backend = self._cpu_backend
        if backend is None:
            raise RuntimeError("Native CPU backend is unavailable; reinstall with setup_native.py")

        min_child = int(self.config.min_samples_leaf)
        node_payload: list[list[np.ndarray]] = []
        index_map: list[int] = []
        duplicate_map: dict[int, int] = {}
        for idx, node in enumerate(nodes):
            era_arrays: list[np.ndarray] = []
            total = 0
            for rows in node.era_rows:
                if rows.numel() == 0:
                    era_arrays.append(np.empty(0, dtype=np.int64))
                    continue
                cpu_rows = rows.detach().to(device="cpu", dtype=torch.int64)
                arr = cpu_rows.contiguous().numpy().astype(np.int64, copy=True)
                era_arrays.append(arr)
                total += int(arr.shape[0])
            if total < 2 * min_child:
                stats.nodes_skipped += 1
                continue
            duplicate_idx: int | None = None
            for ctx_idx, existing in enumerate(node_payload):
                if len(existing) != len(era_arrays):
                    continue
                same = True
                for arr_a, arr_b in zip(existing, era_arrays):
                    if arr_a.shape != arr_b.shape or not np.array_equal(arr_a, arr_b):
                        same = False
                        break
                if same:
                    duplicate_idx = ctx_idx
                    break
            if duplicate_idx is not None:
                duplicate_map[idx] = duplicate_idx
                continue
            node_payload.append(era_arrays)
            index_map.append(idx)

        if not node_payload:
            return decisions, stats

        if duplicate_map:
            stats.nodes_collapsed += len(duplicate_map)

        bins_cpu = bins.detach().to(device="cpu")
        grad_cpu = grad.detach().to(device="cpu")
        feat_cpu = feature_subset.detach().to(device="cpu", dtype=torch.int32)

        bins_np = bins_cpu.contiguous().cpu().numpy().astype(np.int8, copy=False)
        grad_np = grad_cpu.contiguous().cpu().numpy().astype(np.float32, copy=False)
        feats_np = feat_cpu.contiguous().cpu().numpy().astype(np.int32, copy=False)

        decisions_raw, stats_dict = backend.find_best_splits_batched(
            bins_np,
            grad_np,
            node_payload,
            feats_np,
            int(self.config.max_bins),
            int(self.config.k_cuts),
            str(self.config.cut_selection),
            float(self.config.lambda_l2),
            float(self.config.lambda_dro),
            float(self.config.direction_weight),
            float(self.config.era_alpha),
            int(self.config.min_samples_leaf),
        )

        stats.nodes_processed += int(stats_dict.get("nodes_processed", 0))
        stats.nodes_skipped += int(stats_dict.get("nodes_skipped", 0))
        stats.rows_total += int(stats_dict.get("rows_total", 0))
        stats.feature_blocks += int(stats_dict.get("feature_blocks", 0))
        stats.bincount_calls += int(stats_dict.get("bincount_calls", 0))
        stats.hist_ms += float(stats_dict.get("hist_ms", 0.0))
        stats.scan_ms += float(stats_dict.get("scan_ms", 0.0))
        stats.score_ms += float(stats_dict.get("score_ms", 0.0))
        stats.partition_ms += float(stats_dict.get("partition_ms", 0.0))
        stats.nodes_subtract_ok += int(stats_dict.get("nodes_subtract_ok", 0))
        stats.nodes_subtract_fallback += int(stats_dict.get("nodes_subtract_fallback", 0))
        rebuild_backend = int(stats_dict.get("nodes_rebuild", 0))
        stats.block_size = max(stats.block_size, int(stats_dict.get("block_size", 0)))
        if self._histogram_mode == "rebuild":
            stats.nodes_rebuild += stats.nodes_processed
            stats.nodes_subtract_ok = 0
        else:
            stats.nodes_rebuild += rebuild_backend

        decisions_list = list(decisions_raw)
        for rel_idx, node_idx in enumerate(index_map):
            payload = decisions_list[rel_idx]
            if payload is None:
                continue
            feat = int(payload["feature"])
            thr = int(payload["threshold"])
            score = float(payload["score"])
            left_grad = float(payload["left_grad"])
            left_hess = float(payload["left_hess"])
            right_grad = float(payload["right_grad"])
            right_hess = float(payload["right_hess"])
            left_cnt = int(payload["left_count"])
            right_cnt = int(payload["right_count"])

            left_rows_np = [np.asarray(arr, dtype=np.int64).reshape(-1) for arr in payload["left_rows"]]
            right_rows_np = [np.asarray(arr, dtype=np.int64).reshape(-1) for arr in payload["right_rows"]]
            left_rows = [torch.from_numpy(arr.copy()).to(device=self._device) for arr in left_rows_np]
            right_rows = [torch.from_numpy(arr.copy()).to(device=self._device) for arr in right_rows_np]

            score_val = score
            need_des_score = (
                self._native_validate
                or self.config.lambda_dro != 0.0
                or self.config.direction_weight != 0.0
            )
            if self._native_validate and need_des_score:
                try:
                    node_ref = nodes[node_idx]
                    era_counts_tensor = torch.tensor(
                        [float(r.numel()) for r in node_ref.era_rows],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    era_weights_tensor = self._compute_era_weights(era_counts_tensor)
                    with torch.no_grad():
                        all_rows, era_ids = self._stack_node_rows(node_ref.era_rows)
                        if all_rows.numel() > 0 and 0 <= thr < int(self.config.max_bins) - 1:
                            grad_rows = grad.index_select(0, all_rows)
                            hess_rows = hess.index_select(0, all_rows)
                            feature_column = bins.index_select(0, all_rows)[:, feat]
                            counts_t, grad_hist_t, hess_hist_t = self._compute_histograms(
                                feature_column,
                                grad_rows,
                                hess_rows,
                                era_ids,
                                num_eras=len(node_ref.era_rows),
                                num_bins=int(self.config.max_bins),
                            )
                            if counts_t.shape[1] > thr:
                                prefix_counts = counts_t[:, :-1].cumsum(dim=1)
                                prefix_grad = grad_hist_t[:, :-1].cumsum(dim=1)
                                prefix_hess = hess_hist_t[:, :-1].cumsum(dim=1)
                                total_count_e = counts_t.sum(dim=1, keepdim=True)
                                total_grad_e = grad_hist_t.sum(dim=1, keepdim=True)
                                total_hess_e = hess_hist_t.sum(dim=1, keepdim=True)
                                left_count_e = prefix_counts[:, thr]
                                right_count_e = total_count_e[:, 0] - left_count_e
                                valid_mask = (left_count_e > 0) & (right_count_e > 0)
                                if bool(torch.any(valid_mask)):
                                    left_grad_e = prefix_grad[:, thr]
                                    right_grad_e = total_grad_e[:, 0] - left_grad_e
                                    left_hess_e = prefix_hess[:, thr]
                                    right_hess_e = total_hess_e[:, 0] - left_hess_e
                                    parent_gain_e = 0.5 * (total_grad_e[:, 0] ** 2) / (
                                        total_hess_e[:, 0] + self.config.lambda_l2
                                    )
                                    gain_e = 0.5 * (
                                        (left_grad_e**2) / (left_hess_e + self.config.lambda_l2)
                                        + (right_grad_e**2) / (right_hess_e + self.config.lambda_l2)
                                    ) - parent_gain_e
                                    left_val_e = -left_grad_e / (left_hess_e + self.config.lambda_l2)
                                    right_val_e = -right_grad_e / (right_hess_e + self.config.lambda_l2)
                                    mean_vec, std_vec, weight_vec = self._weighted_welford(
                                        gain_e.unsqueeze(1),
                                        valid_mask.unsqueeze(1),
                                        era_weights_tensor,
                                    )
                                    agreement_vec = self._directional_agreement(
                                        left_val_e.unsqueeze(1),
                                        right_val_e.unsqueeze(1),
                                        valid_mask.unsqueeze(1),
                                        era_weights_tensor,
                                        weight_vec,
                                    )
                                    score_py = mean_vec[0] - self.config.lambda_dro * std_vec[0]
                                    if self.config.direction_weight != 0.0:
                                        score_py = score_py + self.config.direction_weight * agreement_vec[0]
                                    left_total_global = left_count_e.sum()
                                    right_total_global = total_count_e[:, 0].sum() - left_total_global
                                    if (
                                        left_total_global >= self.config.min_samples_leaf
                                        and right_total_global >= self.config.min_samples_leaf
                                    ):
                                        score_val = float(score_py.item())
                except Exception:  # pragma: no cover - defensive recalculation fallback
                    pass
            elif need_des_score:
                try:
                    era_counts_tensor = torch.tensor(
                        [float(l.numel() + r.numel()) for l, r in zip(left_rows, right_rows)],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    if era_counts_tensor.numel() == 0:
                        raise ValueError("No era counts for native split validation")
                    era_weights_tensor = self._compute_era_weights(era_counts_tensor)
                    if not torch.any(era_weights_tensor > 0):
                        raise ValueError("No positive era weights for native split validation")

                    left_grad_e = torch.zeros_like(era_counts_tensor)
                    right_grad_e = torch.zeros_like(era_counts_tensor)
                    left_hess_e = torch.zeros_like(era_counts_tensor)
                    right_hess_e = torch.zeros_like(era_counts_tensor)
                    left_count_e = torch.zeros_like(era_counts_tensor)
                    right_count_e = torch.zeros_like(era_counts_tensor)

                    for e_idx, (rows_left, rows_right) in enumerate(zip(left_rows, right_rows)):
                        if rows_left.numel() > 0:
                            left_grad_e[e_idx] = grad.index_select(0, rows_left).sum()
                            left_hess_e[e_idx] = hess.index_select(0, rows_left).sum()
                            left_count_e[e_idx] = float(rows_left.numel())
                        if rows_right.numel() > 0:
                            right_grad_e[e_idx] = grad.index_select(0, rows_right).sum()
                            right_hess_e[e_idx] = hess.index_select(0, rows_right).sum()
                            right_count_e[e_idx] = float(rows_right.numel())

                    total_grad_e = left_grad_e + right_grad_e
                    total_hess_e = left_hess_e + right_hess_e
                    valid_mask = (left_count_e > 0) & (right_count_e > 0)
                    if not bool(torch.any(valid_mask)):
                        raise ValueError("Native split lacks valid children for DES recompute")

                    parent_gain_e = 0.5 * (total_grad_e**2) / (total_hess_e + self.config.lambda_l2)
                    gain_e = 0.5 * (
                        (left_grad_e**2) / (left_hess_e + self.config.lambda_l2)
                        + (right_grad_e**2) / (right_hess_e + self.config.lambda_l2)
                    ) - parent_gain_e
                    left_val_e = -left_grad_e / (left_hess_e + self.config.lambda_l2)
                    right_val_e = -right_grad_e / (right_hess_e + self.config.lambda_l2)

                    mean_vec, std_vec, weight_vec = self._weighted_welford(
                        gain_e.view(-1, 1),
                        valid_mask.view(-1, 1),
                        era_weights_tensor,
                    )
                    agreement_vec = self._directional_agreement(
                        left_val_e.view(-1, 1),
                        right_val_e.view(-1, 1),
                        valid_mask.view(-1, 1),
                        era_weights_tensor,
                        weight_vec,
                    )
                    score_py = mean_vec[0] - self.config.lambda_dro * std_vec[0]
                    if self.config.direction_weight != 0.0:
                        score_py = score_py + self.config.direction_weight * agreement_vec[0]
                    score_val = float(score_py.item())
                except Exception:  # pragma: no cover - defensive recalculation fallback
                    pass

            decisions[node_idx] = SplitDecision(
                feature=feat,
                threshold=thr,
                score=score_val,
                left_grad=left_grad,
                left_hess=left_hess,
                right_grad=right_grad,
                right_hess=right_hess,
                left_count=left_cnt,
                right_count=right_cnt,
                left_rows=left_rows,
                right_rows=right_rows,
            )

        if duplicate_map:
            for dup_idx, ctx_idx in duplicate_map.items():
                ref_node_idx = index_map[ctx_idx]
                decisions[dup_idx] = decisions[ref_node_idx]

        return decisions, stats

    def _find_best_splits_batched_cuda(
        self, nodes, grad, hess, bins, feature_subset,
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

        # ----- build compact contexts -----
        contexts: list[_NodeContext] = []
        index_map: list[int] = []
        row_chunks: list[torch.Tensor] = []
        grad_chunks: list[torch.Tensor] = []
        hess_chunks: list[torch.Tensor] = []
        context_rows: list[torch.Tensor] = []
        context_era_ids: list[torch.Tensor] = []
        duplicate_map: dict[int, int] = {}
        row_start = 0

        for idx, node in enumerate(nodes):
            all_rows, era_ids = self._stack_node_rows(node.era_rows)
            if all_rows.device != self._device: all_rows = all_rows.to(self._device)
            if era_ids.device  != self._device: era_ids  = era_ids.to(self._device)
            all_rows = all_rows.to(torch.int64).contiguous()
            era_ids  = era_ids.to(torch.int16).contiguous()
            total_count = int(all_rows.numel())
            if total_count < 2 * self.config.min_samples_leaf:
                stats.nodes_skipped += 1
                continue

            dup: int | None = None
            for j, rows_ref in enumerate(context_rows):
                if rows_ref.shape[0] != all_rows.shape[0]: continue
                if not torch.equal(rows_ref, all_rows):    continue
                if not torch.equal(context_era_ids[j], era_ids): continue
                dup = j; break
            if dup is not None:
                duplicate_map[idx] = dup
                continue

            g_rows = grad.index_select(0, all_rows).contiguous()
            h_rows = hess.index_select(0, all_rows).contiguous()
            g_tot = float(g_rows.sum().item())
            h_tot = float(h_rows.sum().item())
            era_w = self._compute_era_weights(torch.tensor(
                [float(r.numel()) for r in node.era_rows],
                dtype=torch.float32, device=self._device,
            ))

            contexts.append(_NodeContext(
                shard=node, era_weights=era_w, parent_dir=torch.tensor(0.0, device=self._device),
                total_grad=g_tot, total_hess=h_tot, total_count=total_count,
                row_start=row_start, row_count=total_count,
            ))
            index_map.append(idx)
            row_chunks.append(all_rows)
            grad_chunks.append(g_rows)
            hess_chunks.append(h_rows)
            context_rows.append(all_rows)
            context_era_ids.append(era_ids)
            row_start += total_count
            stats.rows_total += total_count

        if not contexts:
            return decisions, stats
        if duplicate_map:
            stats.nodes_collapsed += len(duplicate_map)

        stats.nodes_processed += len(contexts)
        Nnodes = len(contexts)
        expected_eras = max(1, max(len(c.shard.era_rows) for c in contexts))
        stats.block_size = max(stats.block_size, int(feat_ids.numel()))

        rows_cat      = torch.cat(row_chunks, 0).to(self._device)
        rows_cat_i32  = rows_cat.to(torch.int32).contiguous()
        grad_cat      = torch.cat(grad_chunks, 0).contiguous()
        hess_cat      = torch.cat(hess_chunks, 0).contiguous()
        node_offsets  = [c.row_start for c in contexts] + [row_start]
        node_row_splits = torch.tensor(node_offsets, dtype=torch.int32, device=self._device)

        # ---- RELATIVE era offsets (0..span) ----
        era_offset_rows: list[list[int]] = []
        for ctx in contexts:
            pref: list[int] = [0]
            run = 0
            for rows_tensor in ctx.shard.era_rows:
                run += int(rows_tensor.numel())
                pref.append(run)
            while len(pref) < expected_eras + 1:
                pref.append(run)
            era_offset_rows.append(pref)   # *** RELATIVE; no +row_start ***
        node_era_splits = torch.tensor(era_offset_rows, dtype=torch.int32, device=self._device).contiguous()

        spans = node_row_splits[1:] - node_row_splits[:-1]
        if (node_era_splits[:, 0] != 0).any() or (node_era_splits[:, -1] != spans).any():
            raise RuntimeError("node_era_splits must be relative (0..span) for each node.")
        if (node_era_splits[:, 1:] - node_era_splits[:, :-1] < 0).any():
            raise RuntimeError("node_era_splits must be non-decreasing per node.")

        # feature-major int8 bins for frontier kernels
        bins_fm_i8 = self._bins_fm_i8 if self._bins_fm_i8 is not None else bins.t().contiguous()
        # quick sanity
        if bins_fm_i8.dtype != torch.int8 or not bins_fm_i8.is_contiguous():
            bins_fm_i8 = bins_fm_i8.to(torch.int8).contiguous()
        # ---- split scoring kernel ----
        result = self._cuda_backend.find_best_splits_batched_cuda(
            bins_fm_i8, grad_cat, hess_cat,
            rows_cat_i32,
            node_row_splits, node_era_splits,
            torch.stack([c.era_weights for c in contexts], 0).contiguous(),
            torch.tensor([c.total_grad for c in contexts], dtype=torch.float32, device=self._device),
            torch.tensor([c.total_hess for c in contexts], dtype=torch.float32, device=self._device),
            torch.tensor([c.total_count for c in contexts], dtype=torch.int64,   device=self._device),
            feat_ids,
            int(self.config.max_bins),
            int(self.config.k_cuts),
            str(self.config.cut_selection),
            lam_l2,
            float(self.config.lambda_dro),
            float(self.config.direction_weight),
            int(self.config.min_samples_leaf),
            int(rows_cat_i32.shape[0]),
        )
        stats.score_ms += float(result.get("kernel_ms", 0.0))

        scores      = result["scores"].to("cpu").contiguous().numpy()
        thresholds  = result["thresholds"].to("cpu").contiguous().numpy()
        left_grad   = result["left_grad"].to("cpu").contiguous().numpy()
        left_hess   = result["left_hess"].to("cpu").contiguous().numpy()
        left_count  = result["left_count"].to("cpu").contiguous().numpy()
        feat_list   = feat_ids_i64.to("cpu").tolist()
        tot_g_list  = [c.total_grad for c in contexts]
        tot_h_list  = [c.total_hess for c in contexts]
        tot_c_list  = [c.total_count for c in contexts]

        best_feat = torch.full((Nnodes,), -1, dtype=torch.int32, device=self._device)
        best_thr  = torch.full((Nnodes,), -1, dtype=torch.int32, device=self._device)
        best_scr  = torch.full((Nnodes,), float("-inf"), device=self._device)
        best_lg   = torch.zeros(Nnodes, dtype=torch.float32, device=self._device)
        best_lh   = torch.zeros(Nnodes, dtype=torch.float32, device=self._device)
        best_lc   = torch.full((Nnodes,), -1, dtype=torch.int64, device=self._device)

        for cid in range(Nnodes):
            sc  = scores[cid]; th = thresholds[cid]
            lg  = left_grad[cid]; lh = left_hess[cid]; lc = left_count[cid]
            best = float("-inf"); bf=-1; bt=-1; blg=0.0; blh=0.0; blc=-1
            tot_cnt = int(tot_c_list[cid])

            for j, fglob in enumerate(feat_list):
                sv = float(sc[j]); tv = int(th[j]); if_bad = (tv < 0) or (not math.isfinite(sv))
                if if_bad: continue
                lcj = int(lc[j]); rcj = tot_cnt - lcj
                if lcj < self.config.min_samples_leaf or rcj < self.config.min_samples_leaf:
                    continue
                better = (sv > best) or (abs(sv-best) <= 1e-12 and (lcj > blc or (lcj==blc and (fglob>bf or (fglob==bf and tv>bt)))))
                if better:
                    best, bf, bt, blg, blh, blc = sv, int(fglob), tv, float(lg[j]), float(lh[j]), lcj

            best_scr[cid]=best; best_feat[cid]=bf; best_thr[cid]=bt; best_lg[cid]=blg; best_lh[cid]=blh; best_lc[cid]=blc

        sel_mask = best_feat >= 0
        if torch.any(sel_mask):
            sel_idx = torch.nonzero(sel_mask, as_tuple=False).view(-1)
            # Compact splits for selected nodes
            node_row_splits_sel = torch.tensor(
                [contexts[i].row_start for i in sel_idx.tolist()] + [row_start],
                dtype=torch.int32, device=self._device
            )
            node_era_splits_sel = node_era_splits.index_select(0, sel_idx).contiguous()
            feat_sel = best_feat.index_select(0, sel_idx).to(torch.int32).contiguous()
            thr_sel  = best_thr.index_select(0, sel_idx).to(torch.int32).contiguous()

            # Try backend partition; if the shim wants a different type and raises
            # an AttributeError (e.g. '.slice'), fall back to Python partition.
            used_cuda_partition = False
            try:
                part = self._cuda_backend.partition_frontier_cuda(
                    (self._bins_fm_i8 if self._bins_fm_i8 is not None else bins.t().contiguous()),
                    rows_cat_i32,
                    node_row_splits_sel,
                    node_era_splits_sel,
                    feat_sel,
                    thr_sel,
                )
                used_cuda_partition = True
            except (AttributeError, TypeError) as _shim_issue:
                used_cuda_partition = False

            if used_cuda_partition:
                part_start = perf_counter()
                left_idx:  torch.Tensor = part["left_index"]
                right_idx: torch.Tensor = part["right_index"]
                left_spl:  torch.Tensor = part["left_splits"].to(torch.int64).contiguous()
                right_spl: torch.Tensor = part["right_splits"].to(torch.int64).contiguous()
                stats.partition_ms += (perf_counter() - part_start) * 1000.0

                expected_eras = node_era_splits_sel.shape[1] - 1
                for j, cid in enumerate(sel_idx.tolist()):
                    node_idx = index_map[cid]
                    f = int(best_feat[cid].item()); t = int(best_thr[cid].item())
                    s = float(best_scr[cid].item())
                    Ls: list[torch.Tensor] = []; Rs: list[torch.Tensor] = []
                    for e in range(expected_eras):
                        lb, le = int(left_spl[j, e].item()),  int(left_spl[j, e+1].item())
                        rb, re = int(right_spl[j, e].item()), int(right_spl[j, e+1].item())
                        Ls.append(left_idx[lb:le].to(torch.int64, device=self._device))
                        Rs.append(right_idx[rb:re].to(torch.int64, device=self._device))

                    l_cnt = int((left_spl[j, -1] - left_spl[j, 0]).item())
                    r_cnt = int((right_spl[j, -1] - right_spl[j, 0]).item())
                    if l_cnt > 0:
                        l_flat = left_idx[left_spl[j, 0]: left_spl[j, -1]].to(torch.int64)
                        l_g = float(grad.index_select(0, l_flat).sum().item())
                        l_h = float(hess.index_select(0, l_flat).sum().item())
                    else:
                        l_g = 0.0; l_h = 0.0
                    r_g = float(tot_g_list[cid] - l_g)
                    r_h = float(tot_h_list[cid] - l_h)

                    decisions[node_idx] = SplitDecision(
                        feature=f, threshold=t, score=s,
                        left_grad=l_g, left_hess=l_h, right_grad=r_g, right_hess=r_h,
                        left_count=l_cnt, right_count=r_cnt,
                        left_rows=Ls, right_rows=Rs,
                    )

            else:
                # Fallback: safe Python partition on *row-major* bins
                for cid in sel_idx.tolist():
                    node_idx = index_map[cid]
                    f = int(best_feat[cid].item()); t = int(best_thr[cid].item())
                    s = float(best_scr[cid].item())
                    Ls, Rs = self._partition_rows(bins, contexts[cid].shard.era_rows, f, t)  # row-major!
                    l_cnt = sum(int(x.numel()) for x in Ls)
                    r_cnt = sum(int(x.numel()) for x in Rs)
                    l_g = float(grad.index_select(0, torch.cat([x for x in Ls if x.numel()>0], 0)).sum().item()) if l_cnt>0 else 0.0
                    l_h = float(hess.index_select(0, torch.cat([x for x in Ls if x.numel()>0], 0)).sum().item()) if l_cnt>0 else 0.0
                    r_g = float(tot_g_list[cid] - l_g)
                    r_h = float(tot_h_list[cid] - l_h)
                    decisions[node_idx] = SplitDecision(
                        feature=f, threshold=t, score=s,
                        left_grad=l_g, left_hess=l_h, right_grad=r_g, right_hess=r_h,
                        left_count=l_cnt, right_count=r_cnt,
                        left_rows=Ls, right_rows=Rs,
                    )

        # propagate decisions to duplicates
        if duplicate_map:
            for dup_idx, ctx_idx in duplicate_map.items():
                ref = index_map[ctx_idx]
                decisions[dup_idx] = decisions[ref]

        stats.feature_blocks += 1
        stats.nodes_subtract_ok += len(contexts)
        return decisions, stats



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
        if self._use_native_cpu:
            return self._find_best_splits_batched_native(nodes, grad, hess, bins, feature_subset)
        if self._use_native_cuda:
            try:
                return self._find_best_splits_batched_cuda(nodes, grad, hess, bins, feature_subset)
            except RuntimeError as exc:
                msg = str(exc)
                if "CUDA error" in msg:
                    self._logger.warning(
                        "CUDA frontier failed (%s); falling back to Torch implementation.",
                        msg,
                    )
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                    self._use_native_cuda = False
                    self._cuda_backend = None
                else:
                    raise

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
        context_rows: list[torch.Tensor] = []
        context_era_ids: list[torch.Tensor] = []
        duplicate_map: dict[int, int] = {}
        row_start = 0

        for idx, node in enumerate(nodes):
            all_rows, era_ids = self._stack_node_rows(node.era_rows)
            total_count = int(all_rows.numel())
            if total_count < 2 * self.config.min_samples_leaf:
                stats.nodes_skipped += 1
                continue

            duplicate_idx: int | None = None
            for ctx_idx, rows_ref in enumerate(context_rows):
                if rows_ref.shape[0] != total_count:
                    continue
                if not torch.equal(rows_ref, all_rows):
                    continue
                if not torch.equal(context_era_ids[ctx_idx], era_ids):
                    continue
                duplicate_idx = ctx_idx
                break
            if duplicate_idx is not None:
                duplicate_map[idx] = duplicate_idx
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
            node_chunks.append(
                torch.full((total_count,), cid, dtype=torch.int32, device=self._device)
            )
            context_rows.append(all_rows)
            context_era_ids.append(era_ids)
            row_start += total_count
            stats.rows_total += total_count

        if not contexts:
            return decisions, stats

        if duplicate_map:
            stats.nodes_collapsed += len(duplicate_map)

        stats.nodes_processed += len(contexts)
        Nnodes = len(contexts)
        Eras = len(contexts[0].shard.era_rows)
        block_size = self._resolve_feature_block_size(int(feat_ids.numel()))
        stats.block_size = max(stats.block_size, block_size)

        rows_cat = torch.cat(row_chunks, 0)
        era_cat = torch.cat(era_chunks, 0)
        grad_cat = torch.cat(grad_chunks, 0)
        hess_cat = torch.cat(hess_chunks, 0)  # kept for API parity; not used in hist if H≡count
        node_cat = torch.cat(node_chunks, 0)
        bins_cat = bins.index_select(0, rows_cat)

        era_weights_tensor = torch.stack([c.era_weights for c in contexts], 0).contiguous()  # [Nnodes, E]
        total_grad_nodes = torch.tensor([c.total_grad for c in contexts], dtype=torch.float32, device=self._device)  # [Nnodes]
        total_hess_nodes = torch.tensor([c.total_hess for c in contexts], dtype=torch.float32, device=self._device)
        total_count_nodes = torch.tensor([c.total_count for c in contexts], dtype=torch.int64, device=self._device)
        parent_dirs = torch.stack([c.parent_dir.to(torch.float32) for c in contexts], 0)  # [Nnodes]

        # Best trackers
        best_scores   = torch.full((Nnodes,), float("-inf"), device=self._device)
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
            block = feat_ids[start : start + block_size]          # int32
            block_i64 = feat_ids_i64[start : start + block_size]  # int64
            if block.numel() == 0:
                continue

            # Histograms for this block: counts + gradient only (H ≡ count)
            t0 = perf_counter()
            counts, g_hist, h_hist = self._batched_histograms_nodes(
                bins_cat.index_select(1, block_i64),
                grad_cat, hess_cat,  # hess_cat may be ignored inside
                era_cat, node_cat,
                Nnodes, Eras, num_bins,
            )
            stats.hist_ms += (perf_counter() - t0) * 1000.0
            stats.feature_blocks += 1
            stats.bincount_calls += 3  # counts + grad + hess
            if counts.numel() == 0:
                continue

            # Prefix scans along bins (≤ t)
            scan_start = perf_counter()
            left_c_full = counts[:, :, :, :-1].cumsum(3)   # [B,N,E,T]
            left_g_full = g_hist[:, :, :, :-1].cumsum(3)   # [B,N,E,T]
            left_h_full = h_hist[:, :, :, :-1].cumsum(3)   # [B,N,E,T]
            # totals per era
            tot_c_e = counts.sum(3, keepdim=True)          # [B,N,E,1]
            tot_g_e = g_hist.sum(3, keepdim=True)          # [B,N,E,1]
            tot_h_e = h_hist.sum(3, keepdim=True)          # [B,N,E,1]
            stats.scan_ms += (perf_counter() - scan_start) * 1000.0

            # K-cut selection
            K_user = int(self.config.k_cuts)
            use_K = (K_user > 0 and K_user < full_T)
            if use_K and self.config.cut_selection == "mass":
                cut_idx_bk_long = self._mass_cut_indices(counts, K_user)  # [B,K] int64
                thr_k_long = cut_idx_bk_long  # per-feature
            else:
                if use_K:
                    thr_k_long = self._even_cut_indices(num_bins, K_user).to(torch.int64)  # [K]
                else:
                    thr_k_long = torch.arange(full_T, device=self._device, dtype=torch.int64)  # full sweep
                cut_idx_bk_long = None  # shared thresholds
            K = int(thr_k_long.shape[-1])

            # --------- DES scoring with WEIGHTED WELFORD (streaming over eras) ---------
            Bblk, Nn, Ee, _T = left_c_full.shape
            wsum = torch.zeros((Bblk, Nn, K), dtype=torch.float32, device=self._device)
            mean = torch.zeros_like(wsum)
            M2   = torch.zeros_like(wsum)
            left_count_total = torch.zeros((Bblk, Nn, K), dtype=counts.dtype, device=self._device)

            use_dir = (dir_w != 0.0) and (parent_dirs != 0).any()
            if use_dir:
                agr_num   = torch.zeros_like(wsum)
                agr_denom = torch.zeros_like(wsum)
                pa = parent_dirs.to(torch.float32).view(1, Nnodes, 1).expand(Bblk, Nnodes, K)

            # Helper to gather K thresholds from the T-axis for one era slice
            def gather_K(tensor_BNT: torch.Tensor) -> torch.Tensor:
                # tensor_BNT: [B,N,T]
                if cut_idx_bk_long is not None:  # per-feature thresholds
                    idx = cut_idx_bk_long.unsqueeze(1).expand(Bblk, Nn, K)  # [B,N,K]
                else:  # shared thresholds
                    idx = thr_k_long.view(1, 1, K).expand(Bblk, Nn, K)      # [B,N,K]
                return tensor_BNT.gather(2, idx)  # [B,N,K]

            # Stream eras
            score_start = perf_counter()
            for e in range(Eras):
                # weights per node for this era → broadcast to [B,N,K]
                w_e = era_weights_tensor[:, e]  # [N]
                if not torch.any(w_e > 0):
                    continue
                w_e_full = w_e.view(1, Nn, 1).expand(Bblk, Nn, K).to(torch.float32)

                # left stats at K thresholds for era e
                lc_e = gather_K(left_c_full[:, :, e, :])              # [B,N,K] (int)
                lg_e = gather_K(left_g_full[:, :, e, :].to(torch.float32))  # [B,N,K] (float)
                lh_e = gather_K(left_h_full[:, :, e, :].to(torch.float32))  # [B,N,K] (float)
                # totals per era (broadcast to K)
                tc_e = tot_c_e[:, :, e, 0].unsqueeze(-1).expand(Bblk, Nn, K)  # int
                tg_e = tot_g_e[:, :, e, 0].unsqueeze(-1).expand(Bblk, Nn, K)  # float
                th_e = tot_h_e[:, :, e, 0].unsqueeze(-1).expand(Bblk, Nn, K)  # float

                rc_e = tc_e - lc_e
                rg_e = tg_e - lg_e
                rh_e = th_e - lh_e

                valid_e = (lc_e > 0) & (rc_e > 0)
                w_e_full = torch.where(
                    valid_e,
                    w_e_full,
                    torch.zeros((), dtype=w_e_full.dtype, device=w_e_full.device),
                )

                # gains per era at K cuts (subtract parent)
                denom_L = lh_e + lam_l2
                denom_R = rh_e + lam_l2
                parent_gain_e = 0.5 * (tg_e * tg_e) / (th_e + lam_l2)
                gain_e = 0.5 * ((lg_e * lg_e) / denom_L + (rg_e * rg_e) / denom_R) - parent_gain_e

                # Welford update
                wsum_new = wsum + w_e_full
                factor = torch.zeros_like(wsum_new)
                nz = wsum_new > 0
                factor[nz] = w_e_full[nz] / wsum_new[nz]

                delta = gain_e - mean
                mean = mean + factor * delta
                M2 = M2 + w_e_full * delta * (gain_e - mean)
                wsum = wsum_new

                # directional agreement (optional)
                if use_dir:
                    lg_dir = -lg_e / denom_L  # leaf directions
                    rg_dir = -rg_e / denom_R
                    agree_e = 0.5 * (
                        (torch.sign(lg_dir) == pa).float() + (torch.sign(rg_dir) == pa).float()
                    )
                    agr_num = agr_num + w_e_full * agree_e
                    agr_denom = agr_denom + w_e_full

                # global left counts aggregation for min_samples_leaf
                left_count_total = left_count_total + lc_e

            # Final DES score across eras
            ws = torch.clamp_min(wsum, 1e-12)
            std = torch.sqrt(torch.clamp_min(M2 / ws, 0.0))
            score = mean - lam_dro * std  # [B,N,K]
            if use_dir:
                nz = agr_denom > 0
                agr = torch.zeros_like(agr_num)
                agr[nz] = agr_num[nz] / agr_denom[nz]
                score = score + dir_w * agr
            stats.score_ms += (perf_counter() - score_start) * 1000.0
            # ---------------------------------------------------------------------

            # Global min_samples_leaf (aggregate across eras)
            right_total = total_count_nodes.view(1, Nnodes, 1) - left_count_total  # [B,N,K]
            min_child = int(self.config.min_samples_leaf)
            valid_global = (left_count_total >= min_child) & (right_total >= min_child)

            # Prepare flattened selection per node: M = B * K candidates
            score_perm = score.permute(1, 0, 2)  # [N,B,K]
            score_perm = torch.where(
                valid_global.permute(1, 0, 2),
                score_perm,
                torch.full_like(score_perm, float("-inf"))
            )
            score_flat = score_perm.reshape(Nnodes, -1)  # [N, M]
            left_total_flat = left_count_total.permute(1, 0, 2).reshape(Nnodes, -1)

            best_block = score_flat.max(dim=1).values  # [N]
            m1 = score_flat == best_block.unsqueeze(1)
            NEG = torch.iinfo(torch.int64).min
            best_cnt = torch.where(m1, left_total_flat, torch.full_like(left_total_flat, NEG)).max(dim=1).values
            m2 = m1 & (left_total_flat == best_cnt.unsqueeze(1))
            Mflat = score_flat.shape[1]
            flat_idx = torch.arange(Mflat, device=self._device, dtype=torch.int64).view(1, -1)
            sel_idx = torch.where(m2, flat_idx, torch.full_like(flat_idx, -1)).max(dim=1).values  # [N], -1 invalid
            valid_mask = sel_idx >= 0
            if not torch.any(valid_mask):
                continue

            # Gather candidate stats for nodes with valid selection
            gather_idx = sel_idx.clamp_min(0).view(-1, 1)
            cand_score = score_flat.gather(1, gather_idx).squeeze(1)
            cand_left_total = left_total_flat.gather(1, gather_idx).squeeze(1)

            # Unravel (feature_in_block, k_in_block) within this block
            num_thresh = int(K)  # number of K-cuts considered in this block
            f_in_blk = (sel_idx // num_thresh).to(torch.int64)   # [N]
            t_in_blk = (sel_idx % num_thresh).to(torch.int64)   # [N]

            # Map to actual feature id
            cand_feature = block_i64.gather(0, f_in_blk.clamp_(0, block_i64.numel() - 1)).to(torch.int32)

            # Map to threshold (bin edge index) for output
            if cut_idx_bk_long is not None:
                # per-feature threshold table [B,K]
                cand_threshold = cut_idx_bk_long.gather(0, f_in_blk.clamp_(0, cut_idx_bk_long.shape[0] - 1)).gather(
                    1, t_in_blk.clamp_(0, num_thresh - 1).view(-1, 1)
                ).squeeze(1).to(torch.int32)
            else:
                cand_threshold = thr_k_long.gather(0, t_in_blk.clamp_(0, num_thresh - 1)).to(torch.int32)

            # Left totals for leaf values (H ≡ count), and left grad totals across eras
            # Compute left_g_tot_all = sum over eras of left_g_full (B,N,T) then gather to K
            left_g_tot_all = left_g_full.sum(dim=2)  # [B,N,T]
            left_h_tot_all = left_h_full.sum(dim=2)  # [B,N,T]
            if cut_idx_bk_long is not None:
                idx_BNK = cut_idx_bk_long.unsqueeze(1).expand(Bblk, Nn, K)
            else:
                idx_BNK = thr_k_long.view(1, 1, K).expand(Bblk, Nn, K)
            left_g_tot_BNK = left_g_tot_all.gather(2, idx_BNK)  # [B,N,K]
            left_g_tot_flat = left_g_tot_BNK.permute(1, 0, 2).reshape(Nnodes, -1)
            left_g_tot = left_g_tot_flat.gather(1, gather_idx).squeeze(1)

            left_h_tot_BNK = left_h_tot_all.gather(2, idx_BNK)
            left_h_tot_flat = left_h_tot_BNK.permute(1, 0, 2).reshape(Nnodes, -1)
            left_h_tot = left_h_tot_flat.gather(1, gather_idx).squeeze(1).to(torch.float32)

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

        # Accounting
        if self._histogram_mode == "rebuild":
            rebuild_cnt = Nnodes
        else:
            rebuild_cnt = int((~subtract_success).sum().item())
        stats.nodes_subtract_ok += int(subtract_success.sum().item())
        stats.nodes_rebuild += rebuild_cnt
        stats.nodes_subtract_fallback += int(fallback_any.sum().item())

        # ---- Batched partition via CUDA (count+scatter) ----
        try:
            sel_mask = best_features >= 0
            if torch.any(sel_mask):
                sel_idx = torch.nonzero(sel_mask, as_tuple=False).view(-1)

                # Build compact metadata for selected nodes (Nsel)
                node_row_offsets: list[int] = [contexts[i.item()].row_start for i in sel_idx.tolist()]
                # Sentinel can be the global end; kernel doesn't use it. Kept for shape.
                node_row_offsets.append(row_start)
                node_row_splits_sel = torch.tensor(node_row_offsets, dtype=torch.int32, device=self._device)

                expected_eras = max(1, max(len(contexts[i.item()].shard.era_rows) for i in sel_idx.tolist()))
                era_offset_rows_sel: list[list[int]] = []
                for i in sel_idx.tolist():
                    ctx = contexts[i]
                    pref: list[int] = [0]
                    run = 0
                    for rows_t in ctx.shard.era_rows:
                        run += int(rows_t.numel()); pref.append(run)
                    while len(pref) < expected_eras + 1:
                        pref.append(run)
                    # add base offset into rows_cat
                    pref = [p + ctx.row_start for p in pref]
                    era_offset_rows_sel.append(pref)
                node_era_splits_sel = torch.tensor(era_offset_rows_sel, dtype=torch.int32, device=self._device).contiguous()

                feat_sel = best_features.index_select(0, sel_idx).to(torch.int32).contiguous()
                thr_sel = best_thresholds.index_select(0, sel_idx).to(torch.int32).contiguous()

                # Inputs for kernel
                rows_cat_i32 = rows_cat.to(torch.int32).contiguous()
                bins_i8 = (bins_fm if 'bins_fm' in locals() and bins_fm is not None else bins).contiguous()

                part_start = perf_counter()
                part = self._cuda_backend.partition_frontier_cuda(
                    bins_i8,
                    rows_cat_i32,
                    node_row_splits_sel,
                    node_era_splits_sel,
                    feat_sel,
                    thr_sel,
                )
                stats.partition_ms += (perf_counter() - part_start) * 1000.0

                left_idx: torch.Tensor = part["left_index"]  # int32 flat, dataset row ids
                right_idx: torch.Tensor = part["right_index"]
                left_spl: torch.Tensor = part["left_splits"].to(torch.int64).contiguous()   # [Nsel, E+1]
                right_spl: torch.Tensor = part["right_splits"].to(torch.int64).contiguous()

                # Sanity: lengths
                assert left_spl.shape[0] == sel_idx.numel()
                assert right_spl.shape[0] == sel_idx.numel()

                # Build SplitDecision for selected nodes
                for j, cid in enumerate(sel_idx.tolist()):
                    node_idx = index_map[cid]
                    feat = int(best_features[cid].item())
                    thr = int(best_thresholds[cid].item())
                    score_val = float(best_scores[cid].item())

                    # reconstruct per-era slices (GPU tensors of dataset row ids, int64)
                    Ls: list[torch.Tensor] = []
                    Rs: list[torch.Tensor] = []
                    for e in range(expected_eras):
                        lb, le = int(left_spl[j, e].item()), int(left_spl[j, e + 1].item())
                        rb, re = int(right_spl[j, e].item()), int(right_spl[j, e + 1].item())
                        Ls.append(left_idx[lb:le].to(dtype=torch.int64, device=self._device))
                        Rs.append(right_idx[rb:re].to(dtype=torch.int64, device=self._device))

                    l_cnt = int((left_spl[j, -1] - left_spl[j, 0]).item())
                    r_cnt = int((right_spl[j, -1] - right_spl[j, 0]).item())

                    # Conservation: left + right equals node span
                    node_span = int(contexts[cid].row_count)
                    if (l_cnt + r_cnt) != node_span:
                        raise RuntimeError("partition_frontier_cuda count mismatch for node")

                    # sums from full grad/hess (length N): index by dataset row ids
                    if l_cnt > 0:
                        l_flat = left_idx[left_spl[j, 0]: left_spl[j, -1]].to(torch.int64)
                        l_g = float(grad.index_select(0, l_flat).sum().item())
                        l_h = float(hess.index_select(0, l_flat).sum().item())
                    else:
                        l_g = 0.0; l_h = 0.0

                    r_g = float(total_grad_nodes[cid].item() - l_g)
                    r_h = float(total_hess_nodes[cid].item() - l_h)

                    decisions[node_idx] = SplitDecision(
                        feature=feat, threshold=thr, score=score_val,
                        left_grad=l_g, left_hess=l_h, right_grad=r_g, right_hess=r_h,
                        left_count=l_cnt, right_count=r_cnt,
                        left_rows=Ls, right_rows=Rs,
                    )

            # Fallback for nodes without valid selection
            for cid, ctx in enumerate(contexts):
                if int(best_features[cid].item()) < 0:
                    node_idx = index_map[cid]
                    decisions[node_idx] = None
        except Exception:
            # Fallback: per-node Python partition
            for cid, ctx in enumerate(contexts):
                node_idx = index_map[cid]
                feat = int(best_features[cid].item())
                if feat < 0:
                    decisions[node_idx] = None
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
                    feature=feat,
                    threshold=thr,
                    score=score_val,
                    left_grad=l_g,
                    left_hess=l_h,
                    right_grad=r_g,
                    right_hess=r_h,
                    left_count=l_cnt,
                    right_count=r_cnt,
                    left_rows=left_rows,
                    right_rows=right_rows,
                )

        if duplicate_map:
            for dup_idx, ctx_idx in duplicate_map.items():
                rep_node_idx = index_map[ctx_idx]
                decisions[dup_idx] = decisions[rep_node_idx]

        return decisions, stats


    # --- misc helpers ---

    def _find_best_split(
        self,
        node: NodeShard,
        grad: torch.Tensor,
        hess: torch.Tensor,
        bins: torch.Tensor,
        feature_subset: torch.Tensor,
    ) -> SplitDecision | None:
        decisions, _ = self._find_best_splits_batched([node], grad, hess, bins, feature_subset)
        return decisions[0]

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
            echunks.append(
                torch.full((rows.numel(),), e_idx, dtype=torch.int16, device=rows.device)
            )
        if not rchunks:
            empty_rows = torch.empty(0, dtype=torch.int64, device=self._device)
            empty_eras = torch.empty(0, dtype=torch.int16, device=self._device)
            return empty_rows, empty_eras
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
