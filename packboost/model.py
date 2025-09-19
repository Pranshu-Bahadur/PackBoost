"""Model structures and inference utilities for PackBoost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import PackBoostConfig
from .backends import cuda_predict_forest


@dataclass
class FlattenedForest:
    """Contiguous representation of the ensemble for fast traversal."""

    tree_offsets: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    lefts: np.ndarray
    rights: np.ndarray
    is_leaf: np.ndarray
    values: np.ndarray


@dataclass
class TreeNode:
    """Represents a single node in a regression tree."""

    is_leaf: bool
    prediction: float
    feature: Optional[int] = None
    threshold: Optional[int] = None
    left: Optional[int] = None
    right: Optional[int] = None
    depth: int = 0


@dataclass
class Tree:
    """Regression tree stored in breadth-first order."""

    nodes: List[TreeNode] = field(default_factory=list)

    def add_node(self, node: TreeNode) -> int:
        """Append ``node`` and return its index."""
        self.nodes.append(node)
        return len(self.nodes) - 1

    def ensure_leaf(self, node_id: int, prediction: float) -> None:
        """Mark ``node_id`` as a leaf with ``prediction``."""
        node = self.nodes[node_id]
        node.is_leaf = True
        node.prediction = prediction
        node.feature = None
        node.threshold = None
        node.left = None
        node.right = None

    def predict_binned(self, X_binned: np.ndarray) -> np.ndarray:
        """Return leaf predictions for ``X_binned`` rows."""
        n_samples = X_binned.shape[0]
        contribution = np.zeros(n_samples, dtype=np.float32)
        stack: List[tuple[int, np.ndarray]] = [(0, np.arange(n_samples, dtype=np.int32))]
        while stack:
            node_id, indices = stack.pop()
            node = self.nodes[node_id]
            if (
                node.is_leaf
                or node.feature is None
                or node.threshold is None
                or node.left is None
                or node.right is None
            ):
                contribution[indices] = node.prediction
                continue
            feature = node.feature
            threshold = node.threshold
            bins = X_binned[indices, feature]
            left_mask = bins <= threshold
            right_mask = ~left_mask
            if np.any(left_mask):
                stack.append((node.left, indices[left_mask]))
            if np.any(right_mask):
                stack.append((node.right, indices[right_mask]))
        return contribution


@dataclass
class PackBoostModel:
    """Trained PackBoost ensemble."""

    config: PackBoostConfig
    bin_edges: Optional[np.ndarray]
    initial_prediction: float
    trees: List[Tree]
    _flattened: Optional[FlattenedForest] = field(default=None, init=False, repr=False)

    def to_dict(self) -> Dict[str, object]:
        """Serialise the model to a dictionary."""
        return {
            "config": self.config.__dict__,
            "bin_edges": self.bin_edges.tolist() if self.bin_edges is not None else None,
            "initial_prediction": self.initial_prediction,
            "trees": [
                [
                    {
                        "is_leaf": node.is_leaf,
                        "prediction": node.prediction,
                        "feature": node.feature,
                        "threshold": node.threshold,
                        "left": node.left,
                        "right": node.right,
                        "depth": node.depth,
                    }
                    for node in tree.nodes
                ]
                for tree in self.trees
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PackBoostModel":
        """Create a model from ``payload`` produced by :meth:`to_dict`."""
        config = PackBoostConfig(**payload["config"])  # type: ignore[arg-type]
        bin_edges_payload = payload["bin_edges"]
        bin_edges = None
        if bin_edges_payload is not None:
            bin_edges = np.asarray(bin_edges_payload, dtype=np.float32)
        initial_prediction = float(payload["initial_prediction"])
        tree_payloads = payload["trees"]
        trees: List[Tree] = []
        for tree_nodes in tree_payloads:  # type: ignore[assignment]
            tree = Tree()
            for node_data in tree_nodes:
                node = TreeNode(
                    is_leaf=bool(node_data["is_leaf"]),
                    prediction=float(node_data["prediction"]),
                    feature=node_data["feature"],
                    threshold=node_data["threshold"],
                    left=node_data["left"],
                    right=node_data["right"],
                    depth=int(node_data["depth"]),
                )
                tree.add_node(node)
            trees.append(tree)
        return cls(
            config=config,
            bin_edges=bin_edges,
            initial_prediction=initial_prediction,
            trees=trees,
        )

    def predict_binned(self, X_binned: np.ndarray) -> np.ndarray:
        """Predict using pre-binned features."""
        X_arr = np.asarray(X_binned)
        if X_arr.ndim != 2:
            raise ValueError("X_binned must be 2D")

        if self.config.device == "cuda" and cuda_predict_forest is not None:
            flat = self.flatten_forest()
            X_device = np.ascontiguousarray(X_arr, dtype=np.uint8)
            return cuda_predict_forest(
                X_device,
                flat.tree_offsets,
                flat.features,
                flat.thresholds,
                flat.lefts,
                flat.rights,
                flat.is_leaf,
                flat.values,
                float(self.config.learning_rate / self.config.pack_size),
                float(self.initial_prediction),
            )

        preds = np.full(X_arr.shape[0], self.initial_prediction, dtype=np.float32)
        tree_weight = self.config.learning_rate / self.config.pack_size
        for tree in self.trees:
            preds += tree_weight * tree.predict_binned(X_arr)
        return preds

    def flatten_forest(self) -> FlattenedForest:
        """Return a flattened view of the ensemble suitable for GPU traversal."""

        if self._flattened is not None:
            return self._flattened

        n_trees = len(self.trees)
        tree_offsets = np.zeros(n_trees + 1, dtype=np.int32)
        total_nodes = int(sum(len(tree.nodes) for tree in self.trees))

        features = np.full(total_nodes, -1, dtype=np.int32)
        thresholds = np.full(total_nodes, -1, dtype=np.int32)
        lefts = np.full(total_nodes, -1, dtype=np.int32)
        rights = np.full(total_nodes, -1, dtype=np.int32)
        is_leaf = np.zeros(total_nodes, dtype=np.uint8)
        values = np.zeros(total_nodes, dtype=np.float32)

        cursor = 0
        for tree_idx, tree in enumerate(self.trees):
            tree_offsets[tree_idx] = cursor
            for local_idx, node in enumerate(tree.nodes):
                global_idx = cursor + local_idx
                if node.feature is not None:
                    features[global_idx] = int(node.feature)
                if node.threshold is not None:
                    thresholds[global_idx] = int(node.threshold)
                if node.left is not None:
                    lefts[global_idx] = cursor + int(node.left)
                if node.right is not None:
                    rights[global_idx] = cursor + int(node.right)
                is_leaf[global_idx] = 1 if node.is_leaf else 0
                values[global_idx] = np.float32(node.prediction)
            cursor += len(tree.nodes)
        tree_offsets[n_trees] = cursor

        flattened = FlattenedForest(
            tree_offsets=tree_offsets,
            features=features,
            thresholds=thresholds,
            lefts=lefts,
            rights=rights,
            is_leaf=is_leaf,
            values=values,
        )
        self._flattened = flattened
        return flattened
