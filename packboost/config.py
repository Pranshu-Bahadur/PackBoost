"""Configuration objects for PackBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PackBoostConfig:
    """Hyper-parameters steering PackBoost training.

    Parameters
    ----------
    pack_size:
        Number of trees constructed jointly per boosting round.
    max_depth:
        Maximum depth (exclusive of root depth 0) for each tree.
    learning_rate:
        Shrinkage applied to tree outputs before updating predictions.
    lambda_l2:
        L2 regularisation added to Hessian sums when computing leaf values.
    lambda_dro:
        Penalty weight applied to the per-era gain standard deviation.
    direction_weight:
        Weight applied to the directional agreement metric per split.
    min_samples_leaf:
        Minimum number of samples required in each child after a split.
    max_bins:
        Maximum number of histogram bins per feature (<= 256 for uint8 storage).
    layer_feature_fraction:
        Fraction of features sampled uniformly without replacement per depth.
    era_alpha:
        Non-negative pseudo-count blended with per-era sample counts when
        computing DES weights. ``0`` keeps equal-era weighting while larger
        values gradually favour eras with more rows.
    era_tile_size:
        Number of eras processed together when streaming DES statistics.
    histogram_mode:
        Policy for deriving right child histograms. ``"rebuild"`` recomputes
        them via reverse scans, ``"subtract"`` reuses parent totals minus left
        aggregates, and ``"auto"`` picks the faster option per node.
    feature_block_size:
        Maximum number of features scored together for a node batch. ``0``
        processes the entire feature subset at once.
    enable_node_batching:
        Toggle for the depth-synchronous node batching frontier. When ``False``
        nodes are processed sequentially (useful for debugging).
    random_state:
        Optional seed controlling RNG for feature subsampling.
    device:
        Torch device identifier (``"cpu"`` or ``"cuda"``) for tensor ops.
    """

    pack_size: int = 8
    max_depth: int = 6
    learning_rate: float = 0.1
    lambda_l2: float = 1e-6
    lambda_dro: float = 0.0
    direction_weight: float = 0.0
    min_samples_leaf: int = 20
    max_bins: int = 63
    layer_feature_fraction: float = 1.0
    era_alpha: float = 0.0
    era_tile_size: int = 32
    histogram_mode: Literal["rebuild", "subtract", "auto"] = "subtract"
    feature_block_size: int = 64
    enable_node_batching: bool = True
    random_state: int | None = None
    device: str = "cpu"
