"""Configuration objects for PackBoost."""

from __future__ import annotations

from dataclasses import dataclass


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
    era_tile_size:
        Number of eras processed together when streaming DES statistics.
    random_state:
        Optional seed controlling RNG for feature subsampling.
    histogram_subtraction:
        If ``True`` reuse cumulative histograms to derive right child stats.
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
    era_tile_size: int = 64
    random_state: int | None = None
    histogram_subtraction: bool = True
    device: str = "cpu"
