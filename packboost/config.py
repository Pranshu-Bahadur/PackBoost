"""Configuration objects for PackBoost."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackBoostConfig:
    """Configuration for PackBoost booster.

    Attributes
    ----------
    pack_size: int
        Number of trees to grow per boosting round (the pack size ``B``).
    max_depth: int
        Maximum tree depth per tree (depth ``D``).
    learning_rate: float
        Learning rate applied to each tree's leaf values.
    lambda_l2: float
        L2 regularisation term applied when computing leaf values.
    lambda_dro: float
        Regularisation weight for the DRO score (multiplier on standard deviation).
    min_samples_leaf: int
        Minimum number of samples per leaf; splits violating this are rejected.
    max_bins: int
        Number of quantile bins per feature (``B`` in the design doc).
    random_state: int
        Seed controlling deterministic feature sampling.
    layer_feature_fraction: float
        Fraction of features sampled at each depth; all trees share the sampled subset.
    direction_weight: float
        Optional weight added for directional agreement when choosing splits.
    era_tile_size: int
        Number of eras processed at once when evaluating splits (``E``).
    device: str
        Execution device, ``"cpu"`` or ``"cuda"`` (GPU requires CuPy).
    cuda_threads_per_block: int
        Threads per block for CUDA frontier kernels.
    cuda_rows_per_thread: int
        Number of frontier rows processed per CUDA thread.
    """

    pack_size: int = 4
    max_depth: int = 4
    learning_rate: float = 0.05
    lambda_l2: float = 1.0
    lambda_dro: float = 0.5
    min_samples_leaf: int = 10
    max_bins: int = 128
    random_state: int = 42
    layer_feature_fraction: float = 0.5
    direction_weight: float = 0.0
    era_tile_size: int = 32
    device: str = "cpu"
    cuda_threads_per_block: int = 128
    cuda_rows_per_thread: int = 1

    def validate(self, n_features: int) -> None:
        """Validate configuration values.

        Parameters
        ----------
        n_features: int
            Total number of features in the dataset.
        """
        if self.pack_size <= 0:
            raise ValueError("pack_size must be positive")
        if not (1 <= self.max_depth <= 16):
            raise ValueError("max_depth must be in [1, 16]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        if self.lambda_dro < 0:
            raise ValueError("lambda_dro must be non-negative")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if not (2 <= self.max_bins <= 255):
            raise ValueError("max_bins must be in [2, 255]")
        if not (0 < self.layer_feature_fraction <= 1):
            raise ValueError("layer_feature_fraction must be in (0, 1]")
        if not (0 <= self.direction_weight <= 1):
            raise ValueError("direction_weight must be in [0, 1]")
        if self.era_tile_size <= 0:
            raise ValueError("era_tile_size must be positive")
        if self.cuda_threads_per_block <= 0 or self.cuda_threads_per_block > 1024:
            raise ValueError("cuda_threads_per_block must be in [1, 1024]")
        if self.cuda_rows_per_thread <= 0:
            raise ValueError("cuda_rows_per_thread must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        min_features = max(1, int(self.layer_feature_fraction * n_features))
        if min_features == 0:
            raise ValueError("layer_feature_fraction too small for number of features")
