"""scikit-learn wrapper for PackBoost."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .booster import PackBoost
from .config import PackBoostConfig


class PackBoostRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn compatible estimator wrapping :class:`PackBoost`."""

    def __init__(
        self,
        *,
        pack_size: int = 4,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        lambda_l2: float = 1.0,
        lambda_dro: float = 0.5,
        min_samples_leaf: int = 10,
        max_bins: int = 128,
        random_state: int = 42,
        layer_feature_fraction: float = 0.5,
        direction_weight: float = 0.0,
        era_tile_size: int = 32,
        device: str = "cpu",
        cuda_threads_per_block: int = 128,
        cuda_rows_per_thread: int = 1,
        num_rounds: int = 10,
        prebinned: bool = False,
    ) -> None:
        self.pack_size = pack_size
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.lambda_dro = lambda_dro
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.random_state = random_state
        self.layer_feature_fraction = layer_feature_fraction
        self.direction_weight = direction_weight
        self.era_tile_size = era_tile_size
        self.device = device
        self.cuda_threads_per_block = cuda_threads_per_block
        self.cuda_rows_per_thread = cuda_rows_per_thread
        self.num_rounds = num_rounds
        self.prebinned = prebinned
        self._booster: Optional[PackBoost] = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, era: Optional[Sequence[int]] = None) -> "PackBoostRegressor":
        """Fit the estimator.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y: np.ndarray
            Targets of shape (n_samples,).
        era: Sequence[int] | None
            Optional era identifiers. If ``None`` a single era is assumed.
        """
        if era is not None:
            era_array: Sequence[int] | None = np.asarray(era)
        else:
            era_array = None
        config = PackBoostConfig(
            pack_size=self.pack_size,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            lambda_l2=self.lambda_l2,
            lambda_dro=self.lambda_dro,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            random_state=self.random_state,
            layer_feature_fraction=self.layer_feature_fraction,
            direction_weight=self.direction_weight,
            era_tile_size=self.era_tile_size,
            device=self.device,
            cuda_threads_per_block=self.cuda_threads_per_block,
            cuda_rows_per_thread=self.cuda_rows_per_thread,
            prebinned=self.prebinned,
        )
        booster = PackBoost(config)
        booster.fit(np.asarray(X), np.asarray(y), era_array, num_rounds=self.num_rounds)
        self._booster = booster
        self.model_ = booster.model
        self.bin_edges_ = booster.model.bin_edges
        self.prebinned_ = booster.model.config.prebinned
        self.auto_prebinned_ = booster._auto_prebinned
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Estimator has not been fitted")
        return self._booster.predict(np.asarray(X))

    def get_model(self) -> PackBoost:
        if self._booster is None:
            raise RuntimeError("Estimator has not been fitted")
        return self._booster
