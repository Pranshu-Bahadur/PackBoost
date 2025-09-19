"""Standalone prediction utilities for PackBoost models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import PackBoostModel
from .utils.binning import apply_binning, ensure_prebinned


class PackBoostPredictor:
    """Lightweight predictor that depends only on a serialised model."""

    def __init__(self, model: PackBoostModel) -> None:
        self._model = model

    @classmethod
    def from_json(cls, path: str | Path) -> "PackBoostPredictor":
        payload = json.loads(Path(path).read_text())
        model = PackBoostModel.from_dict(payload)
        return cls(model)

    def to_json(self, path: str | Path) -> None:
        payload = self._model.to_dict()
        Path(path).write_text(json.dumps(payload))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X)
        if self._model.config.prebinned:
            X_binned = ensure_prebinned(X_array, self._model.config.max_bins)
        else:
            X_binned = apply_binning(X_array.astype(np.float32, copy=False), self._model.bin_edges)
        return self._model.predict_binned(X_binned)

    @property
    def model(self) -> PackBoostModel:
        return self._model


def load_predictor(payload: dict[str, Any]) -> PackBoostPredictor:
    model = PackBoostModel.from_dict(payload)
    return PackBoostPredictor(model)
