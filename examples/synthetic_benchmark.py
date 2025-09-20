"""Benchmark PackBoost against leading tree ensembles on synthetic data."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packboost.booster import PackBoost
from packboost.config import PackBoostConfig

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install xgboost to run this benchmark") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install lightgbm to run this benchmark") from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install catboost to run this benchmark") from exc


N_SAMPLES = 4000
N_FEATURES = 20
N_ERAS = 1
SEED = 123

N_TREES = 800
MAX_DEPTH = 5
LEARNING_RATE = 0.1


@dataclass
class BenchmarkResult:
    name: str
    fit_time: float
    predict_time: float
    r2: float


def generate_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an OOD-style regression dataset with optional era shifts."""
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=10.0,
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    era = rng.integers(0, N_ERAS, size=N_SAMPLES, dtype=np.int32)
    offsets = rng.normal(0.0, 5.0, size=N_ERAS)
    y = y + offsets[era]
    return X.astype(np.float32), y.astype(np.float32), era


def benchmark(
    name: str,
    fit_fn: Callable[[], None],
    predict_fn: Callable[[], np.ndarray],
    y_true: np.ndarray,
) -> BenchmarkResult:
    """Measure fit/predict time and compute R^2."""
    t0 = time.perf_counter()
    fit_fn()
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = predict_fn()
    predict_time = time.perf_counter() - t0

    r2 = float(r2_score(y_true, preds))
    return BenchmarkResult(name=name, fit_time=fit_time, predict_time=predict_time, r2=r2)


if __name__ == "__main__":
    X, y, era = generate_data()
    X_train, X_test, y_train, y_test, era_train, era_test = train_test_split(
        X, y, era, test_size=0.2, random_state=SEED
    )
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    results: List[BenchmarkResult] = []

    # PackBoost configuration
    pack_config = PackBoostConfig(
        pack_size=2,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        lambda_l2=1e-6,
        lambda_dro=0.0,
        min_samples_leaf=20,
        max_bins=127,
        random_state=SEED,
        layer_feature_fraction=1.0,
        direction_weight=0.0,
        era_tile_size=32,
    )
    pack_rounds = N_TREES // pack_config.pack_size
    if pack_rounds * pack_config.pack_size != N_TREES:
        raise ValueError("N_TREES must be divisible by pack_size for this benchmark")
    pack = PackBoost(pack_config)

    def pack_fit() -> None:
        pack.fit(X_train, y_train, era_train, num_rounds=pack_rounds)

    def pack_predict() -> np.ndarray:
        return pack.predict(X_test)

    results.append(benchmark("PackBoost", pack_fit, pack_predict, y_test))

    # XGBoost baseline
    xgb = XGBRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
        max_bins=127
    )
    results.append(
        benchmark(
            "XGBoost",
            lambda: xgb.fit(X_train, y_train),
            lambda: xgb.predict(X_test),
            y_test,
        )
    )

    # LightGBM baseline
    lgbm = LGBMRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        max_bins=127
    )
    results.append(
        benchmark(
            "LightGBM",
            lambda: lgbm.fit(X_train_df, y_train),
            lambda: lgbm.predict(X_test_df),
            y_test,
        )
    )

    # CatBoost baseline
    cat = CatBoostRegressor(
        iterations=N_TREES,
        depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
        #max_bins=127
    )
    results.append(
        benchmark(
            "CatBoost",
            lambda: cat.fit(X_train, y_train),
            lambda: cat.predict(X_test),
            y_test,
        )
    )

    print("Model         Fit (s)   Predict (s)   R^2")
    print("-" * 44)
    for res in results:
        print(f"{res.name:<12} {res.fit_time:>8.3f} {res.predict_time:>12.3f} {res.r2:>7.4f}")
