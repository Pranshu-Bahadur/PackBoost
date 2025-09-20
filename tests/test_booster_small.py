import numpy as np
import numpy as np
import pandas as pd

from packboost.booster import PackBoost
from packboost.config import PackBoostConfig


def make_dataset(n_rows: int = 200, n_features: int = 6, n_eras: int = 10):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    coefs = rng.normal(size=n_features).astype(np.float32)
    y = X @ coefs + 0.1 * rng.standard_normal(n_rows)
    eras = np.repeat(np.arange(n_eras, dtype=np.int16), n_rows // n_eras)
    if eras.shape[0] < n_rows:
        extra = rng.integers(0, n_eras, size=n_rows - eras.shape[0], dtype=np.int16)
        eras = np.concatenate([eras, extra])
    return X, y.astype(np.float32), eras


def test_booster_fit_predict_improves_loss():
    X, y, eras = make_dataset()
    config = PackBoostConfig(
        pack_size=4,
        max_depth=3,
        learning_rate=0.2,
        lambda_l2=1e-3,
        lambda_dro=0.1,
        direction_weight=0.05,
        min_samples_leaf=5,
        max_bins=63,
        layer_feature_fraction=0.6,
        random_state=13,
    )
    booster = PackBoost(config)
    booster.fit(X, y, eras, num_rounds=3)
    preds = booster.predict(X)
    assert preds.shape == (X.shape[0],)
    baseline_loss = float(np.mean((y - y.mean()) ** 2))
    trained_loss = float(np.mean((y - preds) ** 2))
    assert trained_loss < baseline_loss


def test_booster_accepts_dataframe():
    X, y, eras = make_dataset(n_rows=120, n_features=4, n_eras=6)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    config = PackBoostConfig(pack_size=2, max_depth=2, learning_rate=0.1, random_state=7)
    booster = PackBoost(config)
    booster.fit(df, y, eras, num_rounds=2)
    preds = booster.predict(df)
    assert preds.shape == (df.shape[0],)
