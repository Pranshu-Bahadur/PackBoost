import unittest

import numpy as np
from sklearn.datasets import make_regression

from packboost.wrapper import PackBoostRegressor


class SklearnWrapperTest(unittest.TestCase):
    def test_sklearn_wrapper_deterministic(self) -> None:
        X, y = make_regression(n_samples=200, n_features=5, random_state=123)
        rng = np.random.default_rng(42)
        era = rng.integers(low=0, high=5, size=X.shape[0], dtype=np.int32)

        est1 = PackBoostRegressor(
            pack_size=2,
            max_depth=3,
            learning_rate=0.1,
            lambda_l2=0.1,
            lambda_dro=0.2,
            min_samples_leaf=5,
            max_bins=32,
            random_state=7,
            layer_feature_fraction=0.6,
            direction_weight=0.1,
            era_tile_size=4,
            num_rounds=3,
        )
        est1.fit(X, y, era=era)
        preds1 = est1.predict(X)

        est2 = PackBoostRegressor(
            pack_size=2,
            max_depth=3,
            learning_rate=0.1,
            lambda_l2=0.1,
            lambda_dro=0.2,
            min_samples_leaf=5,
            max_bins=32,
            random_state=7,
            layer_feature_fraction=0.6,
            direction_weight=0.1,
            era_tile_size=4,
            num_rounds=3,
        )
        est2.fit(X, y, era=era)
        preds2 = est2.predict(X)

        self.assertEqual(preds1.shape, (X.shape[0],))
        self.assertTrue(np.allclose(preds1, preds2))


if __name__ == "__main__":
    unittest.main()
