import unittest

import numpy as np

from packboost.config import PackBoostConfig
from packboost.des import evaluate_node_split


class DESTest(unittest.TestCase):
    def test_evaluate_node_split_prefers_consistent_direction(self) -> None:
        X_binned = np.array([[0], [1], [0], [1]], dtype=np.uint8)
        gradients = np.array([-2.0, 2.0, -1.0, 1.0], dtype=np.float32)
        hessians = np.ones_like(gradients)
        era_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        node_indices = np.arange(4, dtype=np.int32)

        config = PackBoostConfig(
            pack_size=1,
            max_depth=2,
            learning_rate=0.1,
            lambda_l2=0.0,
            lambda_dro=0.0,
            min_samples_leaf=1,
            max_bins=4,
            random_state=1,
            layer_feature_fraction=1.0,
            direction_weight=0.0,
            era_tile_size=2,
        )

        decision = evaluate_node_split(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            node_indices=node_indices,
            features=[0],
            config=config,
        )

        self.assertEqual(decision.feature, 0)
        self.assertEqual(decision.threshold, 0)
        self.assertAlmostEqual(decision.direction_agreement, 1.0, places=6)
        self.assertGreater(decision.score, 0)

    def test_dro_penalises_variance(self) -> None:
        # era-specific gains with identical means but differing variance
        X_binned = np.array(
            [
                [0],
                [1],
                [0],
                [1],
                [0],
                [1],
            ],
            dtype=np.uint8,
        )
        # Era 0 (samples 0-1, 2-3) produce stable gains, era 1 introduces noise
        gradients = np.array([-1.0, 1.0, -1.0, 1.0, -3.0, 3.0], dtype=np.float32)
        hessians = np.ones_like(gradients)
        era_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        node_indices = np.arange(6, dtype=np.int32)

        config = PackBoostConfig(
            pack_size=1,
            max_depth=2,
            learning_rate=0.1,
            lambda_l2=0.0,
            lambda_dro=1.0,
            min_samples_leaf=1,
            max_bins=4,
            random_state=1,
            layer_feature_fraction=1.0,
            direction_weight=0.0,
            era_tile_size=2,
        )

        decision = evaluate_node_split(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            node_indices=node_indices,
            features=[0],
            config=config,
        )

        self.assertLess(decision.score, 0.5)


if __name__ == "__main__":
    unittest.main()
