import unittest
from dataclasses import replace

import numpy as np

from packboost.booster import PackBoost
from packboost.backends import cpu_frontier_evaluate, cuda_frontier_evaluate
from packboost.config import PackBoostConfig
from packboost.des import evaluate_frontier, evaluate_node_split


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

    def test_cpu_frontier_evaluate_matches_python_partitions(self) -> None:
        if cpu_frontier_evaluate is None:
            self.skipTest("cpu_frontier_evaluate unavailable")

        X_binned = np.array(
            [
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )
        gradients = np.array([-2.0, 2.5, -1.5, 1.0, -0.5, 0.5], dtype=np.float32)
        hessians = np.ones_like(gradients)
        era_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)

        config = PackBoostConfig(
            pack_size=1,
            max_depth=2,
            learning_rate=0.1,
            lambda_l2=0.0,
            lambda_dro=0.0,
            min_samples_leaf=1,
            max_bins=4,
            random_state=0,
            layer_feature_fraction=1.0,
            direction_weight=0.0,
            era_tile_size=2,
        )

        node_samples = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([3, 4, 5], dtype=np.int32),
        ]
        feature_subset = np.array([0, 1], dtype=np.int32)

        booster = PackBoost(config)
        booster._num_eras = int(era_ids.max()) + 1

        batch = booster._prepare_frontier_batch(node_samples, era_ids)  # type: ignore[attr-defined]
        native_decisions = booster._evaluate_frontier_batch(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            batch=batch,
            features=feature_subset,
        )

        python_decisions = evaluate_frontier(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            node_indices_list=node_samples,
            features=feature_subset,
            config=config,
        )

        for native, python_decision in zip(native_decisions, python_decisions):
            self.assertEqual(native.feature, python_decision.feature)
            self.assertEqual(native.threshold, python_decision.threshold)
            self.assertAlmostEqual(native.score, python_decision.score, places=6)
            self.assertAlmostEqual(native.direction_agreement, python_decision.direction_agreement, places=6)
            self.assertAlmostEqual(native.left_value, python_decision.left_value, places=6)
            self.assertAlmostEqual(native.right_value, python_decision.right_value, places=6)
            np.testing.assert_array_equal(native.left_indices, python_decision.left_indices)
            np.testing.assert_array_equal(native.right_indices, python_decision.right_indices)

    def test_cuda_frontier_evaluate_matches_cpu(self) -> None:
        if cuda_frontier_evaluate is None:
            self.skipTest("cuda_frontier_evaluate unavailable")

        X_binned = np.array(
            [
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )
        gradients = np.array([-2.0, 2.5, -1.5, 1.0, -0.5, 0.5], dtype=np.float32)
        hessians = np.ones_like(gradients)
        era_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)

        config = PackBoostConfig(
            pack_size=1,
            max_depth=2,
            learning_rate=0.1,
            lambda_l2=0.0,
            lambda_dro=0.0,
            min_samples_leaf=1,
            max_bins=4,
            random_state=0,
            layer_feature_fraction=1.0,
            direction_weight=0.0,
            era_tile_size=2,
            device="cuda",
        )

        node_samples = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([3, 4, 5], dtype=np.int32),
        ]
        feature_subset = np.array([0, 1], dtype=np.int32)

        booster = PackBoost(config)
        booster._num_eras = int(era_ids.max()) + 1

        batch = booster._prepare_frontier_batch(node_samples, era_ids)  # type: ignore[attr-defined]
        gpu_decisions = booster._evaluate_frontier_batch(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            batch=batch,
            features=feature_subset,
        )

        cpu_booster = PackBoost(replace(config, device="cpu"))
        cpu_booster._num_eras = booster._num_eras
        cpu_decisions = cpu_booster._evaluate_frontier_batch(
            X_binned=X_binned,
            gradients=gradients,
            hessians=hessians,
            era_ids=era_ids,
            batch=batch,
            features=feature_subset,
        )

        for gpu_decision, cpu_decision in zip(gpu_decisions, cpu_decisions):
            self.assertEqual(gpu_decision.feature, cpu_decision.feature)
            self.assertEqual(gpu_decision.threshold, cpu_decision.threshold)
            self.assertAlmostEqual(gpu_decision.score, cpu_decision.score, places=6)
            self.assertAlmostEqual(gpu_decision.direction_agreement, cpu_decision.direction_agreement, places=6)
            self.assertAlmostEqual(gpu_decision.left_value, cpu_decision.left_value, places=6)
            self.assertAlmostEqual(gpu_decision.right_value, cpu_decision.right_value, places=6)
            np.testing.assert_array_equal(gpu_decision.left_indices, cpu_decision.left_indices)
            np.testing.assert_array_equal(gpu_decision.right_indices, cpu_decision.right_indices)

    def test_fit_supports_callbacks_and_eval_set(self) -> None:
        X = np.random.RandomState(0).randn(40, 3).astype(np.float32)
        y = np.random.RandomState(1).randn(40).astype(np.float32)

        config = PackBoostConfig(
            pack_size=1,
            max_depth=2,
            learning_rate=0.1,
            lambda_l2=0.1,
            lambda_dro=0.0,
            min_samples_leaf=2,
            max_bins=16,
            random_state=2,
            layer_feature_fraction=1.0,
            direction_weight=0.0,
            era_tile_size=4,
        )

        booster = PackBoost(config)
        metrics: list[tuple[int, float, float]] = []

        class Capture:
            def on_round(self, _booster, info: dict) -> None:
                metrics.append(
                    (
                        info.get("round", 0),
                        float(info.get("train_corr", 0.0)),
                        float(info.get("valid_corr", 0.0)),
                    )
                )

        booster.fit(
            X,
            y,
            era_ids=None,
            num_rounds=3,
            eval_set=(X, y, None),
            callbacks=[Capture()],
            log_evaluation=None,
        )

        self.assertEqual(len(metrics), 3)
        for _, train_corr, valid_corr in metrics:
            self.assertTrue(np.isfinite(train_corr))
            self.assertTrue(np.isfinite(valid_corr))


if __name__ == "__main__":
    unittest.main()
