import numpy as np
import pytest

from packboost.booster import PackBoost
from packboost.config import PackBoostConfig


def _make_dataset(seed: int = 0, n_rows: int = 320, n_features: int = 12, n_eras: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    coefs = rng.normal(size=n_features).astype(np.float32)
    y = (X @ coefs + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
    eras = np.arange(n_rows, dtype=np.int16) % n_eras
    rng.shuffle(eras)
    return X, y, eras


def _fit_model(config: PackBoostConfig, num_rounds: int = 3):
    X, y, eras = _make_dataset(seed=123)
    booster = PackBoost(config)
    booster.fit(X, y, eras, num_rounds=num_rounds)
    return booster, X, y, eras


def test_predict_packwise_matches_predict():
    config = PackBoostConfig(
        pack_size=3,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=10,
        max_bins=63,
        random_state=11,
        histogram_mode="auto",
    )
    booster, X, _, _ = _fit_model(config)
    preds_default = booster.predict(X)
    preds_packwise = booster.predict_packwise(X, block_size_trees=2)
    np.testing.assert_allclose(preds_default, preds_packwise, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("seed", [3, 7, 19])
def test_split_determinism(seed: int):
    config = PackBoostConfig(
        pack_size=2,
        max_depth=4,
        learning_rate=0.15,
        min_samples_leaf=8,
        max_bins=63,
        histogram_mode="auto",
        random_state=seed,
    )
    booster1, X, y, eras = _fit_model(config)
    booster2 = PackBoost(config)
    booster2.fit(X, y, eras, num_rounds=3)

    assert len(booster1.trees) == len(booster2.trees)
    for tree_a, tree_b in zip(booster1.trees, booster2.trees):
        assert len(tree_a.nodes) == len(tree_b.nodes)
        for node_a, node_b in zip(tree_a.nodes, tree_b.nodes):
            assert node_a.feature == node_b.feature
            assert node_a.threshold == node_b.threshold
            assert node_a.is_leaf == node_b.is_leaf
            if node_a.is_leaf:
                assert np.isclose(node_a.value, node_b.value, atol=1e-9)


@pytest.mark.parametrize("hist_mode", ["auto", "subtract", "rebuild"])
def test_validator_accounting(hist_mode: str):
    config = PackBoostConfig(
        pack_size=2,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=6,
        max_bins=63,
        histogram_mode=hist_mode,
        random_state=5,
    )
    booster, _, _, _ = _fit_model(config, num_rounds=2)

    for depth_log in booster._depth_logs:
        nodes_processed = depth_log["nodes_processed"]
        assert depth_log["nodes_subtract_ok"] + depth_log["nodes_rebuild"] == nodes_processed
        if hist_mode == "rebuild":
            assert depth_log["nodes_subtract_ok"] == 0
        else:
            assert depth_log["nodes_subtract_ok"] >= 0
            assert depth_log["nodes_subtract_fallback"] <= depth_log["nodes_rebuild"]
