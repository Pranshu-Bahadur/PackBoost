"""Regression tests for execution modes and histogram policies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from packboost.booster import PackBoost
from packboost.config import PackBoostConfig
from packboost.data import apply_bins


def make_random_regression(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 6)).astype(np.float32)
    coefs = rng.normal(size=6).astype(np.float32)
    y = X @ coefs + 0.1 * rng.standard_normal(X.shape[0])
    eras = rng.integers(0, 4, size=X.shape[0], dtype=np.int32)
    return X, y.astype(np.float32), eras


def tree_signature(booster: PackBoost) -> list[tuple[int, int, bool]]:
    signature: list[tuple[int, int, bool]] = []
    for tree in booster.trees:
        for node in tree.nodes:
            signature.append((node.feature, node.threshold, node.is_leaf))
    return signature


def test_pack_average_weight_applied() -> None:
    X, y, eras = make_random_regression()
    config = PackBoostConfig(
        pack_size=4,
        max_depth=2,
        learning_rate=0.2,
        lambda_l2=1e-3,
        min_samples_leaf=5,
        max_bins=63,
        random_state=7,
        histogram_mode="rebuild",
    )
    booster = PackBoost(config)
    booster.fit(X, y, eras, num_rounds=2)

    assert booster._tree_weight is not None
    expected_weight = config.learning_rate / config.pack_size
    assert pytest.approx(expected_weight) == booster._tree_weight

    assert booster._binner is not None
    bins = apply_bins(X, booster._binner.bin_edges, booster.config.max_bins)
    bins_tensor = torch.from_numpy(bins).to(device=booster._device)
    manual = np.zeros(X.shape[0], dtype=np.float32)
    for tree in booster.trees:
        manual += booster._tree_weight * tree.predict_bins(bins_tensor).cpu().numpy()
    preds = booster.predict(X)
    np.testing.assert_allclose(preds, manual, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["subtract", "auto"])
def test_histogram_modes_match_rebuild(mode: str) -> None:
    X, y, eras = make_random_regression(seed=202)
    base_kwargs = dict(
        pack_size=1,
        max_depth=3,
        learning_rate=0.1,
        lambda_l2=1e-3,
        min_samples_leaf=5,
        max_bins=31,
        random_state=11,
        enable_node_batching=True,
    )

    rebuild = PackBoost(PackBoostConfig(histogram_mode="rebuild", **base_kwargs))
    rebuild.fit(X, y, eras, num_rounds=2)

    other = PackBoost(PackBoostConfig(histogram_mode=mode, **base_kwargs))
    other.fit(X, y, eras, num_rounds=2)

    np.testing.assert_allclose(rebuild.predict(X), other.predict(X), rtol=1e-6, atol=1e-6)
    assert tree_signature(rebuild) == tree_signature(other)


def test_histogram_subtraction_invariants_random() -> None:
    config = PackBoostConfig(max_bins=16)
    booster = PackBoost(config)
    device = booster._device
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    for _ in range(20):
        rows = 128
        features = 3
        num_eras = 4
        bins_matrix = torch.randint(0, config.max_bins, (rows, features), generator=rng, dtype=torch.int64, device=device)
        grad_rows = torch.randn(rows, generator=rng, dtype=torch.float32, device=device)
        hess_rows = torch.rand(rows, generator=rng, dtype=torch.float32, device=device) + 0.1
        era_ids = torch.randint(0, num_eras, (rows,), generator=rng, dtype=torch.int64, device=device)
        node_ids = torch.zeros(rows, dtype=torch.int64, device=device)

        counts, grad_hist, hess_hist = booster._batched_histograms_nodes(
            bins_matrix, grad_rows, hess_rows, era_ids, node_ids, num_nodes=1, num_eras=num_eras, num_bins=config.max_bins
        )

        counts = counts[:, 0]
        grad_hist = grad_hist[:, 0]
        hess_hist = hess_hist[:, 0]

        left_count = counts[:, :, :-1].cumsum(dim=2)
        left_grad = grad_hist[:, :, :-1].cumsum(dim=2)
        left_hess = hess_hist[:, :, :-1].cumsum(dim=2)
        total_count = counts.sum(dim=2, keepdim=True)
        total_grad = grad_hist.sum(dim=2, keepdim=True)
        total_hess = hess_hist.sum(dim=2, keepdim=True)

        right_count_sub = total_count - left_count
        right_grad_sub = total_grad - left_grad
        right_hess_sub = total_hess - left_hess

        right_count_rebuild = counts[:, :, 1:].flip(2).cumsum(2).flip(2)
        right_grad_rebuild = grad_hist[:, :, 1:].flip(2).cumsum(2).flip(2)
        right_hess_rebuild = hess_hist[:, :, 1:].flip(2).cumsum(2).flip(2)

        torch.testing.assert_close(right_count_sub, right_count_rebuild, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(right_grad_sub, right_grad_rebuild, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(right_hess_sub, right_hess_rebuild, atol=1e-5, rtol=1e-6)

        assert torch.all(right_count_sub >= -1e-5)
        assert torch.all(right_hess_sub >= -1e-5)
        torch.testing.assert_close(left_count + right_count_sub, total_count.expand_as(left_count), atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(left_grad + right_grad_sub, total_grad.expand_as(left_grad), atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(left_hess + right_hess_sub, total_hess.expand_as(left_hess), atol=1e-5, rtol=1e-6)


def test_node_batching_matches_sequential() -> None:
    X, y, eras = make_random_regression(seed=99)
    common = dict(
        pack_size=1,
        max_depth=3,
        learning_rate=0.1,
        lambda_l2=1e-3,
        min_samples_leaf=4,
        max_bins=31,
        random_state=5,
        histogram_mode="rebuild",
    )

    batched = PackBoost(PackBoostConfig(enable_node_batching=True, **common))
    batched.fit(X, y, eras, num_rounds=3)

    sequential = PackBoost(PackBoostConfig(enable_node_batching=False, **common))
    sequential.fit(X, y, eras, num_rounds=3)

    np.testing.assert_allclose(batched.predict(X), sequential.predict(X), rtol=1e-6, atol=1e-6)
    assert tree_signature(batched) == tree_signature(sequential)
