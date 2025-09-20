import numpy as np
import pytest
import torch

from packboost.booster import NodeShard, PackBoost
from packboost.config import PackBoostConfig


@pytest.mark.parametrize("era_alpha", [0.0, 1.5])
def test_des_metrics_match_numpy(era_alpha: float) -> None:
    config = PackBoostConfig(
        max_bins=4,
        min_samples_leaf=1,
        lambda_l2=0.5,
        lambda_dro=0.1,
        direction_weight=0.2,
        era_alpha=era_alpha,
    )
    booster = PackBoost(config)
    device = booster._device

    grad = torch.tensor(
        [0.5, -0.2, 0.1, -0.3, 0.4, -0.1], dtype=torch.float32, device=device
    )
    hess = torch.tensor(
        [1.0, 0.8, 1.2, 1.0, 0.9, 1.1], dtype=torch.float32, device=device
    )
    bins = torch.tensor(
        [[0], [1], [1], [2], [2], [3]], dtype=torch.int64, device=device
    )

    era_rows = [
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.tensor([2, 3, 4, 5], dtype=torch.int64, device=device),
    ]
    node = NodeShard(tree_id=0, node_id=0, depth=0, era_rows=era_rows)
    feature_subset = torch.tensor([0], dtype=torch.int64, device=device)

    all_rows, era_ids = booster._stack_node_rows(node.era_rows)
    grad_all = grad[all_rows]
    hess_all = hess[all_rows]
    feature_values = bins[all_rows, 0].to(dtype=torch.int64)
    counts, grad_hist, hess_hist = booster._compute_histograms(
        feature_values,
        grad_all,
        hess_all,
        era_ids,
        num_eras=len(era_rows),
        num_bins=config.max_bins,
    )

    prefix_counts = counts[:, :-1].cumsum(dim=1)
    prefix_grad = grad_hist[:, :-1].cumsum(dim=1)
    prefix_hess = hess_hist[:, :-1].cumsum(dim=1)

    total_count_e = counts.sum(dim=1, keepdim=True)
    total_grad_e = grad_hist.sum(dim=1, keepdim=True)
    total_hess_e = hess_hist.sum(dim=1, keepdim=True)

    left_count = prefix_counts
    right_count = total_count_e - left_count
    valid = (left_count > 0) & (right_count > 0)

    left_grad = prefix_grad
    right_grad = total_grad_e - left_grad
    left_hess = prefix_hess
    right_hess = total_hess_e - left_hess

    parent_gain = 0.5 * (total_grad_e**2) / (total_hess_e + config.lambda_l2)
    gain_left = 0.5 * (left_grad**2) / (left_hess + config.lambda_l2)
    gain_right = 0.5 * (right_grad**2) / (right_hess + config.lambda_l2)
    era_gain = gain_left + gain_right - parent_gain

    left_value = -left_grad / (left_hess + config.lambda_l2)
    right_value = -right_grad / (right_hess + config.lambda_l2)

    total_grad = grad_all.sum()
    total_hess = hess_all.sum()
    parent_dir = torch.sign(-total_grad / (total_hess + config.lambda_l2))

    era_counts = torch.tensor(
        [float(rows.numel()) for rows in era_rows],
        dtype=torch.float32,
        device=device,
    )
    era_weights = booster._compute_era_weights(era_counts)
    mean_gain, std_gain, weight_sum = booster._weighted_welford(
        era_gain, valid, era_weights
    )
    agreement_mean = booster._directional_agreement(
        left_value,
        right_value,
        valid,
        era_weights,
        parent_dir,
        weight_sum,
    )

    agg_count = counts.sum(dim=0)
    prefix_count_total = agg_count.cumsum(dim=0)[:-1]
    left_count_total = prefix_count_total
    right_count_total = all_rows.numel() - left_count_total
    valid_global = (left_count_total >= config.min_samples_leaf) & (
        right_count_total >= config.min_samples_leaf
    )

    score = mean_gain - config.lambda_dro * std_gain + config.direction_weight * agreement_mean
    score = torch.where(
        valid_global,
        score,
        torch.full_like(score, float("-inf")),
    )

    # NumPy reference ----------------------------------------------------
    counts_np = counts.cpu().numpy()
    grad_hist_np = grad_hist.cpu().numpy()
    hess_hist_np = hess_hist.cpu().numpy()

    num_eras, num_bins = counts_np.shape
    num_thresholds = num_bins - 1
    era_gain_np = np.zeros((num_eras, num_thresholds), dtype=np.float32)
    left_val_np = np.zeros_like(era_gain_np)
    right_val_np = np.zeros_like(era_gain_np)
    valid_np = np.zeros_like(era_gain_np, dtype=bool)

    for era_idx in range(num_eras):
        for thr in range(num_thresholds):
            left_count_e = counts_np[era_idx, : thr + 1].sum()
            right_count_e = counts_np[era_idx, thr + 1 :].sum()
            if left_count_e <= 0 or right_count_e <= 0:
                continue
            valid_np[era_idx, thr] = True
            left_grad_e = grad_hist_np[era_idx, : thr + 1].sum()
            right_grad_e = grad_hist_np[era_idx, thr + 1 :].sum()
            left_hess_e = hess_hist_np[era_idx, : thr + 1].sum()
            right_hess_e = hess_hist_np[era_idx, thr + 1 :].sum()
            parent_grad_e = grad_hist_np[era_idx].sum()
            parent_hess_e = hess_hist_np[era_idx].sum()
            parent_gain_e = 0.5 * parent_grad_e**2 / (parent_hess_e + config.lambda_l2)
            gain_left_e = 0.5 * left_grad_e**2 / (left_hess_e + config.lambda_l2)
            gain_right_e = 0.5 * right_grad_e**2 / (right_hess_e + config.lambda_l2)
            era_gain_np[era_idx, thr] = gain_left_e + gain_right_e - parent_gain_e
            left_val_np[era_idx, thr] = -left_grad_e / (left_hess_e + config.lambda_l2)
            right_val_np[era_idx, thr] = -right_grad_e / (right_hess_e + config.lambda_l2)

    era_counts_np = np.array([rows.numel() for rows in era_rows], dtype=np.float32)
    if era_alpha > 0:
        weights_np = np.where(era_counts_np > 0, era_counts_np + era_alpha, 0.0)
    else:
        weights_np = np.where(era_counts_np > 0, 1.0, 0.0)

    parent_dir_np = np.sign(float((-total_grad.cpu().item()) / (total_hess.cpu().item() + config.lambda_l2)))

    mean_np = np.zeros(num_thresholds, dtype=np.float32)
    std_np = np.zeros(num_thresholds, dtype=np.float32)
    dir_np = np.zeros(num_thresholds, dtype=np.float32)

    for thr in range(num_thresholds):
        w = weights_np * valid_np[:, thr]
        weight_sum_np = w.sum()
        if weight_sum_np <= 0:
            continue
        gains = era_gain_np[:, thr]
        mean_val = float(np.dot(gains, w) / weight_sum_np)
        mean_np[thr] = mean_val
        var_val = float(np.dot(w, (gains - mean_val) ** 2) / weight_sum_np)
        std_np[thr] = np.sqrt(max(var_val, 0.0))
        if parent_dir_np == 0.0:
            dir_np[thr] = 0.0
        else:
            agreement = 0.5 * (
                (np.sign(left_val_np[:, thr]) == parent_dir_np).astype(np.float32)
                + (np.sign(right_val_np[:, thr]) == parent_dir_np).astype(np.float32)
            )
            dir_np[thr] = float(np.dot(agreement, w) / weight_sum_np)

    agg_count_np = counts_np.sum(axis=0)
    prefix_count_total_np = np.cumsum(agg_count_np)[:-1]
    left_count_total_np = prefix_count_total_np
    right_count_total_np = all_rows.numel() - left_count_total_np
    valid_global_np = (left_count_total_np >= config.min_samples_leaf) & (
        right_count_total_np >= config.min_samples_leaf
    )

    score_np = mean_np - config.lambda_dro * std_np + config.direction_weight * dir_np
    score_np = np.where(valid_global_np, score_np, -np.inf)

    # Assertions ---------------------------------------------------------
    mask_valid = valid.cpu().numpy()
    np.testing.assert_allclose(
        era_gain.cpu().numpy()[mask_valid],
        era_gain_np[mask_valid],
        rtol=1e-6,
        atol=1e-6,
    )

    mask_weighted = weight_sum.cpu().numpy() > 0
    np.testing.assert_allclose(
        mean_gain.cpu().numpy()[mask_weighted],
        mean_np[mask_weighted],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        std_gain.cpu().numpy()[mask_weighted],
        std_np[mask_weighted],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        agreement_mean.cpu().numpy()[mask_weighted],
        dir_np[mask_weighted],
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        score.cpu().numpy()[valid_global_np],
        score_np[valid_global_np],
        rtol=1e-6,
        atol=1e-6,
    )

    split = booster._find_best_split(node, grad, hess, bins, feature_subset)
    assert split is not None
    assert split.feature == 0
    expected_idx = int(np.argmax(score_np))
    assert split.threshold == expected_idx
    assert np.isclose(split.score, score_np[expected_idx])
    assert split.left_count == int(left_count_total_np[expected_idx])


def test_era_alpha_weighting() -> None:
    config = PackBoostConfig(era_alpha=2.5)
    booster = PackBoost(config)
    counts = torch.tensor([0.0, 3.0, 5.0], dtype=torch.float32, device=booster._device)
    weights = booster._compute_era_weights(counts)
    expected = torch.tensor([0.0, 5.5, 7.5], dtype=torch.float32, device=booster._device)
    torch.testing.assert_close(weights, expected)
