#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using Clock = std::chrono::steady_clock;

struct NodeContext {
    std::vector<std::vector<int32_t>> era_rows;
    std::vector<int64_t> era_counts;
    std::vector<float> era_weights;
    std::vector<float> era_grad_totals;
    float total_grad{0.0f};
    float total_hess{0.0f};
    int64_t total_count{0};
    int parent_dir{0};
};

std::vector<int32_t> even_cut_indices(int max_bins, int k) {
    const int max_thr = std::max(1, max_bins - 1);
    int kk = std::max(1, std::min(k, max_thr));
    std::vector<int32_t> out;
    out.reserve(kk);
    if (kk == 1) {
        out.push_back(0);
        return out;
    }
    double step = static_cast<double>(max_thr - 1) / static_cast<double>(kk - 1);
    for (int i = 0; i < kk; ++i) {
        double val = std::round(static_cast<double>(i) * step);
        int32_t idx = static_cast<int32_t>(std::llround(val));
        idx = std::max<int32_t>(0, std::min<int32_t>(idx, max_thr - 1));
        out.push_back(idx);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    while (static_cast<int>(out.size()) < kk) {
        out.push_back(out.back());
    }
    return out;
}

std::vector<int32_t> mass_cut_indices(
    const std::vector<int64_t> &counts,
    int max_bins,
    int k
) {
    const int max_thr = std::max(1, max_bins - 1);
    int kk = std::max(1, std::min(k, max_thr));
    std::vector<int32_t> out;
    out.reserve(kk);
    int64_t total = 0;
    for (int64_t v : counts) {
        total += v;
    }
    if (total <= 0) {
        return even_cut_indices(max_bins, kk);
    }
    std::vector<int64_t> cdf(counts.size(), 0);
    int64_t acc = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        acc += counts[i];
        cdf[i] = acc;
    }
    double upper = static_cast<double>(total) * (1.0 - 1e-12);
    for (int i = 0; i < kk; ++i) {
        double alpha = (kk == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(kk - 1);
        double target = upper * alpha;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
        int64_t idx = (it == cdf.end()) ? static_cast<int64_t>(cdf.size() - 1)
                                        : static_cast<int64_t>(std::distance(cdf.begin(), it));
        int32_t thr = static_cast<int32_t>(std::max<int64_t>(0, std::min<int64_t>(idx - 1, max_thr - 1)));
        out.push_back(thr);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    while (static_cast<int>(out.size()) < kk) {
        auto extra = even_cut_indices(max_bins, kk);
        out.insert(out.end(), extra.begin(), extra.end());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        if (static_cast<int>(out.size()) >= kk) {
            break;
        }
    }
    if (static_cast<int>(out.size()) > kk) {
        out.resize(kk);
    }
    for (auto &v : out) {
        v = std::max<int32_t>(0, std::min<int32_t>(v, max_thr - 1));
    }
    return out;
}

int signf(float x) {
    if (x > 0.0f) {
        return 1;
    }
    if (x < 0.0f) {
        return -1;
    }
    return 0;
}

py::tuple find_best_splits_batched_cpu(
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<float, py::array::c_style | py::array::forcecast> grad,
    const py::list &nodes_era_rows,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature_subset,
    int max_bins,
    int k_cuts,
    const std::string &cut_selection,
    float lambda_l2,
    float lambda_dro,
    float direction_weight,
    float era_alpha,
    int min_samples_leaf
) {
    if (bins.ndim() != 2) {
        throw std::invalid_argument("bins must be 2D (rows, features)");
    }
    if (grad.ndim() != 1) {
        throw std::invalid_argument("grad must be 1D");
    }
    const py::ssize_t num_rows = bins.shape(0);
    const py::ssize_t num_features = bins.shape(1);
    if (grad.shape(0) != num_rows) {
        throw std::invalid_argument("grad length must match bins rows");
    }
    auto bins_view = bins.unchecked<2>();
    auto grad_view = grad.unchecked<1>();

    const int num_bins = std::max(1, max_bins);
    const int thresholds_full = std::max(1, num_bins - 1);

    std::vector<NodeContext> contexts;
    contexts.reserve(nodes_era_rows.size());
    std::vector<int> index_map;
    index_map.reserve(nodes_era_rows.size());
    int64_t rows_total = 0;

    for (py::ssize_t idx = 0; idx < nodes_era_rows.size(); ++idx) {
        py::object item = nodes_era_rows[idx];
        if (!py::isinstance<py::list>(item)) {
            throw std::invalid_argument("Each node entry must be a list of era row arrays");
        }
        py::list eras = item.cast<py::list>();
        if (eras.empty()) {
            continue;
        }
        NodeContext ctx;
        ctx.era_rows.resize(eras.size());
        ctx.era_counts.resize(eras.size());
        ctx.era_weights.resize(eras.size());
        ctx.era_grad_totals.resize(eras.size());
        bool valid = false;
        for (py::ssize_t e = 0; e < eras.size(); ++e) {
            py::array rows_arr = py::cast<py::array>(eras[e]);
            if (rows_arr.ndim() != 1) {
                throw std::invalid_argument("era row array must be 1D");
            }
            std::vector<int32_t> rows_vec;
            rows_vec.reserve(static_cast<std::size_t>(rows_arr.shape(0)));
            auto buf = rows_arr.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
            auto view = buf.unchecked<1>();
            int64_t era_count = 0;
            float era_grad = 0.0f;
            for (py::ssize_t r = 0; r < buf.shape(0); ++r) {
                int64_t row_id = view(r);
                if (row_id < 0 || row_id >= num_rows) {
                    throw std::out_of_range("row index out of bounds");
                }
                rows_vec.push_back(static_cast<int32_t>(row_id));
                era_grad += grad_view(row_id);
                ++era_count;
            }
            ctx.era_rows[e] = std::move(rows_vec);
            ctx.era_counts[e] = era_count;
            ctx.era_grad_totals[e] = era_grad;
            if (era_count > 0) {
                valid = true;
                rows_total += era_count;
            }
        }
        if (!valid) {
            continue;
        }
        ctx.total_count = 0;
        ctx.total_grad = 0.0f;
        for (std::size_t e = 0; e < ctx.era_counts.size(); ++e) {
            ctx.total_count += ctx.era_counts[e];
            ctx.total_grad += ctx.era_grad_totals[e];
            ctx.era_weights[e] = 0.0f;
            if (ctx.era_counts[e] > 0) {
                ctx.era_weights[e] = (era_alpha > 0.0f)
                    ? static_cast<float>(ctx.era_counts[e]) + era_alpha
                    : 1.0f;
            }
        }
        ctx.total_hess = static_cast<float>(ctx.total_count);
        float parent_value = -ctx.total_grad / (ctx.total_hess + lambda_l2);
        ctx.parent_dir = signf(parent_value);
        contexts.push_back(std::move(ctx));
        index_map.push_back(static_cast<int>(idx));
    }

    const std::size_t num_nodes = contexts.size();
    if (num_nodes == 0) {
        py::list decisions(index_map.size());
        py::dict stats;
        stats["nodes_processed"] = 0;
        stats["nodes_skipped"] = static_cast<int>(nodes_era_rows.size()) - static_cast<int>(index_map.size());
        stats["rows_total"] = rows_total;
        stats["feature_blocks"] = 0;
        stats["bincount_calls"] = 0;
        stats["hist_ms"] = 0.0;
        stats["scan_ms"] = 0.0;
        stats["score_ms"] = 0.0;
        stats["partition_ms"] = 0.0;
        stats["nodes_subtract_ok"] = 0;
        stats["nodes_subtract_fallback"] = 0;
        stats["nodes_rebuild"] = 0;
        stats["block_size"] = 0;
        return py::make_tuple(decisions, stats);
    }

    py::buffer_info feat_buf = feature_subset.request();
    if (feat_buf.ndim != 1) {
        throw std::invalid_argument("feature_subset must be 1D");
    }
    const int subset_size = static_cast<int>(feat_buf.size);
    auto feat_view = feature_subset.unchecked<1>();

    const std::size_t num_eras = contexts.front().era_rows.size();
    const std::size_t bins_stride = static_cast<std::size_t>(num_bins);
    const std::size_t era_stride = bins_stride * num_eras;

    std::vector<float> best_score(num_nodes, -std::numeric_limits<float>::infinity());
    std::vector<int32_t> best_feature(num_nodes, -1);
    std::vector<int32_t> best_threshold(num_nodes, -1);
    std::vector<int64_t> best_left_count(num_nodes, -1);
    std::vector<float> best_left_grad(num_nodes, 0.0f);

    double hist_time_ms = 0.0;
    double scan_time_ms = 0.0;
    double score_time_ms = 0.0;
    double partition_time_ms = 0.0;

    std::vector<int64_t> counts(num_nodes * num_eras * num_bins, 0);
    std::vector<float> grads_hist(num_nodes * num_eras * num_bins, 0.0f);
    std::vector<int64_t> prefix_counts(num_nodes * num_eras * thresholds_full, 0);
    std::vector<float> prefix_grads(num_nodes * num_eras * thresholds_full, 0.0f);
    std::vector<int64_t> totals_counts(num_nodes * num_eras, 0);
    std::vector<float> totals_grads(num_nodes * num_eras, 0.0f);
    std::vector<float> parent_gain_era(num_nodes * num_eras, 0.0f);

    for (int f_idx = 0; f_idx < subset_size; ++f_idx) {
        int32_t feature = feat_view(f_idx);
        if (feature < 0 || feature >= num_features) {
            throw std::out_of_range("feature index out of range");
        }
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(grads_hist.begin(), grads_hist.end(), 0.0f);

        auto hist_start = Clock::now();
#pragma omp parallel for schedule(static)
        for (std::int64_t n = 0; n < static_cast<std::int64_t>(num_nodes); ++n) {
            const NodeContext &ctx = contexts[n];
            for (std::size_t e = 0; e < num_eras; ++e) {
                const auto &rows = ctx.era_rows[e];
                std::size_t base = n * era_stride + e * bins_stride;
                for (std::size_t ridx = 0; ridx < rows.size(); ++ridx) {
                    int32_t row = rows[ridx];
                    uint16_t bin = bins_view(row, feature);
                    if (bin >= num_bins) {
                        continue;
                    }
                    std::size_t offset = base + static_cast<std::size_t>(bin);
                    counts[offset] += 1;
                    grads_hist[offset] += grad_view(row);
                }
            }
        }
        auto hist_end = Clock::now();
        hist_time_ms += std::chrono::duration<double, std::milli>(hist_end - hist_start).count();

        auto scan_start = Clock::now();
        std::vector<int64_t> feature_mass_counts(num_bins, 0);
        for (std::size_t n = 0; n < num_nodes; ++n) {
            for (std::size_t e = 0; e < num_eras; ++e) {
                std::size_t base_hist = n * era_stride + e * bins_stride;
                std::size_t base_prefix = n * num_eras * thresholds_full + e * thresholds_full;
                int64_t running_count = 0;
                float running_grad = 0.0f;
                for (int b = 0; b < num_bins; ++b) {
                    std::size_t idx = base_hist + static_cast<std::size_t>(b);
                    running_count += counts[idx];
                    running_grad += grads_hist[idx];
                    feature_mass_counts[b] += counts[idx];
                    if (b < thresholds_full) {
                        std::size_t pref_idx = base_prefix + static_cast<std::size_t>(b);
                        prefix_counts[pref_idx] = running_count;
                        prefix_grads[pref_idx] = running_grad;
                    }
                }
                std::size_t tot_idx = n * num_eras + e;
                totals_counts[tot_idx] = running_count;
                totals_grads[tot_idx] = running_grad;
                float parent_denom = static_cast<float>(running_count) + lambda_l2;
                if (parent_denom > 0.0f) {
                    parent_gain_era[tot_idx] = 0.5f * (running_grad * running_grad) / parent_denom;
                } else {
                    parent_gain_era[tot_idx] = 0.0f;
                }
            }
        }

        std::vector<int32_t> thresholds;
        bool use_k = (k_cuts > 0) && (k_cuts < thresholds_full);
        if (use_k) {
            if (cut_selection == "mass") {
                thresholds = mass_cut_indices(feature_mass_counts, num_bins, k_cuts);
            } else {
                thresholds = even_cut_indices(num_bins, k_cuts);
            }
        } else {
            thresholds.resize(thresholds_full);
            for (int t = 0; t < thresholds_full; ++t) {
                thresholds[t] = t;
            }
        }
        if (thresholds.empty()) {
            continue;
        }
        auto scan_end = Clock::now();
        scan_time_ms += std::chrono::duration<double, std::milli>(scan_end - scan_start).count();

        auto score_start = Clock::now();
#pragma omp parallel for schedule(static)
        for (std::int64_t n = 0; n < static_cast<std::int64_t>(num_nodes); ++n) {
            const NodeContext &ctx = contexts[n];
            const bool use_dir = (direction_weight != 0.0f) && (ctx.parent_dir != 0);
            float local_best_score = best_score[n];
            int32_t local_best_feat = best_feature[n];
            int32_t local_best_thr = best_threshold[n];
            int64_t local_best_left = best_left_count[n];
            float local_best_left_grad = best_left_grad[n];

            for (std::size_t t_idx = 0; t_idx < thresholds.size(); ++t_idx) {
                int32_t thr = thresholds[t_idx];
                float wsum = 0.0f;
                float mean = 0.0f;
                float M2 = 0.0f;
                float agr_num = 0.0f;
                float agr_den = 0.0f;
                float left_grad_sum = 0.0f;
                int64_t left_total = 0;

                for (std::size_t e = 0; e < num_eras; ++e) {
                    std::size_t base_pref = n * num_eras * thresholds_full + e * thresholds_full;
                    std::size_t pref_idx = base_pref + static_cast<std::size_t>(thr);
                    std::size_t tot_idx = n * num_eras + e;
                    int64_t total_e = totals_counts[tot_idx];
                    if (total_e == 0) {
                        continue;
                    }
                    int64_t left_e = prefix_counts[pref_idx];
                    int64_t right_e = total_e - left_e;
                    float left_grad_e = prefix_grads[pref_idx];
                    float right_grad_e = totals_grads[tot_idx] - left_grad_e;
                    bool valid_e = (left_e > 0) && (right_e > 0);
                    float weight = valid_e ? ctx.era_weights[e] : 0.0f;
                    left_total += left_e;
                    left_grad_sum += left_grad_e;
                    if (weight <= 0.0f) {
                        continue;
                    }
                    float denom_L = static_cast<float>(left_e) + lambda_l2;
                    float denom_R = static_cast<float>(right_e) + lambda_l2;
                    float parent_gain = parent_gain_era[tot_idx];
                    float gain_e = 0.5f * ((left_grad_e * left_grad_e) / denom_L +
                                           (right_grad_e * right_grad_e) / denom_R) - parent_gain;
                    float delta = gain_e - mean;
                    float wsum_new = wsum + weight;
                    if (wsum_new > 0.0f) {
                        float factor = weight / wsum_new;
                        mean += factor * delta;
                        M2 += weight * delta * (gain_e - mean);
                        wsum = wsum_new;
                    }
                    if (use_dir) {
                        float lg_dir = -left_grad_e / denom_L;
                        float rg_dir = -right_grad_e / denom_R;
                        int sign_l = signf(lg_dir);
                        int sign_r = signf(rg_dir);
                        float agree = 0.0f;
                        if (sign_l == ctx.parent_dir) {
                            agree += 0.5f;
                        }
                        if (sign_r == ctx.parent_dir) {
                            agree += 0.5f;
                        }
                        agr_num += weight * agree;
                        agr_den += weight;
                    }
                }

                int64_t right_total = ctx.total_count - left_total;
                if (left_total < min_samples_leaf || right_total < min_samples_leaf) {
                    continue;
                }
                float ws = std::max(wsum, 1e-12f);
                float std_dev = std::sqrt(std::max(0.0f, M2 / ws));
                float score = mean - lambda_dro * std_dev;
                if (use_dir && agr_den > 0.0f) {
                    score += direction_weight * (agr_num / agr_den);
                }
                float score_f = score;
                bool better = score_f > local_best_score;
                if (!better && std::abs(score_f - local_best_score) <= 1e-12f) {
                    if (left_total > local_best_left) {
                        better = true;
                    } else if (left_total == local_best_left) {
                        if (feature > local_best_feat) {
                            better = true;
                        } else if (feature == local_best_feat && thr > local_best_thr) {
                            better = true;
                        }
                    }
                }
                if (better) {
                    local_best_score = score_f;
                    local_best_feat = feature;
                    local_best_thr = thr;
                    local_best_left = left_total;
                    local_best_left_grad = static_cast<float>(left_grad_sum);
                }
            }

            best_score[n] = local_best_score;
            best_feature[n] = local_best_feat;
            best_threshold[n] = local_best_thr;
            best_left_count[n] = local_best_left;
            best_left_grad[n] = local_best_left_grad;
        }
        auto score_end = Clock::now();
        score_time_ms += std::chrono::duration<double, std::milli>(score_end - score_start).count();
    }

    py::list decisions(nodes_era_rows.size());
    auto part_start_all = Clock::now();
    for (std::size_t idx = 0; idx < index_map.size(); ++idx) {
        int original = index_map[idx];
        std::size_t ctx_idx = idx;
        const NodeContext &ctx = contexts[ctx_idx];
        if (best_feature[ctx_idx] < 0 || best_threshold[ctx_idx] < 0) {
            decisions[original] = py::none();
            continue;
        }
        py::list left_lists;
        py::list right_lists;
        for (std::size_t e = 0; e < num_eras; ++e) {
            const auto &rows = ctx.era_rows[e];
            std::vector<int64_t> left_side;
            std::vector<int64_t> right_side;
            left_side.reserve(rows.size());
            right_side.reserve(rows.size());
            for (int32_t row : rows) {
                uint16_t bin = bins_view(row, best_feature[ctx_idx]);
                if (bin <= static_cast<uint16_t>(best_threshold[ctx_idx])) {
                    left_side.push_back(static_cast<int64_t>(row));
                } else {
                    right_side.push_back(static_cast<int64_t>(row));
                }
            }
            left_lists.append(py::array_t<int64_t>(left_side.size(), left_side.data()));
            right_lists.append(py::array_t<int64_t>(right_side.size(), right_side.data()));
        }
        float left_grad = best_left_grad[ctx_idx];
        float right_grad = ctx.total_grad - left_grad;
        float left_hess = static_cast<float>(best_left_count[ctx_idx]);
        float right_hess = ctx.total_hess - left_hess;
        int64_t right_count = ctx.total_count - best_left_count[ctx_idx];

        py::dict decision;
        decision["feature"] = best_feature[ctx_idx];
        decision["threshold"] = best_threshold[ctx_idx];
        decision["score"] = best_score[ctx_idx];
        decision["left_grad"] = left_grad;
        decision["left_hess"] = left_hess;
        decision["right_grad"] = right_grad;
        decision["right_hess"] = right_hess;
        decision["left_count"] = best_left_count[ctx_idx];
        decision["right_count"] = right_count;
        decision["left_rows"] = std::move(left_lists);
        decision["right_rows"] = std::move(right_lists);

        decisions[original] = std::move(decision);
    }
    auto part_end_all = Clock::now();
    partition_time_ms = std::chrono::duration<double, std::milli>(part_end_all - part_start_all).count();

    py::dict stats;
    stats["nodes_processed"] = static_cast<int>(index_map.size());
    stats["nodes_skipped"] = static_cast<int>(nodes_era_rows.size()) - static_cast<int>(index_map.size());
    stats["rows_total"] = rows_total;
    stats["feature_blocks"] = (subset_size > 0) ? 1 : 0;
    stats["bincount_calls"] = (subset_size > 0) ? 2 : 0;
    stats["hist_ms"] = hist_time_ms;
    stats["scan_ms"] = scan_time_ms;
    stats["score_ms"] = score_time_ms;
    stats["partition_ms"] = partition_time_ms;
    stats["nodes_subtract_ok"] = static_cast<int>(index_map.size());
    stats["nodes_subtract_fallback"] = 0;
    stats["nodes_rebuild"] = 0;
    stats["block_size"] = subset_size;

    return py::make_tuple(decisions, stats);
}

}  // namespace

void register_cpu_frontier(py::module_ &m) {
    m.def(
        "_cpu_available",
        []() { return true; },
        "Native CPU backend is available."
    );
    m.def(
        "find_best_splits_batched_cpu",
        &find_best_splits_batched_cpu,
        py::arg("bins"),
        py::arg("grad"),
        py::arg("nodes_era_rows"),
        py::arg("feature_subset"),
        py::arg("max_bins"),
        py::arg("k_cuts"),
        py::arg("cut_selection"),
        py::arg("lambda_l2"),
        py::arg("lambda_dro"),
        py::arg("direction_weight"),
        py::arg("era_alpha"),
        py::arg("min_samples_leaf"),
        "Find best splits for a batch of nodes using the native CPU backend."
    );
}
