#include "backend.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cmath>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;

static void check_contiguous(const py::buffer_info& info, const char* name) {
    if (info.ndim == 0) {
        throw std::invalid_argument(std::string(name) + " must be at least 1-D");
    }
    if (info.strides.back() != info.itemsize) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
}

static py::array_t<float> array_from_vector_float(const std::vector<float>& data, const std::vector<py::ssize_t>& shape) {
    py::array_t<float> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    return arr;
}

static py::array_t<int32_t> array_from_vector_int(const std::vector<int32_t>& data, const std::vector<py::ssize_t>& shape) {
    py::array_t<int32_t> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(int32_t));
    return arr;
}

namespace packboost {


HistogramBuffers build_histograms_cpu(
    const uint8_t* bins,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features,
    int max_bins,
    int n_eras) {

    const std::size_t hist_size = static_cast<std::size_t>(n_features) * max_bins * n_eras;
    HistogramBuffers buffers;
    buffers.grad.assign(hist_size, 0.0f);
    buffers.hess.assign(hist_size, 0.0f);
    buffers.count.assign(hist_size, 0);

    const std::size_t feat_stride = static_cast<std::size_t>(max_bins) * n_eras;

#pragma omp parallel for schedule(static)
    for (std::size_t feature = 0; feature < n_features; ++feature) {
        std::vector<float> local_grad(feat_stride, 0.0f);
        std::vector<float> local_hess(feat_stride, 0.0f);
        std::vector<int32_t> local_count(feat_stride, 0);

        for (std::size_t row = 0; row < n_rows; ++row) {
            const int era = static_cast<int>(era_inverse[row]);
            if (era < 0 || era >= n_eras) {
                continue;
            }
            const uint8_t bin = bins[row * n_features + feature];
            if (bin >= max_bins) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(bin) * n_eras + static_cast<std::size_t>(era);
            local_grad[idx] += gradients[row];
            local_hess[idx] += hessians[row];
            local_count[idx] += 1;
        }

        float* grad_out = buffers.grad.data() + feature * feat_stride;
        float* hess_out = buffers.hess.data() + feature * feat_stride;
        int32_t* count_out = buffers.count.data() + feature * feat_stride;
        std::memcpy(grad_out, local_grad.data(), feat_stride * sizeof(float));
        std::memcpy(hess_out, local_hess.data(), feat_stride * sizeof(float));
        std::memcpy(count_out, local_count.data(), feat_stride * sizeof(int32_t));
    }

    return buffers;
}

static py::tuple cpu_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> era_inverse,
    int max_bins,
    int n_eras) {

    py::buffer_info bins_info = bins.request();
    py::buffer_info grad_info = gradients.request();
    py::buffer_info hess_info = hessians.request();
    py::buffer_info era_info = era_inverse.request();

    if (bins_info.ndim != 2) {
        throw std::invalid_argument("bins must be 2D");
    }
    check_contiguous(bins_info, "bins");
    check_contiguous(grad_info, "gradients");
    check_contiguous(hess_info, "hessians");
    check_contiguous(era_info, "era_inverse");

    const std::size_t n_rows = static_cast<std::size_t>(bins_info.shape[0]);
    const std::size_t n_features = static_cast<std::size_t>(bins_info.shape[1]);

    auto buffers = build_histograms_cpu(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        static_cast<int16_t*>(era_info.ptr),
        n_rows,
        n_features,
        max_bins,
        n_eras);

    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    return py::make_tuple(
        array_from_vector_float(buffers.grad, shape),
        array_from_vector_float(buffers.hess, shape),
        array_from_vector_int(buffers.count, shape));
}

HistogramBuffers build_frontier_histograms_cpu(
    const uint8_t* bins,
    const int32_t* node_indices,
    const int32_t* node_offsets,
    const int32_t* feature_subset,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    std::size_t n_features_subset,
    int max_bins,
    int n_eras) {

    const std::size_t feature_stride = static_cast<std::size_t>(max_bins) * n_eras;
    const std::size_t node_stride = n_features_subset * feature_stride;
    HistogramBuffers buffers;
    buffers.grad.assign(n_nodes * node_stride, 0.0f);
    buffers.hess.assign(n_nodes * node_stride, 0.0f);
    buffers.count.assign(n_nodes * node_stride, 0);

#pragma omp parallel for schedule(static)
    for (std::size_t node = 0; node < n_nodes; ++node) {
        const int32_t offset_begin = node_offsets[node];
        const int32_t offset_end = node_offsets[node + 1];
        float* grad_base = buffers.grad.data() + node * node_stride;
        float* hess_base = buffers.hess.data() + node * node_stride;
        int32_t* count_base = buffers.count.data() + node * node_stride;

        for (int32_t idx = offset_begin; idx < offset_end; ++idx) {
            const int32_t row = node_indices[idx];
            if (row < 0 || static_cast<std::size_t>(row) >= n_rows) {
                continue;
            }
            const int era = static_cast<int>(era_inverse[row]);
            if (era < 0 || era >= n_eras) {
                continue;
            }
            const uint8_t* row_bins = bins + static_cast<std::size_t>(row) * n_features_total;
            const float grad = gradients[row];
            const float hess = hessians[row];
            const std::size_t era_offset = static_cast<std::size_t>(era);

            for (std::size_t f = 0; f < n_features_subset; ++f) {
                const int32_t feature = feature_subset[f];
                if (feature < 0 || static_cast<std::size_t>(feature) >= n_features_total) {
                    continue;
                }
                const uint8_t bin = row_bins[feature];
                if (bin >= max_bins) {
                    continue;
                }
                const std::size_t base = f * feature_stride + static_cast<std::size_t>(bin) * n_eras + era_offset;
                grad_base[base] += grad;
                hess_base[base] += hess;
                count_base[base] += 1;
            }
        }
    }

    return buffers;
}

FrontierEvalResult evaluate_frontier_cpu(
    const uint8_t* bins,
    const int32_t* node_indices,
    const int32_t* node_offsets,
    const int32_t* feature_subset,
    const float* gradients,
    const float* hessians,
    const int16_t* era_inverse,
    std::size_t n_rows,
    std::size_t n_features_total,
    std::size_t n_nodes,
    std::size_t n_features_subset,
    int max_bins,
    int n_eras,
    double lambda_l2,
    double lambda_dro,
    int min_samples_leaf,
    double direction_weight) {

    HistogramBuffers hist = build_frontier_histograms_cpu(
        bins,
        node_indices,
        node_offsets,
        feature_subset,
        gradients,
        hessians,
        era_inverse,
        n_rows,
        n_features_total,
        n_nodes,
        n_features_subset,
        max_bins,
        n_eras);

    const std::size_t feature_stride = static_cast<std::size_t>(max_bins) * n_eras;
    const std::size_t node_stride = n_features_subset * feature_stride;
    const float lambda = static_cast<float>(lambda_l2);
    const float lambda_dro_f = static_cast<float>(lambda_dro);
    const float direction_weight_f = static_cast<float>(direction_weight);
    const float eps = 1e-12f;

    FrontierEvalResult result;
    result.best_feature.assign(n_nodes, -1);
    result.best_threshold.assign(n_nodes, 0);
    result.score.assign(n_nodes, -std::numeric_limits<float>::infinity());
    result.agreement.assign(n_nodes, 0.0f);
    result.left_value.assign(n_nodes, 0.0f);
    result.right_value.assign(n_nodes, 0.0f);
    result.base_value.assign(n_nodes, 0.0f);
    result.left_offsets.assign(n_nodes + 1, 0);
    result.right_offsets.assign(n_nodes + 1, 0);

    std::vector<int32_t> left_counts(n_nodes, 0);
    std::vector<int32_t> right_counts(n_nodes, 0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<float> total_grad(n_eras, 0.0f);
        std::vector<float> total_hess(n_eras, 0.0f);
        std::vector<int32_t> total_count(n_eras, 0);
        std::vector<float> left_grad(n_eras, 0.0f);
        std::vector<float> left_hess(n_eras, 0.0f);
        std::vector<int32_t> left_count_vec(n_eras, 0);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (std::size_t node = 0; node < n_nodes; ++node) {
            const int32_t offset_begin = node_offsets[node];
            const int32_t offset_end = node_offsets[node + 1];
            const int32_t node_size = offset_end - offset_begin;

            float node_grad_sum = 0.0f;
            float node_hess_sum = 0.0f;
            for (int32_t idx = offset_begin; idx < offset_end; ++idx) {
                const int32_t row = node_indices[idx];
                if (row < 0 || static_cast<std::size_t>(row) >= n_rows) {
                    continue;
                }
                node_grad_sum += gradients[row];
                node_hess_sum += hessians[row];
            }

            float node_base_value = 0.0f;
            if (node_size > 0) {
                node_base_value = -node_grad_sum / (node_hess_sum + lambda + eps);
            }

            float node_best_score = -std::numeric_limits<float>::infinity();
            int32_t node_best_feature = -1;
            int32_t node_best_threshold = 0;
            float node_best_agreement = 0.0f;
            float node_best_left_value = node_base_value;
            float node_best_right_value = node_base_value;
            int32_t node_best_left_total = node_size;
            int32_t node_best_right_total = 0;

            result.base_value[node] = node_base_value;

            if (n_features_subset > 0 && node_size > 0) {
                float* grad_base = hist.grad.data() + node * node_stride;
                float* hess_base = hist.hess.data() + node * node_stride;
                int32_t* count_base = hist.count.data() + node * node_stride;

                for (std::size_t f = 0; f < n_features_subset; ++f) {
                    float* grad_feat = grad_base + f * feature_stride;
                    float* hess_feat = hess_base + f * feature_stride;
                    int32_t* count_feat = count_base + f * feature_stride;

                    std::fill(total_grad.begin(), total_grad.end(), 0.0f);
                    std::fill(total_hess.begin(), total_hess.end(), 0.0f);
                    std::fill(total_count.begin(), total_count.end(), 0);

                    for (int bin = 0; bin < max_bins; ++bin) {
                        const std::size_t bin_offset = static_cast<std::size_t>(bin) * n_eras;
                        for (int era = 0; era < n_eras; ++era) {
                            const std::size_t idx = bin_offset + static_cast<std::size_t>(era);
                            total_grad[era] += grad_feat[idx];
                            total_hess[era] += hess_feat[idx];
                            total_count[era] += count_feat[idx];
                        }
                    }

                    int total_count_sum = 0;
                    for (int era = 0; era < n_eras; ++era) {
                        total_count_sum += total_count[era];
                    }

                    if (total_count_sum < min_samples_leaf) {
                        continue;
                    }

                    std::fill(left_grad.begin(), left_grad.end(), 0.0f);
                    std::fill(left_hess.begin(), left_hess.end(), 0.0f);
                    std::fill(left_count_vec.begin(), left_count_vec.end(), 0);

                    for (int threshold = 0; threshold < max_bins - 1; ++threshold) {
                        const std::size_t bin_offset = static_cast<std::size_t>(threshold) * n_eras;
                        for (int era = 0; era < n_eras; ++era) {
                            const std::size_t idx = bin_offset + static_cast<std::size_t>(era);
                            left_grad[era] += grad_feat[idx];
                            left_hess[era] += hess_feat[idx];
                            left_count_vec[era] += count_feat[idx];
                        }

                        int left_total = 0;
                        for (int era = 0; era < n_eras; ++era) {
                            left_total += left_count_vec[era];
                        }
                        const int right_total = total_count_sum - left_total;

                        if (left_total < min_samples_leaf || right_total < min_samples_leaf) {
                            continue;
                        }

                        double sum_gain = 0.0;
                        double sum_square = 0.0;
                        int direction_sum = 0;

                        for (int era = 0; era < n_eras; ++era) {
                            const float lg = left_grad[era];
                            const float lh = left_hess[era];
                            const int lc = left_count_vec[era];
                            const float rg = total_grad[era] - lg;
                            const float rh = total_hess[era] - lh;
                            const int rc = total_count[era] - lc;

                            const float parent_gain = 0.5f * (total_grad[era] * total_grad[era]) /
                                                     (total_hess[era] + lambda + eps);
                            const float left_gain = (lc > 0)
                                ? 0.5f * (lg * lg) / (lh + lambda + eps)
                                : 0.0f;
                            const float right_gain = (rc > 0)
                                ? 0.5f * (rg * rg) / (rh + lambda + eps)
                                : 0.0f;
                            const float gain = left_gain + right_gain - parent_gain;

                            sum_gain += static_cast<double>(gain);
                            sum_square += static_cast<double>(gain) * static_cast<double>(gain);

                            const float left_value = (lc > 0) ? -lg / (lh + lambda + eps) : 0.0f;
                            const float right_value = (rc > 0) ? -rg / (rh + lambda + eps) : 0.0f;
                            direction_sum += (left_value >= right_value) ? 1 : -1;
                        }

                        const double mean_gain = sum_gain / static_cast<double>(n_eras);
                        double variance = sum_square / static_cast<double>(n_eras) - mean_gain * mean_gain;
                        if (variance < 0.0) {
                            variance = 0.0;
                        }
                        const float std_gain = static_cast<float>(std::sqrt(variance));
                        const float dro_score = static_cast<float>(mean_gain) - lambda_dro_f * std_gain;
                        const float agreement = std::fabs(static_cast<float>(direction_sum)) /
                                                static_cast<float>(n_eras);
                        const float final_score = dro_score + direction_weight_f * agreement;

                        if (final_score > node_best_score) {
                            float best_left_grad_total = 0.0f;
                            float best_left_hess_total = 0.0f;
                            float best_right_grad_total = 0.0f;
                            float best_right_hess_total = 0.0f;

                            for (int era = 0; era < n_eras; ++era) {
                                best_left_grad_total += left_grad[era];
                                best_left_hess_total += left_hess[era];
                                best_right_grad_total += total_grad[era] - left_grad[era];
                                best_right_hess_total += total_hess[era] - left_hess[era];
                            }

                            node_best_score = final_score;
                            node_best_feature = feature_subset[f];
                            node_best_threshold = threshold;
                            node_best_agreement = agreement;
                            node_best_left_value = -best_left_grad_total / (best_left_hess_total + lambda + eps);
                            node_best_right_value = -best_right_grad_total / (best_right_hess_total + lambda + eps);
                            node_best_left_total = left_total;
                            node_best_right_total = right_total;
                        }
                    }
                }
            }

            if (!(node_best_score > 0.0f)) {
                node_best_feature = -1;
                node_best_threshold = 0;
                node_best_score = -std::numeric_limits<float>::infinity();
                node_best_agreement = 0.0f;
                node_best_left_value = node_base_value;
                node_best_right_value = node_base_value;
                node_best_left_total = node_size;
                node_best_right_total = 0;
            }

            result.best_feature[node] = node_best_feature;
            result.best_threshold[node] = node_best_threshold;
            result.score[node] = node_best_score;
            result.agreement[node] = node_best_agreement;
            result.left_value[node] = node_best_left_value;
            result.right_value[node] = node_best_right_value;
            left_counts[node] = node_best_left_total;
            right_counts[node] = node_best_right_total;
        }
    }

    for (std::size_t node = 0; node < n_nodes; ++node) {
        result.left_offsets[node + 1] = result.left_offsets[node] + left_counts[node];
        result.right_offsets[node + 1] = result.right_offsets[node] + right_counts[node];
    }

    result.left_indices.resize(static_cast<std::size_t>(result.left_offsets.back()));
    result.right_indices.resize(static_cast<std::size_t>(result.right_offsets.back()));

    for (std::size_t node = 0; node < n_nodes; ++node) {
        const int32_t offset_begin = node_offsets[node];
        const int32_t offset_end = node_offsets[node + 1];
        if (offset_begin >= offset_end) {
            continue;
        }

        int32_t left_pos = result.left_offsets[node];
        int32_t right_pos = result.right_offsets[node];

        const int32_t best_feature = result.best_feature[node];
        const float best_score = result.score[node];

        if (best_feature < 0 || !(best_score > 0.0f)) {
            for (int32_t idx = offset_begin; idx < offset_end; ++idx) {
                result.left_indices[static_cast<std::size_t>(left_pos++)] = node_indices[idx];
            }
            continue;
        }

        const int32_t threshold = result.best_threshold[node];

        for (int32_t idx = offset_begin; idx < offset_end; ++idx) {
            const int32_t row = node_indices[idx];
            if (row < 0 || static_cast<std::size_t>(row) >= n_rows) {
                continue;
            }
            const std::size_t row_offset = static_cast<std::size_t>(row) * n_features_total + static_cast<std::size_t>(best_feature);
            const uint8_t bin = bins[row_offset];
            if (bin <= threshold) {
                result.left_indices[static_cast<std::size_t>(left_pos++)] = row;
            } else {
                result.right_indices[static_cast<std::size_t>(right_pos++)] = row;
            }
        }
    }

    return result;
}

}  // namespace packboost

static py::tuple cpu_frontier_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_indices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature_subset,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> era_inverse,
    int max_bins,
    int n_eras) {

    py::buffer_info bins_info = bins.request();
    py::buffer_info idx_info = node_indices.request();
    py::buffer_info offsets_info = node_offsets.request();
    py::buffer_info feat_info = feature_subset.request();
    py::buffer_info grad_info = gradients.request();
    py::buffer_info hess_info = hessians.request();
    py::buffer_info era_info = era_inverse.request();

    if (bins_info.ndim != 2) {
        throw std::invalid_argument("bins must be 2D");
    }
    check_contiguous(bins_info, "bins");
    check_contiguous(idx_info, "node_indices");
    check_contiguous(offsets_info, "node_offsets");
    check_contiguous(feat_info, "feature_subset");
    check_contiguous(grad_info, "gradients");
    check_contiguous(hess_info, "hessians");
    check_contiguous(era_info, "era_inverse");

    const std::size_t n_rows = static_cast<std::size_t>(bins_info.shape[0]);
    const std::size_t n_features_total = static_cast<std::size_t>(bins_info.shape[1]);
    const std::size_t n_nodes = static_cast<std::size_t>(offsets_info.shape[0] - 1);
    const std::size_t n_features_subset = static_cast<std::size_t>(feat_info.shape[0]);

    auto buffers = packboost::build_frontier_histograms_cpu(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<int32_t*>(idx_info.ptr),
        static_cast<int32_t*>(offsets_info.ptr),
        static_cast<int32_t*>(feat_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        static_cast<int16_t*>(era_info.ptr),
        n_rows,
        n_features_total,
        n_nodes,
        n_features_subset,
        max_bins,
        n_eras);

    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_nodes),
        static_cast<py::ssize_t>(n_features_subset),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };

    return py::make_tuple(
        array_from_vector_float(buffers.grad, shape),
        array_from_vector_float(buffers.hess, shape),
        array_from_vector_int(buffers.count, shape));
}

static py::tuple cpu_frontier_evaluate_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_indices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature_subset,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> era_inverse,
    int max_bins,
    int n_eras,
    double lambda_l2,
    double lambda_dro,
    int min_samples_leaf,
    double direction_weight) {

    py::buffer_info bins_info = bins.request();
    py::buffer_info idx_info = node_indices.request();
    py::buffer_info offsets_info = node_offsets.request();
    py::buffer_info feat_info = feature_subset.request();
    py::buffer_info grad_info = gradients.request();
    py::buffer_info hess_info = hessians.request();
    py::buffer_info era_info = era_inverse.request();

    if (bins_info.ndim != 2) {
        throw std::invalid_argument("bins must be 2D");
    }
    check_contiguous(bins_info, "bins");
    check_contiguous(idx_info, "node_indices");
    check_contiguous(offsets_info, "node_offsets");
    check_contiguous(feat_info, "feature_subset");
    check_contiguous(grad_info, "gradients");
    check_contiguous(hess_info, "hessians");
    check_contiguous(era_info, "era_inverse");

    const std::size_t n_rows = static_cast<std::size_t>(bins_info.shape[0]);
    const std::size_t n_features_total = static_cast<std::size_t>(bins_info.shape[1]);
    const std::size_t n_nodes = static_cast<std::size_t>(offsets_info.shape[0] - 1);
    const std::size_t n_features_subset = static_cast<std::size_t>(feat_info.shape[0]);

    auto result = packboost::evaluate_frontier_cpu(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<int32_t*>(idx_info.ptr),
        static_cast<int32_t*>(offsets_info.ptr),
        static_cast<int32_t*>(feat_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        static_cast<int16_t*>(era_info.ptr),
        n_rows,
        n_features_total,
        n_nodes,
        n_features_subset,
        max_bins,
        n_eras,
        lambda_l2,
        lambda_dro,
        min_samples_leaf,
        direction_weight);

    std::vector<py::ssize_t> vec_shape = {static_cast<py::ssize_t>(n_nodes)};
    py::array_t<int32_t> feature_arr(vec_shape);
    py::array_t<int32_t> threshold_arr(vec_shape);
    py::array_t<float> score_arr(vec_shape);
    py::array_t<float> agreement_arr(vec_shape);
    py::array_t<float> left_value_arr(vec_shape);
    py::array_t<float> right_value_arr(vec_shape);
    py::array_t<float> base_value_arr(vec_shape);

    std::vector<py::ssize_t> offset_shape = {static_cast<py::ssize_t>(n_nodes + 1)};
    py::array_t<int32_t> left_offsets_arr(offset_shape);
    py::array_t<int32_t> right_offsets_arr(offset_shape);

    std::vector<py::ssize_t> left_idx_shape = {static_cast<py::ssize_t>(result.left_indices.size())};
    std::vector<py::ssize_t> right_idx_shape = {static_cast<py::ssize_t>(result.right_indices.size())};
    py::array_t<int32_t> left_indices_arr(left_idx_shape);
    py::array_t<int32_t> right_indices_arr(right_idx_shape);

    std::memcpy(feature_arr.mutable_data(), result.best_feature.data(), n_nodes * sizeof(int32_t));
    std::memcpy(threshold_arr.mutable_data(), result.best_threshold.data(), n_nodes * sizeof(int32_t));
    std::memcpy(score_arr.mutable_data(), result.score.data(), n_nodes * sizeof(float));
    std::memcpy(agreement_arr.mutable_data(), result.agreement.data(), n_nodes * sizeof(float));
    std::memcpy(left_value_arr.mutable_data(), result.left_value.data(), n_nodes * sizeof(float));
    std::memcpy(right_value_arr.mutable_data(), result.right_value.data(), n_nodes * sizeof(float));
    std::memcpy(base_value_arr.mutable_data(), result.base_value.data(), n_nodes * sizeof(float));
    std::memcpy(left_offsets_arr.mutable_data(), result.left_offsets.data(), (n_nodes + 1) * sizeof(int32_t));
    std::memcpy(right_offsets_arr.mutable_data(), result.right_offsets.data(), (n_nodes + 1) * sizeof(int32_t));
    if (!result.left_indices.empty()) {
        std::memcpy(left_indices_arr.mutable_data(), result.left_indices.data(), result.left_indices.size() * sizeof(int32_t));
    }
    if (!result.right_indices.empty()) {
        std::memcpy(right_indices_arr.mutable_data(), result.right_indices.data(), result.right_indices.size() * sizeof(int32_t));
    }

    return py::make_tuple(
        feature_arr,
        threshold_arr,
        score_arr,
        agreement_arr,
        left_value_arr,
        right_value_arr,
        base_value_arr,
        left_offsets_arr,
        right_offsets_arr,
        left_indices_arr,
        right_indices_arr);
}

// Forward declaration for the optional CUDA binding (implemented in backend_cuda.cu)
#ifdef PACKBOOST_ENABLE_CUDA
py::tuple cuda_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast>,
    int,
    int);

py::tuple cuda_frontier_evaluate_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast>,
    int,
    int,
    double,
    double,
    int,
    double);
#endif

PYBIND11_MODULE(_backend, m) {
    m.doc() = "PackBoost native histogram backends";
    m.def("cpu_histogram", &packboost::cpu_histogram_binding, "Build histograms on the CPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
    m.def("cpu_frontier_histogram", &cpu_frontier_histogram_binding, "Build histograms for a frontier of nodes",
          py::arg("bins"), py::arg("node_indices"), py::arg("node_offsets"), py::arg("feature_subset"),
          py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
    m.def("cpu_frontier_evaluate", &cpu_frontier_evaluate_binding, "Evaluate frontier nodes with DES scoring",
          py::arg("bins"), py::arg("node_indices"), py::arg("node_offsets"), py::arg("feature_subset"),
          py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"),
          py::arg("max_bins"), py::arg("n_eras"),
          py::arg("lambda_l2"), py::arg("lambda_dro"), py::arg("min_samples_leaf"), py::arg("direction_weight"));
#ifdef PACKBOOST_ENABLE_CUDA
    m.def("cuda_histogram", &cuda_histogram_binding, "Build histograms on the GPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
    m.def("cuda_frontier_evaluate", &cuda_frontier_evaluate_binding, "Evaluate frontier nodes with DES scoring on the GPU",
          py::arg("bins"), py::arg("node_indices"), py::arg("node_offsets"), py::arg("feature_subset"),
          py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"),
          py::arg("max_bins"), py::arg("n_eras"),
          py::arg("lambda_l2"), py::arg("lambda_dro"), py::arg("min_samples_leaf"), py::arg("direction_weight"));
#endif
}
