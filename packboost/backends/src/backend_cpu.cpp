#include "backend.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;

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


static void check_contiguous(const py::buffer_info& info, const char* name) {
    if (info.ndim == 0) {
        throw std::invalid_argument(std::string(name) + " must be at least 1-D");
    }
    if (info.strides.back() != info.itemsize) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
}

static py::array_t<float> make_array_float(const std::vector<float>& data, std::size_t n_features, int max_bins, int n_eras) {
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    py::array_t<float> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    return arr;
}

static py::array_t<int32_t> make_array_int(const std::vector<int32_t>& data, std::size_t n_features, int max_bins, int n_eras) {
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    py::array_t<int32_t> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(int32_t));
    return arr;
}

static py::array_t<float> make_array_float(const std::vector<float>& data, std::size_t n_nodes, std::size_t n_features, int max_bins, int n_eras) {
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_nodes),
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    py::array_t<float> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    return arr;
}

static py::array_t<int32_t> make_array_int(const std::vector<int32_t>& data, std::size_t n_nodes, std::size_t n_features, int max_bins, int n_eras) {
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(n_nodes),
        static_cast<py::ssize_t>(n_features),
        static_cast<py::ssize_t>(max_bins),
        static_cast<py::ssize_t>(n_eras),
    };
    py::array_t<int32_t> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(int32_t));
    return arr;
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

    auto buffers = packboost::build_histograms_cpu(
        static_cast<uint8_t*>(bins_info.ptr),
        static_cast<float*>(grad_info.ptr),
        static_cast<float*>(hess_info.ptr),
        static_cast<int16_t*>(era_info.ptr),
        n_rows,
        n_features,
        max_bins,
        n_eras);

    return py::make_tuple(
        make_array_float(buffers.grad, n_features, max_bins, n_eras),
        make_array_float(buffers.hess, n_features, max_bins, n_eras),
        make_array_int(buffers.count, n_features, max_bins, n_eras));
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
    packboost::check_contiguous(bins_info, "bins");
    packboost::check_contiguous(idx_info, "node_indices");
    packboost::check_contiguous(offsets_info, "node_offsets");
    packboost::check_contiguous(feat_info, "feature_subset");
    packboost::check_contiguous(grad_info, "gradients");
    packboost::check_contiguous(hess_info, "hessians");
    packboost::check_contiguous(era_info, "era_inverse");

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

    return py::make_tuple(
        packboost::make_array_float(buffers.grad, n_nodes, n_features_subset, max_bins, n_eras),
        packboost::make_array_float(buffers.hess, n_nodes, n_features_subset, max_bins, n_eras),
        packboost::make_array_int(buffers.count, n_nodes, n_features_subset, max_bins, n_eras));
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
#endif

PYBIND11_MODULE(_backend, m) {
    m.doc() = "PackBoost native histogram backends";
    m.def("cpu_histogram", &packboost::cpu_histogram_binding, "Build histograms on the CPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
    m.def("cpu_frontier_histogram", &cpu_frontier_histogram_binding, "Build histograms for a frontier of nodes",
          py::arg("bins"), py::arg("node_indices"), py::arg("node_offsets"), py::arg("feature_subset"),
          py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
#ifdef PACKBOOST_ENABLE_CUDA
    m.def("cuda_histogram", &cuda_histogram_binding, "Build histograms on the GPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
#endif
}
