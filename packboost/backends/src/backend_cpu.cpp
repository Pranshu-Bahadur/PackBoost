#include "backend.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
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
    const int32_t* era_inverse,
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

#pragma omp parallel for if(n_features > 1)
    for (std::size_t feature = 0; feature < n_features; ++feature) {
        const uint8_t* feature_bins = bins + feature;
        std::vector<float> thread_grad(feat_stride, 0.0f);
        std::vector<float> thread_hess(feat_stride, 0.0f);
        std::vector<int32_t> thread_count(feat_stride, 0);

        for (std::size_t row = 0; row < n_rows; ++row) {
            const int32_t era = era_inverse[row];
            if (era < 0 || era >= n_eras) {
                continue;
            }
            const uint8_t bin = feature_bins[row * n_features];
            if (bin >= max_bins) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(bin) * n_eras + era;
            thread_grad[idx] += gradients[row];
            thread_hess[idx] += hessians[row];
            thread_count[idx] += 1;
        }

        float* grad_out = buffers.grad.data() + feature * feat_stride;
        float* hess_out = buffers.hess.data() + feature * feat_stride;
        int32_t* count_out = buffers.count.data() + feature * feat_stride;

        for (std::size_t i = 0; i < feat_stride; ++i) {
            grad_out[i] += thread_grad[i];
            hess_out[i] += thread_hess[i];
            count_out[i] += thread_count[i];
        }
    }

    return buffers;
}

}  // namespace packboost


static void check_contiguous(const py::buffer_info& info, const char* name) {
    if (info.ndim == 0) {
        throw std::invalid_argument(std::string(name) + " must be at least 1-D");
    }
    if (info.strides.back() != info.itemsize) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
}

static py::array_t<float> make_array_float(std::vector<float>&& data, std::size_t n_features, int max_bins, int n_eras) {
    return py::array_t<float>({static_cast<ssize_t>(n_features), max_bins, n_eras}, std::move(data));
}

static py::array_t<int32_t> make_array_int(std::vector<int32_t>&& data, std::size_t n_features, int max_bins, int n_eras) {
    return py::array_t<int32_t>({static_cast<ssize_t>(n_features), max_bins, n_eras}, std::move(data));
}

static py::tuple cpu_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bins,
    py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<float, py::array::c_style | py::array::forcecast> hessians,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> era_inverse,
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
        static_cast<int32_t*>(era_info.ptr),
        n_rows,
        n_features,
        max_bins,
        n_eras);

    return py::make_tuple(
        make_array_float(std::move(buffers.grad), n_features, max_bins, n_eras),
        make_array_float(std::move(buffers.hess), n_features, max_bins, n_eras),
        make_array_int(std::move(buffers.count), n_features, max_bins, n_eras));
}

// Forward declaration for the optional CUDA binding (implemented in backend_cuda.cu)
#ifdef PACKBOOST_ENABLE_CUDA
py::tuple cuda_histogram_binding(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<float, py::array::c_style | py::array::forcecast>,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
    int,
    int);
#endif

PYBIND11_MODULE(_backend, m) {
    m.doc() = "PackBoost native histogram backends";
    m.def("cpu_histogram", &cpu_histogram_binding, "Build histograms on the CPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
#ifdef PACKBOOST_ENABLE_CUDA
    m.def("cuda_histogram", &cuda_histogram_binding, "Build histograms on the GPU",
          py::arg("bins"), py::arg("gradients"), py::arg("hessians"), py::arg("era_inverse"), py::arg("max_bins"), py::arg("n_eras"));
#endif
}
