#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_cpu_frontier(py::module_ &m);

#if defined(PACKBOOST_ENABLE_CUDA)
void register_cuda_frontier(py::module_ &m);
#endif

PYBIND11_MODULE(_backend, m) {
    m.doc() = "PackBoost native backend stubs for milestone 1";
    register_cpu_frontier(m);
#if defined(PACKBOOST_ENABLE_CUDA)
    register_cuda_frontier(m);
#else
    m.def(
        "_cuda_available",
        []() {
            return false;
        },
        "CUDA backend is unavailable until milestone 4.");
#endif
}
