#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_cuda_frontier(py::module_ &m) {
    m.def(
        "_cuda_available",
        []() {
            return false;
        },
        "Native CUDA frontier arrives in milestone 4.");
}
