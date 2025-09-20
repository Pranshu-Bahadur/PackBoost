#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_cpu_frontier(py::module_ &m) {
    m.def(
        "_cpu_available",
        []() {
            return false;
        },
        "Native CPU frontier arrives in milestone 2.");
}
