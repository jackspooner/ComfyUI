#include "api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mesh_to_flexible_dual_grid", &mesh_to_flexible_dual_grid_cpu,
          pybind11::call_guard<pybind11::gil_scoped_release>());
}
