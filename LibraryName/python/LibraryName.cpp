#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LibraryName/regression/ElasticNetRegression.h"
#include "LibraryName/regression/LinearRegression.h"
// Add includes for other algorithms as you implement them

namespace py = pybind11;

PYBIND11_MODULE(LibraryName, m) {
    py::class_<ElasticNetRegression>(m, "ElasticNetRegression")
        .def(py::init<double, double>(), py::arg("l1_ratio") = 0.5, py::arg("alpha") = 1.0)
        .def("fit", &ElasticNetRegression::fit, py::arg("X"), py::arg("y"), py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("predict", &ElasticNetRegression::predict, py::arg("X"))
        .def("get_coefficients", &ElasticNetRegression::get_coefficients);

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit, py::arg("X"), py::arg("y"))
        .def("predict", &LinearRegression::predict, py::arg("X"))
        .def("get_coefficients", &LinearRegression::get_coefficients);

    // Add bindings for other algorithms as you implement them
}

