#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "SKLEARN/regression/ElasticNetRegression.h"

namespace py = pybind11;

PYBIND11_MODULE(SKLEARN, m) {
    py::class_<ElasticNetRegression>(m, "ElasticNetRegression")
        .def(py::init<double, double>(), py::arg("l1_ratio") = 0.5, py::arg("alpha") = 1.0)
        .def("fit", &ElasticNetRegression::fit, py::arg("X"), py::arg("y"), py::arg("max_iter") = 1000, py::arg("tol") = 1e-4)
        .def("predict", &ElasticNetRegression::predict, py::arg("X"))
        .def("get_coefficients", &ElasticNetRegression::get_coefficients);
}
