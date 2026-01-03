#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "NeuralNet.h"

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pynn, m) {
    m.doc() = "pybind11 PyNN plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<std::vector<uint>, Scalar>())
        .def("propagateForward", &NeuralNetwork::propagateForward)
        .def("propagateBackward", &NeuralNetwork::propagateBackward)
        .def("train", &NeuralNetwork::trainMat)
        .def("calcErrors", &NeuralNetwork::calcErrors)
        .def("updateWeights", &NeuralNetwork::updateWeights);
}