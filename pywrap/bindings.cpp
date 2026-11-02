#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Graph.h"
#include "Tensor.h"
#include "Operator.h"

class PGraph {
    Graph g;
    public:
    PGraph(unsigned int seed = 42): g(42) {}
};

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pynn, m) {
    m.doc() = "C++ neural network engine python bindings"; // optional module docstring
    
    // =========================================================================
    // 1. Bind Shape Struct
    // =========================================================================
    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<size_t, size_t>(), py::arg("r"), py::arg("c"))
        .def_readwrite("r", &Shape::r)
        .def_readwrite("c", &Shape::c)
        .def("size", &Shape::size)
        .def("__repr__", [](const Shape &s) {
            return "<Shape r=" + std::to_string(s.r) + ", c=" + std::to_string(s.c) + ">";
        });

    // =========================================================================
    // 2. Bind Init Enum (assuming it exists)
    // =========================================================================
    py::enum_<Init>(m, "Init")
        .value("He", Init::He)
        .value("Xavier", Init::Xavier)
        .value("Zero", Init::Zero)
        .export_values();

    // =========================================================================
    // 3. Bind Tensor with 2D NumPy Interoperability
    // =========================================================================
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<Shape>(), py::arg("shape"))
        .def_readwrite("shape", &Tensor::s)
        // Convert C++ Tensor to a 2D NumPy array (zero-copy view)
        .def("to_numpy", [](const Tensor &t) {
            std::vector<ssize_t> shape_dims = {static_cast<ssize_t>(t.s.r), static_cast<ssize_t>(t.s.c)};
            return py::array_t<Scalar>(
                shape_dims, 
                t.v.data(), 
                py::cast(t) // Keeps the parent Tensor alive while NumPy array is in use
            );
        })
        // Copy a 2D NumPy array into the C++ Tensor
        .def("from_numpy", [](Tensor &t, py::array_t<Scalar, py::array::c_style | py::array::forcecast> arr) {
            auto buf = arr.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Expected a 2D NumPy array for Tensor assignment.");
            }
            
            Shape new_shape{static_cast<size_t>(buf.shape[0]), static_cast<size_t>(buf.shape[1])};
            
            // Resize tensor if dimensions changed
            if (new_shape != t.s) {
                t.resize(new_shape);
            }

            Scalar *ptr = static_cast<Scalar *>(buf.ptr);
            t.v.assign(ptr, ptr + arr.size());
        });

    // =========================================================================
    // 4. Bind Graph Class
    // =========================================================================
    py::class_<Graph>(m, "Graph")
        .def(py::init<unsigned>(), py::arg("seed") = 42)
        .def("input", &Graph::input, py::arg("s"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("param", &Graph::param, py::arg("s"), py::arg("init"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward, py::arg("loss"))
        .def("set_input_shape", &Graph::setInputShape)
        .def("reshape", &Graph::reshape)
        .def("num_nodes", &Graph::numNodes)
        .def("print", &Graph::print)
        .def("value", [](Graph &g, size_t id) -> Tensor& {
            return g.value(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("grad", [](Graph &g, size_t id) -> Tensor& {
            return g.grad(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("param_pairs", &Graph::paramPairs, py::return_value_policy::reference_internal);

    // =========================================================================
    // 5. Bind Layer Helpers
    // =========================================================================
    m.def("linear", &linear, py::arg("g"), py::arg("x"), py::arg("out"), py::arg("init") = Init::He);
    m.def("relu", &relu, py::arg("g"), py::arg("x"));
    m.def("sigmoid", &sigmoid, py::arg("g"), py::arg("x"));
    m.def("tanh", &tanh_, py::arg("g"), py::arg("x"));
    m.def("mse_loss", &mseLoss, py::arg("g"), py::arg("pred"), py::arg("target"));

    // =========================================================================
    // 6. Bind Optimizers
    // =========================================================================
    py::class_<Optimizer>(m, "Optimizer");
    
    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init<Scalar, Scalar>(), py::arg("lr"), py::arg("momentum") = 0.0)
        .def("step", &SGD::step);
}