#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "CPU/CPUGraph.hpp"
#include "MLIR/MLIRGraph.hpp"
#include "MLIR/Node.hpp"
#include "Tensor.h"
#include "CPU/CPUOp.h"
#include "CPU/Optimizer.hpp"


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
    // 2. Bind Tensor with 2D NumPy Interoperability
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
    // 3.0. Create submodule for cpu based features
    // =========================================================================
    py::module_ cpum = m.def_submodule("cpu", "CPU Based features module");
    
    // =========================================================================
    // 3.1. Bind Init Enum (assuming it exists)
    // =========================================================================
    py::enum_<cpu::Init>(cpum, "Init")
        .value("He", cpu::Init::He)
        .value("Xavier", cpu::Init::Xavier)
        .value("Zero", cpu::Init::Zero)
        .export_values();

    // =========================================================================
    // 3.2. Bind Graph Class
    // =========================================================================
    py::class_<cpu::Graph>(cpum, "CPUGraph")
        .def(py::init<unsigned>(), py::arg("seed") = 42)
        .def("input", &cpu::Graph::input, py::arg("s"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("param", &cpu::Graph::param, py::arg("s"), py::arg("init"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("forward", &cpu::Graph::forward)
        .def("backward", &cpu::Graph::backward, py::arg("loss"))
        .def("set_input_shape", &cpu::Graph::setInputShape)
        .def("reshape", &cpu::Graph::reshape)
        .def("num_nodes", &cpu::Graph::numNodes)
        .def("print", &cpu::Graph::print)
        .def("value", [](cpu::Graph &g, size_t id) -> Tensor& {
            return g.value(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("grad", [](cpu::Graph &g, size_t id) -> Tensor& {
            return g.grad(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("param_pairs", &cpu::Graph::paramPairs, py::return_value_policy::reference_internal);

    // =========================================================================
    // 3.3. Bind Layer Helpers
    // =========================================================================
    cpum.def("linear", &cpu::linear, py::arg("g"), py::arg("x"), py::arg("out"), py::arg("init") = cpu::Init::He);
    cpum.def("relu", &cpu::relu, py::arg("g"), py::arg("x"));
    cpum.def("sigmoid", &cpu::sigmoid, py::arg("g"), py::arg("x"));
    cpum.def("tanh", &cpu::tanh_, py::arg("g"), py::arg("x"));
    cpum.def("mse_loss", &cpu::mseLoss, py::arg("g"), py::arg("pred"), py::arg("target"));

    // =========================================================================
    // 3.4. Bind Optimizers
    // =========================================================================
    py::class_<cpu::Optimizer>(cpum, "Optimizer");
    
    py::class_<cpu::SGD, cpu::Optimizer>(cpum, "SGD")
        .def(py::init<Scalar, Scalar>(), py::arg("lr"), py::arg("momentum") = 0.0)
        .def("step", &cpu::SGD::step);
    
    // =========================================================================
    // 4.0. Create submodule for cpu based features
    // =========================================================================
    py::module_ mlirm = m.def_submodule("mlir", "MLIR Based features module");
    
    // =========================================================================
    // 4.1. Bind MLIRInit Enum (assuming it exists)
    // =========================================================================
    py::enum_<mlg::MLIRInit>(mlirm, "MLIRInit")
        .value("He", mlg::MLIRInit::He)
        .value("Xavier", mlg::MLIRInit::Xavier)
        .value("Zero", mlg::MLIRInit::Zero)
        .export_values();

    // =========================================================================
    // 4.1. Bind MLIRGraph 
    // =========================================================================
    py::class_<mlg::Graph>(mlirm, "MLIRGraph")
        .def(py::init<unsigned>(), py::arg("seed") = 42)
        .def("input", &mlg::Graph::input, py::arg("s"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("param", &mlg::Graph::param, py::arg("s"), py::arg("init"), py::arg("label") = "",
             py::return_value_policy::reference_internal)
        .def("finalizeForward", &mlg::Graph::finalizeForward, py::arg("outputNode"))
        .def("forward", &mlg::Graph::forward)
        .def("backward", &mlg::Graph::backward, py::arg("loss"))
        .def("set_input_shape", &mlg::Graph::setInputShape)
        .def("reshape", &mlg::Graph::reshape)
        .def("num_nodes", &mlg::Graph::numNodes)
        .def("print", &mlg::Graph::print)
        .def("dump", &mlg::Graph::dump)
        .def("value", [](mlg::Graph &g, size_t id) -> Tensor& {
            return g.value(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("grad", [](mlg::Graph &g, size_t id) -> Tensor& {
            return g.grad(id);
        }, py::return_value_policy::reference_internal, py::arg("id"))
        .def("param_pairs", &mlg::Graph::paramPairs, py::return_value_policy::reference_internal);

    // =========================================================================
    // 4.2. Bind Layer Helpers
    // =========================================================================
    mlirm.def("linear", &mlg::linear, py::arg("g"), py::arg("x"), py::arg("out"), py::arg("init") = mlg::MLIRInit::He);
    mlirm.def("relu", &mlg::relu, py::arg("g"), py::arg("x"));
    // mlirm.def("sigmoid", &mlg::sigmoid, py::arg("g"), py::arg("x"));
    // mlirm.def("tanh", &mlg::tanh_, py::arg("g"), py::arg("x"));
    mlirm.def("mse_loss", &mlg::mseLoss, py::arg("g"), py::arg("pred"), py::arg("target"));
}