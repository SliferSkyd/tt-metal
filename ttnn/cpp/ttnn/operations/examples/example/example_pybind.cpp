// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/examples/example/example.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

void bind_example_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::example,
        R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // The overload with `queue_id` argument will be added automatically for primitive operations
        // This specific function can be called from python as `ttnn.prim.example(input_tensor)` or
        // `ttnn.prim.example(input_tensor, queue_id=queue_id)`
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::example)& self,
               const Tensor& output_grad,
               std::variant<int64_t, float, double, bool> scalar,
               const Tensor& input,
               const Tensor& input_grad,
               const QueueId& queue_id) -> ttnn::Tensor {
                return self(queue_id, output_grad, scalar, input, input_grad);
            },
            py::arg("output_grad"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("input"),
            py::arg("input_grad"),
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::examples
