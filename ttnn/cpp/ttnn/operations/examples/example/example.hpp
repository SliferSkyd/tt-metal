
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"
#include "device/unary_op_types.hpp"

namespace ttnn::operations::examples {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct ExampleOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& grad,
        std::variant<int64_t, float, double, bool> scalar,
        const std::optional<Tensor>& input,
        const Tensor& input_grad) {
        return prim::example(grad, scalar, input, UnaryOpType::POW_EXP_FLOAT, input_grad);
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto example = ttnn::register_operation<"ttnn::example", operations::examples::ExampleOperation>();
}  // namespace ttnn
