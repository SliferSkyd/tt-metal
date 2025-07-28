// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleDeviceOperation::program_factory_t ExampleDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiCore{};
}

void ExampleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void ExampleDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

ExampleDeviceOperation::spec_return_value_t ExampleDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.output_grad;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), MemoryConfig{}));
}

ExampleDeviceOperation::tensor_return_value_t ExampleDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.output_grad.device());
}

std::tuple<ExampleDeviceOperation::operation_attributes_t, ExampleDeviceOperation::tensor_args_t>
ExampleDeviceOperation::invoke(
    const Tensor& output_grad,
    std::variant<int64_t, float, double, bool> scalar,
    std::optional<Tensor> input,
    UnaryOpType op_type,
    const Tensor& input_grad) {
    return {
        operation_attributes_t{
            .op_type = op_type,
            .scalar1 = scalar,
        },
        tensor_args_t{
            .output_grad = output_grad,
            .input = input,
            .input_grad = input_grad,
        }};
}

}  // namespace ttnn::operations::examples
