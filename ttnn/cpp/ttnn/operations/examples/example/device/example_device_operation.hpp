// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include "unary_op_types.hpp"

namespace ttnn::operations::examples {

struct ExampleDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        UnaryOpType op_type;
        std::optional<std::variant<int64_t, float, double, bool>> scalar1 = std::nullopt;
        std::optional<std::variant<int64_t, float, double, bool>> scalar2 = std::nullopt;
    };

    struct tensor_args_t {
        const Tensor& output_grad;
        std::optional<Tensor> input = std::nullopt;
        std::optional<Tensor> output = std::nullopt;
        const Tensor& input_grad;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;
    struct MultiCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output specs based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& output_grad,
        std::variant<int64_t, float, double, bool> scalar,
        std::optional<Tensor> input,
        UnaryOpType op_type,
        const Tensor& input_grad);
};

}  // namespace ttnn::operations::examples

// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
namespace ttnn::prim {
constexpr auto example =
    ttnn::register_operation<"ttnn::prim::example", ttnn::operations::examples::ExampleDeviceOperation>();
}  // namespace ttnn::prim
