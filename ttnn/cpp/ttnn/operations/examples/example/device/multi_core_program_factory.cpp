// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "cb_config.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <filesystem>

#include "tt-metalium/tt_backend_api_types.hpp"

namespace ttnn::operations::examples {
using tt::DataFormat;
using tt::constants::FACE_WIDTH;
using tt::constants::TILE_HW;
using tt::constants::TILE_WIDTH;

using namespace tt::tt_metal;
using namespace tt;
using namespace tt::tt_metal::detail;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace {
struct CircularBufferArg {
    std::variant<tt::CBIndex, ttsl::SmallVector<tt::CBIndex>> cb_index;
    uint32_t num_tiles = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    const Buffer* buffer = nullptr;

    CircularBufferArg(
        tt::CBIndex cb_index,
        uint32_t num_tiles,
        tt::DataFormat data_format = tt::DataFormat::Invalid,
        const Buffer* buffer = nullptr) :
        cb_index(cb_index), num_tiles(num_tiles), data_format(data_format), buffer(buffer) {}

    CircularBufferArg(
        std::initializer_list<tt::CBIndex> cb_indices,
        uint32_t num_tiles,
        tt::DataFormat data_format = tt::DataFormat::Invalid,
        const Buffer* buffer = nullptr) :
        cb_index(ttsl::SmallVector<tt::CBIndex>(cb_indices)),
        num_tiles(num_tiles),
        data_format(data_format),
        buffer(buffer) {}
};

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_range,
    tt::DataFormat data_format,
    const CircularBufferArg& arg) {
    auto data_format_to_use = (arg.data_format != tt::DataFormat::Invalid) ? arg.data_format : data_format;

    auto data_format_size = TileSize(data_format_to_use);

    ttnn::SmallVector<tt::CBIndex> cb_indices;
    if (std::holds_alternative<tt::CBIndex>(arg.cb_index)) {
        auto cb_index = std::get<tt::CBIndex>(arg.cb_index);
        cb_indices.push_back(cb_index);
    } else {
        cb_indices = std::get<ttnn::SmallVector<tt::CBIndex>>(arg.cb_index);
    }

    std::map<uint8_t, tt::DataFormat> data_format_spec;
    for (auto cb_index : cb_indices) {
        data_format_spec.insert({cb_index, data_format_to_use});
    }
    CircularBufferConfig cb_config(arg.num_tiles * data_format_size, data_format_spec);
    for (auto cb_index : cb_indices) {
        cb_config.set_page_size(cb_index, data_format_size);
    }

    if (arg.buffer) {
        cb_config.set_globally_allocated_address(*arg.buffer);
    }

    return CreateCircularBuffer(program, core_range, cb_config);
}

std::vector<CBHandle> CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_range,
    tt::DataFormat data_format,
    const std::vector<CircularBufferArg>& args) {
    std::vector<CBHandle> cb_ids{};
    CBHandle cb_id{};
    for (auto arg : args) {
        cb_id = CreateCircularBuffer(program, core_range, data_format, arg);
        cb_ids.push_back(cb_id);
    }
    return cb_ids;
}

DataFormat get_data_format(const Tensor& tensor) { return datatype_to_dataformat_converter(tensor.dtype()); }

DataFormat get_data_format(const std::optional<Tensor>& tensor, DataFormat default_format = tt::DataFormat::Invalid) {
    return tensor.has_value() ? datatype_to_dataformat_converter(tensor.value().dtype()) : default_format;
}
inline uint32_t is_dram(const Tensor& tensor) { return tensor.memory_config().is_dram(); }
inline uint32_t is_dram(const std::optional<Tensor>& tensor) {
    return tensor.has_value() ? is_dram(tensor.value()) : 0;
}

uint32_t get_Wt(const Tensor& tensor) {
    auto shape = tensor.logical_shape();
    return div_up(shape[-1], TILE_WIDTH);
}

inline uint32_t buffer_addr(const Tensor& tensor) { return tensor.buffer()->address(); }
inline uint32_t buffer_addr(const std::optional<Tensor>& tensor) {
    return tensor.has_value() ? buffer_addr(tensor.value()) : 0xFFFF'FFFF;
}

uint32_t reinterpret_as_uint32_backward(const std::variant<int64_t, float, double, bool>& scalar, UnaryOpType op_type) {
    union Converter {
        float f;
        int32_t i;
        uint32_t u;
    };

    switch (op_type) {
        case UnaryOpType::POW_EXP_FLOAT: {
            Converter c{.f = std::visit([](auto v) -> float { return static_cast<float>(v); }, scalar)};
            return c.u;
        } break;
        default: TT_THROW("Invalid UnaryOpType");
    }
}

[[maybe_unused]] KernelHandle CreateReadKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    std::map<std::string, std::string> defines) {
    return CreateKernel(
        program,
        file_name,
        core_spec,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(hal::get_arch()),
            .compile_args = compile_args,
            .defines = defines});
}

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    std::map<std::string, std::string> defines) {
    return CreateKernel(
        program,
        file_name,
        core_spec,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(hal::get_arch()),
            .compile_args = compile_args,
            .defines = defines});
}
}  // namespace

ExampleDeviceOperation::MultiCore::cached_program_t ExampleDeviceOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};
    namespace cb = cb_backward;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    auto& input_grad = tensor_return_value;

    const auto op_type = operation_attributes.op_type;
    const auto& scalar1 = operation_attributes.scalar1;
    const auto& scalar2 = operation_attributes.scalar2;

    const auto shape = output_grad.logical_shape();
    const uint32_t num_tiles = output_grad.physical_volume() / TILE_HW;
    const auto output_grad_data_format = get_data_format(output_grad);
    const auto input_data_format = get_data_format(input);
    const auto output_data_format = get_data_format(output);
    const auto input_grad_data_format = get_data_format(input_grad);
    const auto cb_mask_w_data_format = DataFormat::Float16_b;

    const bool do_mask_w = (input_grad_data_format == DataFormat::Bfp8_b) && shape[-1] % FACE_WIDTH != 0;
    const uint32_t mask_w = do_mask_w ? shape[-1] % TILE_WIDTH : TILE_WIDTH;
    const uint32_t Wt = get_Wt(input_grad);

    auto device = output_grad.device();
    auto arch = device->arch();
    const auto grid = device->compute_with_storage_grid_size();

    auto fp32_dest_acc_en = true;

    uint32_t num_cores, num_tiles_per_core[2];
    CoreRangeSet all_cores, core_groups[2];
    std::tie(num_cores, all_cores, core_groups[0], core_groups[1], num_tiles_per_core[0], num_tiles_per_core[1]) =
        split_work_to_cores(grid, num_tiles);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    uint32_t num_cores_group_1 = core_groups[0].num_cores();

    const uint32_t num_input_cbs = 1 /* output_grad */ + (input.has_value() ? 1 : 0) + (output.has_value() ? 1 : 0);
    uint32_t max_block_size = 8 / num_input_cbs;
    if (fp32_dest_acc_en) {
        max_block_size = 4 / num_input_cbs;
    }
    if (do_mask_w) {
        max_block_size = 1;
    }

    const uint32_t cb_num_tiles = 2 * max_block_size;
    CreateCircularBuffer(
        program,
        all_cores,
        output_grad_data_format,
        {
            {cb::output_grad, cb_num_tiles},
            {cb::input, input.has_value() ? cb_num_tiles : 0, input_data_format},
            {cb::output, output.has_value() ? cb_num_tiles : 0, output_data_format},
            {cb::mask_w, do_mask_w ? 1 : 0, cb_mask_w_data_format},
            {cb::input_grad, cb_num_tiles, input_grad_data_format},
        });

    std::vector<uint32_t> reader_ctas{
        is_dram(output_grad),
        is_dram(input),
        is_dram(output),
    };
    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> writer_ctas{
        is_dram(input_grad),
    };
    std::map<std::string, std::string> writer_defines;

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/reader_unary.cpp";
    auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_ctas, reader_defines);

    // Writer
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/writer_unary.cpp";
    auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_ctas, writer_defines);

    std::vector<uint32_t> compute_ctas{
        static_cast<uint32_t>(op_type),
        max_block_size,
    };
    std::map<std::string, std::string> compute_defines;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        if (output_grad_data_format == DataFormat::Float32) {
            unpack_to_dest_mode[cb::output_grad] = UnpackToDestMode::UnpackToDestFp32;
        }
        if (input_data_format == DataFormat::Float32) {
            unpack_to_dest_mode[cb::input] = UnpackToDestMode::UnpackToDestFp32;
        }
        if (output_data_format == DataFormat::Float32) {
            unpack_to_dest_mode[cb::output] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/eltwise_sfpu.cpp";
    auto compute_kernel_id = CreateKernel(
        program,
        compute_kernel_file,
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = true,
            .compile_args = compute_ctas,
            .defines = compute_defines});

    ttsl::SmallVector<uint32_t> reader_crtas = {
        num_tiles_per_core[0],
        num_tiles_per_core[1],
        core_groups[0].num_cores(),
        core_groups[1].num_cores(),
        grid.y,
        buffer_addr(output_grad),
        buffer_addr(input),
        buffer_addr(output),
        static_cast<uint32_t>(input.has_value()),
        static_cast<uint32_t>(output.has_value()),
        do_mask_w,
        mask_w,
    };
    ttsl::SmallVector<uint32_t> writer_crtas = {
        num_tiles_per_core[0],
        num_tiles_per_core[1],
        core_groups[0].num_cores(),
        core_groups[1].num_cores(),
        grid.y,
        buffer_addr(input_grad),
        cb::input_grad,
    };
    SetCommonRuntimeArgs(program, reader_kernel_id, reader_crtas);
    SetCommonRuntimeArgs(program, writer_kernel_id, writer_crtas);
    SetCommonRuntimeArgs(
        program,
        compute_kernel_id,
        {
            num_tiles_per_core[0],
            num_tiles_per_core[1],
            core_groups[0].num_cores(),
            core_groups[1].num_cores(),
            grid.y,
            static_cast<uint32_t>(input.has_value()),
            static_cast<uint32_t>(output.has_value()),
            scalar1.has_value() ? reinterpret_as_uint32_backward(scalar1.value(), UnaryOpType::POW_EXP_FLOAT)
                                : 0xFFFF'FFFF,
            0xFFFF'FFFF,
            do_mask_w,
            Wt,
        });

    return {
        std::move(program), {.unary_reader_kernel_id = reader_kernel_id, .unary_writer_kernel_id = writer_kernel_id}};
}

void ExampleDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::examples
