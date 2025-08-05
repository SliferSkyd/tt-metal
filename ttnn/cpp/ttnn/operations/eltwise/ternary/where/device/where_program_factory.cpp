// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "tt-metalium/core_coord.hpp"
#include <cmath>
#include <algorithm>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

template <typename F>
void set_or_update_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const WhereDeviceOperation::operation_attributes_t& operation_attributes,
    const WhereDeviceOperation::tensor_args_t& tensor_args,
    WhereDeviceOperation::tensor_return_value_t& output,
    F handle_args) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    WhereVariant variant = operation_attributes.where_variant;

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    // Adopt binary_ng work distribution strategy for better large tensor support
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Use the same logic as binary_ng for better scalability
    uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    std::vector<CoreCoord> cores;

    // For large tensors, use compute_with_storage_grid; for small ones, use all_device_cores
    if (num_output_tiles <= num_cores_total) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);
        cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, num_output_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }
    constexpr size_t num_writer_args = 3;
    constexpr size_t num_kernel_args = 1;

    // Reader args count depends on variant
    constexpr size_t num_reader_args_basic = 5;
    constexpr size_t num_reader_args_colbcast = 27;
    uint32_t dummy_arg = 0;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // Use correct number of reader args based on variant
            if (variant == WhereVariant::TTT_COL) {
                handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args_colbcast>{0});
            } else {
                handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args_basic>{0});
            }
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, num_writer_args>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, num_kernel_args>{0});
            continue;
        }

        // Set reader runtime arguments based on variant
        if (variant == WhereVariant::TTS) {
            // TTS: predicate (arg 0) + value_true tensor (arg 1)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor.value().buffer()->address(),
                dummy_arg,
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else if (variant == WhereVariant::TST) {
            // TST: predicate (arg 0) + value_false tensor (arg 1, maps to c_1)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_false_tensor.value().buffer()->address(),
                dummy_arg,
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        }

        // For TTT_COL, declare variables outside the if blocks so they can be used by both reader and writer
        uint32_t Wt = 0, Ht = 0, HtWt = 0, C = 0, N = 0, D = 0, cND = 0;
        uint32_t nD_stride_0 = 0, d_stride_0 = 0, n_stride_0 = 0, c_stride_0 = 0;
        uint32_t nD_stride_1 = 0, d_stride_1 = 0, n_stride_1 = 0, c_stride_1 = 0;
        uint32_t nD_stride_2 = 0, d_stride_2 = 0, n_stride_2 = 0, c_stride_2 = 0;

        if (variant == WhereVariant::TTT_COL) {
            // TTT_COL: Column broadcast variant with complex stride arguments
            const auto& pred_shape = predicate_tensor.logical_shape();
            const auto& true_shape = value_true_tensor.value().logical_shape();
            const auto& false_shape = value_false_tensor.value().logical_shape();

            // Use the largest tensor's shape for output dimensions
            auto output_shape = pred_shape;
            if (true_shape.volume() > output_shape.volume()) {
                output_shape = true_shape;
            }
            if (false_shape.volume() > output_shape.volume()) {
                output_shape = false_shape;
            }

            // Calculate tile dimensions using binary_ng pattern (padded_shape + tile division)
            const auto& padded_shape = output.padded_shape();
            const auto& tile = output.tensor_spec().tile();
            Wt = padded_shape[-1] / tile.get_width();   // Width in tiles
            Ht = padded_shape[-2] / tile.get_height();  // Height in tiles
            HtWt = Ht * Wt;

            // Use binary_ng pattern for all dimensions
            C = padded_shape.rank() >= 3 ? padded_shape[-3] : 1;
            N = padded_shape.rank() >= 4 ? padded_shape[-4] : 1;
            D = padded_shape.rank() >= 5 ? padded_shape[-5] : 1;
            cND = D;

            // Calculate strides for tensor memory layout (binary_ng pattern)
            auto calc_strides = [&HtWt](
                                    const auto& shape,
                                    uint32_t target_Wt,
                                    uint32_t target_Ht,
                                    uint32_t target_C,
                                    uint32_t target_N,
                                    uint32_t target_D) {
                // For non-broadcast tensors, we need proper strides for spatial traversal
                // Only check for broadcast if the dimension is 1 AND we're doing column broadcast
                bool tensor_Wt_is_broadcast = (shape[-1] == 1) && (target_Wt > 1);
                bool tensor_Ht_is_broadcast = (shape.rank() >= 2 && shape[-2] == 1) && (target_Ht > 1);
                bool tensor_C_is_broadcast = (shape.rank() >= 3 && shape[-3] == 1) && (target_C > 1);
                bool tensor_N_is_broadcast = (shape.rank() >= 4 && shape[-4] == 1) && (target_N > 1);

                // For stride calculation, use actual tensor tile dimensions
                uint32_t tensor_Wt = tensor_Wt_is_broadcast ? 1 : target_Wt;
                uint32_t tensor_Ht = tensor_Ht_is_broadcast ? 1 : target_Ht;
                uint32_t tensor_HtWt = tensor_Ht * tensor_Wt;

                // For non-broadcast tensors, strides are the normal memory layout offsets
                // For broadcast tensors, strides are 0 (no advancement in that dimension)
                uint32_t c_stride = tensor_C_is_broadcast ? 0 : tensor_HtWt;
                uint32_t n_stride = tensor_N_is_broadcast ? 0 : (tensor_HtWt * target_C);
                uint32_t d_stride = (target_D > 1) ? (tensor_HtWt * target_C * target_N) : 0;
                uint32_t nD_stride = (target_D > 1) ? (tensor_HtWt * target_C * target_N * target_D) : 0;

                return std::tuple{nD_stride, d_stride, n_stride, c_stride};
            };

            // Calculate strides for all tensors using same logic
            auto [nD_stride, d_stride, n_stride, c_stride] = calc_strides(pred_shape, Wt, Ht, C, N, D);
            nD_stride_0 = nD_stride;
            d_stride_0 = d_stride;
            n_stride_0 = n_stride;
            c_stride_0 = c_stride;

            auto [temp_nD_stride_1, temp_d_stride_1, temp_n_stride_1, temp_c_stride_1] =
                calc_strides(true_shape, Wt, Ht, C, N, D);
            nD_stride_1 = temp_nD_stride_1;
            d_stride_1 = temp_d_stride_1;
            n_stride_1 = temp_n_stride_1;
            c_stride_1 = temp_c_stride_1;

            auto [temp_nD_stride_2, temp_d_stride_2, temp_n_stride_2, temp_c_stride_2] =
                calc_strides(false_shape, Wt, Ht, C, N, D);
            nD_stride_2 = temp_nD_stride_2;
            d_stride_2 = temp_d_stride_2;
            n_stride_2 = temp_n_stride_2;
            c_stride_2 = temp_c_stride_2;

            // Extended runtime args for column broadcast (27 arguments)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),                              // 0: src0_addr
                start_tile_id,                                                     // 1: start_tile_id
                static_cast<uint32_t>(predicate_tensor.physical_volume() / 1024),  // 2: src0_num_tiles
                num_tiles_per_core,                                                // 3: dst_num_tiles
                Wt,                                             // 4: dst_shard_width (full width for non-sharded)
                nD_stride_0,                                    // 5: nD_stride
                d_stride_0,                                     // 6: d_stride
                n_stride_0,                                     // 7: n_stride
                c_stride_0,                                     // 8: c_stride
                D,                                              // 9: D
                N,                                              // 10: N
                C,                                              // 11: C
                Ht,                                             // 12: Ht
                Wt,                                             // 13: Wt
                cND,                                            // 14: cND
                value_true_tensor.value().buffer()->address(),  // 15: src1_addr
                nD_stride_1,                                    // 16: nD_stride_1
                d_stride_1,                                     // 17: d_stride_1
                n_stride_1,                                     // 18: n_stride_1
                c_stride_1,                                     // 19: c_stride_1
                static_cast<uint32_t>(value_true_tensor.value().physical_volume() / 1024),   // 20: src1_num_tiles
                value_false_tensor.value().buffer()->address(),                              // 21: src2_addr
                nD_stride_2,                                                                 // 22: nD_stride_2
                d_stride_2,                                                                  // 23: d_stride_2
                n_stride_2,                                                                  // 24: n_stride_2
                c_stride_2,                                                                  // 25: c_stride_2
                static_cast<uint32_t>(value_false_tensor.value().physical_volume() / 1024),  // 26: src2_num_tiles
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else {
            // TTT: predicate (arg 0) + value_true (arg 1) + value_false (arg 2)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor.value().buffer()->address(),
                value_false_tensor.value().buffer()->address(),
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        }

        if (variant == WhereVariant::TTT_COL) {
            // TTT_COL writer needs binary_ng pattern args
            std::array writer_runtime_args = {
                output.buffer()->address(),  // 0: dst_addr
                start_tile_id,               // 1: start_tile_id
                num_tiles_per_core,          // 2: dst_num_tiles
                Wt,                          // 3: dst_shard_width (full width for non-sharded)
                nD_stride_0,                 // 4: nD_stride (using output strides)
                d_stride_0,                  // 5: d_stride
                n_stride_0,                  // 6: n_stride
                c_stride_0,                  // 7: c_stride
                D,                           // 8: D
                N,                           // 9: N
                C,                           // 10: C
                Ht,                          // 11: Ht
                Wt,                          // 12: Wt
                cND,                         // 13: cND
            };
            handle_args(program, writer_kernel_id, core, writer_runtime_args);
        } else {
            // Regular TTT writer args
            std::array writer_runtime_args = {
                output.buffer()->address(),
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, writer_kernel_id, core, writer_runtime_args);
        }

        // All variants use same compute runtime args now
        if (variant == WhereVariant::TTS) {
            auto bit_cast_scalar =
                pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());
            std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else if (variant == WhereVariant::TST) {
            auto bit_cast_scalar =
                pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
            std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            // TTT and TTT_COL use same compute arguments (no scalars)
            std::array compute_runtime_args = {num_tiles_per_core};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::ternary {
WhereDeviceOperation::WhereProgramFactory::cached_program_t WhereDeviceOperation::WhereProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    WhereVariant variant = operation_attributes.where_variant;

    // Use WhereKernelConfig to get the appropriate kernel names
    WhereKernelConfig kernel_config(variant);

    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    // (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());

    // Handle data formats based on variant and tensor availability
    DataFormat value_true_data_format, value_false_data_format;
    if (variant == WhereVariant::TTS) {
        // TTS: only value_true tensor exists
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());

        // the bfloat16 impl of where_llk uses UINT16 instr set.
        // If the bfloat16 inputs' CBs are set to UINT16 dataformat this will enable us to get 'NaN' for bfloat16 dtype
        // We need to test the impact of this on the composite ops that use where op and on the models, since bfloat16
        // packs nan as inf in all other ops.

        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());

        // Use predicate format as fallback for value_false
        value_false_data_format = predicate_data_format;
    } else if (variant == WhereVariant::TST) {
        // TST: only value_false tensor exists
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
        // Use predicate format as fallback for value_true
        value_true_data_format = predicate_data_format;
    } else {
        // TTT and TTT_COL: both tensors exist
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
    }

    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    // datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype());

    uint32_t predicate_single_tile_size = tt_metal::detail::TileSize(predicate_data_format);
    uint32_t value_true_single_tile_size = tt_metal::detail::TileSize(value_true_data_format);
    uint32_t value_false_single_tile_size = tt_metal::detail::TileSize(value_false_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;

    // Input buffers - Create predicate CB (always c_0)
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor

    // Create c_1 based on variant - this is the primary tensor CB
    uint32_t value_true_tensor_cb = 0;
    tt::tt_metal::CBHandle value_true_tensor_cb_handle;
    uint32_t value_false_tensor_cb = 0;
    tt::tt_metal::CBHandle value_false_tensor_cb_handle;

    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor (value_false is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb;
        value_true_tensor_cb_handle = cb_handle;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor (value_true is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb;
        value_false_tensor_cb_handle = cb_handle;
    } else {
        // TTT and TTT_COL: c_1 = value_true tensor, c_2 = value_false tensor
        auto [cb1, cb1_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb1;
        value_true_tensor_cb_handle = cb1_handle;

        auto [cb2, cb2_handle] = create_cb(
            tt::CBIndex::c_2,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb2;
        value_false_tensor_cb_handle = cb2_handle;
    }

    // Output buffer
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    auto predicate_is_dram =
        static_cast<uint32_t>(predicate_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    // Handle DRAM flags based on variant and tensor availability
    uint32_t value_true_is_dram = 0, value_false_is_dram = 0;
    if (variant == WhereVariant::TTS) {
        value_true_is_dram =
            static_cast<uint32_t>(value_true_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    } else if (variant == WhereVariant::TST) {
        value_false_is_dram =
            static_cast<uint32_t>(value_false_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    } else {
        // TTT and TTT_COL: both tensors exist
        value_true_is_dram =
            static_cast<uint32_t>(value_true_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
        value_false_is_dram =
            static_cast<uint32_t>(value_false_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    }

    auto output_is_dram = static_cast<uint32_t>(output.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    // READER KERNEL - Use kernel path from utils
    tt_metal::ReaderDataMovementConfig reader_config;
    if (variant == WhereVariant::TTS) {
        // TTS: c_0 = predicate, c_1 = value_true tensor
        reader_config = tt_metal::ReaderDataMovementConfig(
            {predicate_is_dram, predicate_tensor_cb, value_true_is_dram, value_true_tensor_cb});
    } else if (variant == WhereVariant::TST) {
        // TST: c_0 = predicate, c_1 = value_false tensor
        reader_config = tt_metal::ReaderDataMovementConfig(
            {predicate_is_dram, predicate_tensor_cb, value_false_is_dram, value_false_tensor_cb});
    } else if (variant == WhereVariant::TTT_COL) {
        // TTT_COL: c_0 = predicate, c_1 = value_true, c_2 = value_false with column broadcasting
        // Additional compile-time args for broadcasting
        const auto& pred_shape = predicate_tensor.logical_shape();
        const auto& true_shape = value_true_tensor.value().logical_shape();
        const auto& false_shape = value_false_tensor.value().logical_shape();

        // Determine which tensors need column broadcasting at ELEMENT level
        // Column broadcasting is needed when tensors have different element widths
        uint32_t pred_elem_width = pred_shape[-1];  // Actual element width
        uint32_t true_elem_width = true_shape[-1];
        uint32_t false_elem_width = false_shape[-1];

        // Find the maximum element width (output element width)
        uint32_t max_elem_width = std::max({pred_elem_width, true_elem_width, false_elem_width});

        // A tensor needs broadcasting if its element width is less than the output element width
        bool src0_bcast = pred_elem_width < max_elem_width;   // predicate element broadcast
        bool src1_bcast = true_elem_width < max_elem_width;   // value_true element broadcast
        bool src2_bcast = false_elem_width < max_elem_width;  // value_false element broadcast

        // Set up defines for column broadcasting
        std::map<std::string, std::string> reader_defines;

        // Add broadcast flags as defines
        if (src0_bcast) {
            reader_defines["SRC0_BCAST"] = "1";
        }
        if (src1_bcast) {
            reader_defines["SRC1_BCAST"] = "1";
        }
        if (src2_bcast) {
            reader_defines["SRC2_BCAST"] = "1";
        }

        // Set up column fill macros based on data formats
        if (src0_bcast) {
            if (predicate_data_format == tt::DataFormat::Float16_b) {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column_bfloat16";
            } else {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
            }
        }

        if (src1_bcast) {
            if (value_true_data_format == tt::DataFormat::Float16_b) {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column_bfloat16";
            } else {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
            }
        }

        if (src2_bcast) {
            if (value_false_data_format == tt::DataFormat::Float16_b) {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column_bfloat16";
            } else {
                reader_defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
            }
        }

        reader_config = tt_metal::ReaderDataMovementConfig(
            {predicate_is_dram,
             predicate_tensor_cb,
             static_cast<uint32_t>(false),  // has_sharding - set to false for now
             value_true_is_dram,
             value_true_tensor_cb,
             value_false_is_dram,
             value_false_tensor_cb},
            reader_defines);
    } else {
        // TTT: c_0 = predicate, c_1 = value_true, c_2 = value_false
        reader_config = tt_metal::ReaderDataMovementConfig(
            {predicate_is_dram,
             predicate_tensor_cb,
             value_true_is_dram,
             value_true_tensor_cb,
             value_false_is_dram,
             value_false_tensor_cb});
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.reader_kernel), all_device_cores, reader_config);

    // WRITER KERNEL - Use kernel path from utils
    tt_metal::WriterDataMovementConfig writer_config;
    if (variant == WhereVariant::TTT_COL) {
        // TTT_COL uses binary_ng pattern with 3 compile-time args
        writer_config = tt_metal::WriterDataMovementConfig({
            output_tensor_cb,
            output_is_dram,
            static_cast<uint32_t>(false)  // has_sharding - set to false for now
        });
    } else {
        writer_config = tt_metal::WriterDataMovementConfig({output_tensor_cb, output_is_dram});
    }

    auto writer_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.writer_kernel), all_device_cores, writer_config);

    // COMPUTE KERNEL - Use kernel path from utils
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    // c_0 is always predicate
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;

    // c_1 assignment depends on variant
    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else {
        // TTT and TTT_COL: c_1 = value_true tensor, c_2 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
        unpack_to_dest_mode[tt::CBIndex::c_2] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    }

    // c_3 is always output
    unpack_to_dest_mode[tt::CBIndex::c_3] =
        (output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    // All variants use the same compile args now
    std::vector<uint32_t> compute_kernel_args;
    if (variant == WhereVariant::TTS) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else if (variant == WhereVariant::TST) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else {
        // TTT and TTT_COL use same compute args
        compute_kernel_args = {num_tiles_per_cycle};
    }

    std::map<std::string, std::string> kernel_defines;
    kernel_defines["WHERE_LLK"] = "where_tile";
    kernel_defines["FILL_LLK"] = "fill_tile";
    if (predicate_tensor.dtype() == DataType::FLOAT32) {
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    }
    if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["WHERE_LLK"] = "where_int32_tile";
        kernel_defines["FILL_LLK"] = "fill_tile_int";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else {
        kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
    }

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.compute_kernel),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void WhereDeviceOperation::WhereProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto update_args =
        [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        };

    unity_677ecc7cff58e96986f480c485e0b631::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        update_args);
}

}  // namespace ttnn::operations::ternary
