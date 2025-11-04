// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/cb_utils.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/conv/conv2d/conv2d_utils.hpp>
#include <ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp>
#include <ttnn/operations/sliding_window/sliding_window.hpp>
#include <ttnn/tensor/shape/shape.hpp>

// Pool-related includes for reusing pool kernels
#include <ttnn/operations/pool/pool_utils.hpp>

namespace ttnn::operations::conv {
namespace conv2d {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_conv2d_depthwise(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    const std::optional<const Tensor>& bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const sliding_window::ParallelConfig& parallel_config,
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const Conv2dParallelizationConfig& parallelization_config,
    const Conv2dBlockConfig& block_config,
    bool is_col_major,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool config_tensors_in_dram,
    std::optional<bool> force_split_reader) {
    TT_FATAL(
        groups == ashape[3] && groups == output_channels,
        "Depthwise factory requires groups == input_channels == output_channels");

    log_info(tt::LogOp, "Creating 2D depthwise convolution program using pool-based approach");
    log_info(
        tt::LogOp,
        "Input shape: [{}, {}, {}, {}], groups: {}, output_channels: {}",
        ashape[0],
        ashape[1],
        ashape[2],
        ashape[3],
        groups,
        output_channels);

    // Extract parameters from sliding window config for pool kernel
    const uint32_t kernel_h = sliding_window_config.window_hw.first;
    const uint32_t kernel_w = sliding_window_config.window_hw.second;
    const uint32_t stride_h = sliding_window_config.stride_hw.first;
    const uint32_t stride_w = sliding_window_config.stride_hw.second;
    const uint32_t pad_t = sliding_window_config.padding[0];
    const uint32_t pad_b = sliding_window_config.padding[1];
    const uint32_t pad_l = sliding_window_config.padding[2];
    const uint32_t pad_r = sliding_window_config.padding[3];

    log_info(
        tt::LogOp,
        "Kernel: {}x{}, Stride: {}x{}, Padding: t={} b={} l={} r={}",
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_t,
        pad_b,
        pad_l,
        pad_r);

    // Get output shape from sliding window config
    auto output_shape = sliding_window_config.get_output_shape();
    const uint32_t out_h = output_shape[1];
    const uint32_t out_w = output_shape[2];

    log_info(
        tt::LogOp, "Sliding window output shape: [{}, {}, {}], groups: {}", ashape[0], out_h, out_w, output_channels);
    log_info(
        tt::LogOp,
        "Actual output tensor shape: [{}, {}, {}, {}]",
        output.logical_shape()[0],
        output.logical_shape()[1],
        output.logical_shape()[2],
        output.logical_shape()[3]);

    // Suppress unused variable warnings for parameters that will be used in kernel integration
    (void)stride_h;
    (void)stride_w;
    (void)pad_t;
    (void)pad_b;
    (void)pad_l;
    (void)pad_r;
    (void)out_h;
    (void)out_w;

    // For depthwise convolution, we need to work with the flattened tensor layout
    // The output tensor comes in as [1, 1, batch*out_h*out_w, out_channels] = [1, 1, 64, 4]
    // We need to set up pool parameters to handle this correctly
    const uint32_t num_shards_c = 1;  // Keep simple for now
    ttnn::operations::pool::FactoryParameters params = ttnn::operations::pool::get_factory_parameters(
        num_shards_c,    // num_shards_c - simple sharding
        a.dtype(),       // input dtype
        output.dtype(),  // output dtype
        kernel_h,
        kernel_w,
        output_channels,                                 // Use output channels (4) for pool calculations
        ttnn::operations::pool::Pool2DType::AVG_POOL2D,  // Use AVG_POOL2D as base for depthwise
        false,                                           // return_indices
        output.layout()                                  // output layout
    );

    log_info(
        tt::LogOp,
        "Pool factory params: data_format={}, output_data_format={}, nbytes={}",
        static_cast<int>(params.data_format),
        static_cast<int>(params.output_data_format),
        params.nbytes);
    log_info(
        tt::LogOp,
        "Pool factory params: in_ntiles_c={}, out_ntiles_c={}, num_shards_c={}",
        params.in_ntiles_c,
        params.out_ntiles_c,
        num_shards_c);
    log_info(
        tt::LogOp,
        "Pool factory params: max_rows_for_reduction={}, is_large_kernel={}",
        params.max_rows_for_reduction,
        params.is_large_kernel);
    log_info(
        tt::LogOp, "Input tensor: channels={}, parallel_grid cores={}", ashape[3], parallel_config.grid.num_cores());

    // Calculate effective tiles the same way as pool factory
    uint32_t effective_tiles = (kernel_h * kernel_w * params.in_ntiles_c * 32 + 1023) / 1024;
    log_info(tt::LogOp, "Effective tiles for depthwise conv: {}", effective_tiles);

    // Create circular buffers for depthwise convolution (same as pool factory pattern)
    uint32_t next_cb_index = 0;

    // Input CB - using same logic as pool factory
    const uint32_t in_cb_id = next_cb_index++;
    const uint32_t in_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t in_cb_npages = params.multi_buffering_factor * std::min(effective_tiles, 3u);
    tt::tt_metal::create_cb(in_cb_id, program, parallel_config.grid, in_cb_pagesize, in_cb_npages, params.data_format);
    log_info(tt::LogOp, "CB {} (input) :: PS = {}, NP = {}", in_cb_id, in_cb_pagesize, in_cb_npages);

    // mul_cb - stores results of element-wise multiplication (using effective tiles logic)
    const uint32_t mul_cb_id = next_cb_index++;
    const uint32_t mul_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t mul_cb_npages = std::min(effective_tiles, 3u) * params.multi_buffering_factor;
    tt::tt_metal::create_cb(
        mul_cb_id, program, parallel_config.grid, mul_cb_pagesize, mul_cb_npages, params.data_format);
    log_info(tt::LogOp, "CB {} (mul_cb) :: PS = {}, NP = {}", mul_cb_id, mul_cb_pagesize, mul_cb_npages);

    // weight_cb - stores weight tensors (reduced size)
    const uint32_t weight_cb_id = next_cb_index++;
    const uint32_t weight_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t weight_cb_npages = 2;  // Reduced from 8 to 2
    tt::tt_metal::create_cb(
        weight_cb_id, program, parallel_config.grid, weight_cb_pagesize, weight_cb_npages, params.data_format);
    log_info(tt::LogOp, "CB {} (weight_cb) :: PS = {}, NP = {}", weight_cb_id, weight_cb_pagesize, weight_cb_npages);

    // Output CB
    const uint32_t out_cb_id = next_cb_index++;
    const uint32_t out_cb_pagesize = tt::tile_size(params.output_data_format);
    const uint32_t out_cb_npages = 2;  // Double buffer
    tt::tt_metal::create_cb(
        out_cb_id, program, parallel_config.grid, out_cb_pagesize, out_cb_npages, params.output_data_format);
    log_info(tt::LogOp, "CB {} (output) :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    // Scalar CB - stores scalar values for pool operations
    const uint32_t in_scalar_cb_id_0 = next_cb_index++;
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t in_scalar_cb_npages = 1;  // Single buffer for depthwise
    tt::tt_metal::create_cb(
        in_scalar_cb_id_0,
        program,
        parallel_config.grid,
        in_scalar_cb_pagesize,
        in_scalar_cb_npages,
        params.data_format);
    log_info(
        tt::LogOp,
        "CB {} (in_scalar_cb_0) :: PS = {}, NP = {}",
        in_scalar_cb_id_0,
        in_scalar_cb_pagesize,
        in_scalar_cb_npages);

    // Clear value CB - stores "clear value" (-inf for maxpool, 0 for avgpool)
    const uint32_t clear_value_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(
        clear_value_cb_id, program, parallel_config.grid, tt::tile_size(params.data_format), 1, params.data_format);
    log_info(
        tt::LogOp,
        "CB {} (clear_value_cb) :: PS = {}, NP = {}",
        clear_value_cb_id,
        tt::tile_size(params.data_format),
        1);

    // Reader indices will be created after out_nhw_per_core is calculated

    // Now create the actual pool kernels by referencing their paths directly
    log_info(tt::LogOp, "Setting up pool kernels for depthwise convolution");

    // We need to set up the arguments exactly like the pool factory does
    // Let me get the key parameters that pool factory calculates
    const uint32_t out_nhw_per_core =
        (out_h * out_w + parallel_config.grid.num_cores() - 1) / parallel_config.grid.num_cores();

    // Generate proper top_left_indices using sliding window infrastructure
    // This ensures correct multi-core memory access patterns like pool operations
    const uint32_t num_cores = parallel_config.grid.num_cores();

    log_info(
        tt::LogOp,
        "Generating proper sliding window indices: cores={}, output_per_core={}",
        num_cores,
        out_nhw_per_core);

    // Use the sliding window infrastructure to generate proper reader indices
    // This accounts for stride patterns, shard boundaries, and memory access patterns
    std::vector<std::vector<uint16_t>> top_left_indices = sliding_window::generate_sliding_window_op_config(
        op_trace_metadata,
        shard_boundaries,
        stride_w,
        true,  // is_conv = true (for depthwise convolution)
        0,     // reader0_datums = 0 (use defaults)
        0,     // reader1_datums = 0 (use defaults)
        true   // pad_cores = true
    );

    log_info(tt::LogOp, "top_left_indices ", top_left_indices);

    for (const auto& core_indices : top_left_indices) {
        log_info(tt::LogOp, "Core indices size: {}", core_indices.size());
        for (size_t i = 0; i < std::min(core_indices.size(), size_t(10)); ++i) {
            log_info(tt::LogOp, "  Index[{}] = {}", i, core_indices[i]);
        }
    }

    log_info(
        tt::LogOp,
        "Generated proper sliding window indices: cores={}, indices_per_core={}",
        top_left_indices.size(),
        top_left_indices.empty() ? 0 : top_left_indices[0].size());
    if (!top_left_indices.empty() && !top_left_indices[0].empty()) {
        log_info(
            tt::LogOp,
            "Core 0 indices: size={}, first_few=[{}, {}, {}...]",
            top_left_indices[0].size(),
            !top_left_indices[0].empty() ? top_left_indices[0][0] : 0,
            top_left_indices[0].size() > 1 ? top_left_indices[0][1] : 0,
            top_left_indices[0].size() > 2 ? top_left_indices[0][2] : 0);
    }

    log_info(tt::LogOp, "About to call construct_on_host_config_tensor with {} cores", top_left_indices.size());
    Tensor reader_indices = sliding_window::construct_on_host_config_tensor(top_left_indices, parallel_config);
    log_info(tt::LogOp, "construct_on_host_config_tensor completed successfully");
    auto reader_shape = reader_indices.logical_shape();
    log_info(tt::LogOp, "reader_indices shape: rank={}, shape_size={}", reader_shape.rank(), reader_shape.size());
    for (uint32_t i = 0; i < reader_shape.rank(); ++i) {
        log_info(tt::LogOp, "  shape[{}] = {}", i, reader_shape[i]);
    }

    bool is_block_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
    log_info(tt::LogOp, "About to call move_config_tensor_to_device with is_block_sharded={}", is_block_sharded);
    Tensor reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, a.device());
    log_info(tt::LogOp, "move_config_tensor_to_device completed successfully");

    auto moved_shape = reader_indices_on_device.logical_shape();
    log_info(
        tt::LogOp, "Reader indices tensor on device: rank={}, shape_size={}", moved_shape.rank(), moved_shape.size());
    for (uint32_t i = 0; i < moved_shape.rank(); ++i) {
        log_info(tt::LogOp, "  moved_shape[{}] = {}", i, moved_shape[i]);
    }

    // Create reader indices CB using the same pattern as pool factory
    const tt::tt_metal::DeviceStorage& reader_indices_storage = reader_indices_on_device.device_storage();
    const uint32_t in_reader_indices_cb_id = next_cb_index++;
    const uint32_t reader_indices_size = top_left_indices[0].size();
    const uint32_t in_reader_indices_cb_pagesize = tt::round_up(reader_indices_size * sizeof(uint16_t), 4);
    constexpr uint32_t in_reader_indices_cb_npages = 1;

    tt::tt_metal::create_cb(
        in_reader_indices_cb_id,
        program,
        parallel_config.grid,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        tt::DataFormat::UInt16,
        reader_indices_storage.get_buffer());

    log_info(
        tt::LogOp,
        "CB {} (reader_indices_cb) :: PS = {}, NP = {}, reader_size={}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        reader_indices_size);

    const uint32_t in_w = ashape[2];
    const uint32_t in_c_per_shard_ceil = ashape[3];  // For now, simplified
    const uint32_t in_nblocks_c = 1;                 // Simplified
    const uint32_t in_cb_sz = tt::tile_size(params.data_format);
    const uint32_t in_nbytes_leftover = 0;  // Simplified
    const uint32_t shard_width_bytes = ashape[3] * params.nbytes;
    const uint32_t in_nbytes_c = ashape[3] * params.nbytes;
    const bool is_output_tiled = (output.layout() == Layout::TILE);
    const bool is_output_block_format = tt::tt_metal::is_block_float(output.dtype());

    log_info(tt::LogOp, "Setting up pool kernel arguments: out_nhw_per_core={}, in_w={}", out_nhw_per_core, in_w);

    // Set up reader arguments following exact pool factory pattern (47 args)
    std::vector<uint32_t> reader_ct_args = {
        out_nhw_per_core,               // 0
        kernel_h,                       // 1
        kernel_w,                       // 2
        pad_l,                          // 3 - pad_w
        in_nbytes_leftover,             // 4
        in_w,                           // 5
        in_c_per_shard_ceil,            // 6
        false,                          // 7 - split_reader (disabled for now)
        0,                              // 8 - split reader id
        0,                              // 9 - bf16_scalar (simplified)
        0,                              // 10 - bf16_init_value (simplified)
        in_nblocks_c,                   // 11
        in_cb_sz,                       // 12
        params.max_rows_for_reduction,  // 13 - max_rows_for_reduction
        0,                              // 14 - ceil_pad_w
        in_cb_id,                       // 15 - in_cb_id_0
        32,                             // 16 - in_cb_id_1 (invalid CB for no split reader)
        32,                             // 17 - raw_in_cb_id (invalid CB)
        in_reader_indices_cb_id,        // 18 - in_reader_indices_cb_id
        in_scalar_cb_id_0,              // 19 - in_scalar_cb_id_0
        32,                             // 20 - in_scalar_cb_id_1 (invalid CB)
        32,                             // 21 - in_idx_cb_id (invalid CB)
        32,                             // 22 - pack_tmp_cb_id (invalid CB)
        32,                             // 23 - pack_idx_tmp_cb_id (invalid CB)
        32,                             // 24 - right_inc_cb_id (invalid CB)
        32,                             // 25 - down_left_wrap_inc_cb_id (invalid CB)
        32,                             // 26 - up_left_wrap_inc_cb_id (invalid CB)
        clear_value_cb_id,              // 27 - clear_value_cb_id
        static_cast<uint32_t>(ttnn::operations::pool::Pool2DType::AVG_POOL2D),  // 28 - pool_type
        1,                                                                      // 29 - one_scalar_per_core (simplified)
        32,                                                                     // 30 - config_cb_id (invalid CB)
        in_nbytes_c,                                                            // 31
        shard_width_bytes,                                                      // 32
        1,            // 33 - multi_buffering_factor (simplified)
        stride_w,     // 34
        1,            // 35 - dilation_h (simplified)
        1,            // 36 - dilation_w (simplified)
        false,        // 37 - return_indices
        pad_t,        // 38
        pad_l,        // 39
        0,            // 40 - right_inc
        0,            // 41 - down_left_wrap_inc
        0,            // 42 - up_left_wrap_inc
        false,        // 43 - zero_pages
        out_cb_id,    // 44
        32,           // 45 - out_idx_cb_id (invalid CB)
        weight_cb_id  // 46
    };

    // Set up compute arguments following exact pool factory pattern (33 args)
    std::vector<uint32_t> compute_ct_args = {
        params.in_ntiles_c,             // 0
        kernel_h * kernel_w,            // 1
        false,                          // 2 - split_reader
        out_nhw_per_core,               // 3
        in_c_per_shard_ceil,            // 4
        in_nblocks_c,                   // 5
        params.max_rows_for_reduction,  // 6 - max_rows_for_reduction
        in_cb_id,                       // 7 - in_cb_id_0
        32,                             // 8 - in_cb_id_1 (invalid CB)
        in_scalar_cb_id_0,              // 9 - in_scalar_cb_id_0
        32,                             // 10 - in_scalar_cb_id_1 (invalid CB)
        32,                             // 11 - in_idx_cb_id (invalid CB)
        32,                             // 12 - pack_tmp_cb_id (invalid CB)
        32,                             // 13 - pack_idx_tmp_cb_id (invalid CB)
        32,                             // 14 - right_inc_cb_id (invalid CB)
        32,                             // 15 - down_left_wrap_inc_cb_id (invalid CB)
        32,                             // 16 - up_left_wrap_inc_cb_id (invalid CB)
        out_cb_id,                      // 17
        32,                             // 18 - out_idx_cb_id (invalid CB)
        1,                              // 19 - one_scalar_per_core
        32,                             // 20 - pre_tilize_cb_id (invalid CB for now)
        is_output_tiled,                // 21
        is_output_block_format,         // 22
        false,                          // 23 - return_indices
        stride_h,                       // 24
        stride_w,                       // 25
        ashape[1] + pad_t + pad_b,      // 26 - in_h_padded
        ashape[2] + pad_l + pad_r,      // 27 - in_w_padded
        kernel_h,                       // 28 - eff_kernel_h
        kernel_w,                       // 29 - eff_kernel_w
        pad_l,                          // 30
        weight_cb_id,                   // 31
        mul_cb_id                       // 32
    };

    // Create reader kernel using pool reader kernel path
    std::string reader_kernel_path = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp";
    auto reader_kernel = CreateKernel(
        program,
        reader_kernel_path,
        parallel_config.grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_ct_args});

    // Create compute kernel using pool compute kernel path
    std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp";

    // Get the defines needed for pool kernel compilation (same as pool factory)
    std::map<std::string, std::string> compute_defines = ttnn::operations::pool::get_defines(
        ttnn::operations::pool::Pool2DType::AVG_POOL2D  // Use AVG_POOL2D for depthwise
    );

    // Let the kernel compilation system handle MATH_FIDELITY and DST_ACCUM_MODE automatically
    // based on the ComputeConfig settings. Only set TILE_C_DIM manually.
    compute_defines["TILE_C_DIM"] = std::to_string(tt::constants::TILE_WIDTH);

    auto compute_kernel = CreateKernel(
        program,
        compute_kernel_path,
        parallel_config.grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = get_math_fidelity(compute_kernel_config),
            .fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config),
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
            .defines = compute_defines});

    log_info(tt::LogOp, "Depthwise conv2d factory: Pool kernels created successfully");
    log_info(tt::LogOp, "Using reader: {}", reader_kernel_path);
    log_info(tt::LogOp, "Using compute: {}", compute_kernel_path);
    log_info(
        tt::LogOp,
        "Grid size: {}x{} = {} cores",
        parallel_config.grid.bounding_box().grid_size().x,
        parallel_config.grid.bounding_box().grid_size().y,
        parallel_config.grid.num_cores());
    log_info(
        tt::LogOp,
        "out_nhw_per_core={}, expected_sticks={}",
        out_nhw_per_core,
        out_nhw_per_core * in_c_per_shard_ceil / in_nblocks_c);

    // Suppress unused variable warnings for kernel handles (they are stored in the program)
    (void)reader_kernel;
    (void)compute_kernel;

    // Create runtime arguments callback to properly set up buffer addresses for pool kernels
    auto override_runtime_arguments_callback =
        [reader_kernel,
         compute_kernel,
         in_cb_id,
         out_cb_id,
         weight_cb_id,
         mul_cb_id,
         in_reader_indices_cb_id,
         reader_indices_on_device](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            log_info(tt::LogOp, "Setting up runtime arguments for depthwise conv2d pool kernels");

            const auto& input_tensor = input_tensors.at(0);
            const auto& output_tensor = output_tensors.at(0);

            auto src_buffer = input_tensor.buffer();
            auto dst_buffer = output_tensor.buffer();

            // Update circular buffer addresses for input and output
            if (input_tensor.is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, in_cb_id, *src_buffer);
            }
            if (output_tensor.is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, out_cb_id, *dst_buffer);
            }

            // Update reader indices circular buffer address
            auto reader_indices_buffer = reader_indices_on_device.buffer();
            UpdateDynamicCircularBufferAddress(program, in_reader_indices_cb_id, *reader_indices_buffer);

            log_info(tt::LogOp, "Depthwise conv2d runtime arguments configured successfully");
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace conv2d
}  // namespace ttnn::operations::conv
