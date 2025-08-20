// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file global_config_state_example.cpp
 * @brief Example demonstrating the use of global configuration state management (Issue #22904)
 *
 * This example shows how the new global configuration state management simplifies compute kernel APIs
 * by eliminating the need for explicit uninit functions and reducing redundant hardware reconfigurations.
 */

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/global_config_state.h"
#include "compute_kernel_api/reduce_new.h"
#include "compute_kernel_api/tilize_new.h"
#include "compute_kernel_api/pack_untilize_new.h"
#include "compute_kernel_api/reconfig_data_format_new.h"

using namespace ckernel;

namespace NAMESPACE {

void MAIN {
    // Initialize global configuration state at the beginning of the kernel
    global_config_state_init();

    // Compile-time parameters
    constexpr uint32_t src_cb_id = 0;
    constexpr uint32_t scaler_cb_id = 1;
    constexpr uint32_t intermediate_cb_id = 2;
    constexpr uint32_t output_cb_id = 3;

    // Runtime parameters
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    //==========================================
    // BEFORE: Old API with explicit uninit calls
    //==========================================
    /*
    // Old way - requires manual state management:

    // 1. Reduce operation
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, intermediate_cb_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, i, 0, i);
    }

    // REQUIRED: Manual uninit to reset packer edge mask state
    reduce_uninit();

    // 2. Tilize operation
    tilize_init(intermediate_cb_id, num_blocks, output_cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        tilize_block(intermediate_cb_id, 1, output_cb_id, block, block);
    }

    // REQUIRED: Manual uninit to reset packer stride state
    tilize_uninit(intermediate_cb_id, output_cb_id);

    // 3. Pack untilize operation
    pack_untilize_init<8>(output_cb_id, intermediate_cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        pack_untilize_block<8>(output_cb_id, 1, intermediate_cb_id);
    }

    // REQUIRED: Manual uninit to reset packer stride state
    pack_untilize_uninit(intermediate_cb_id);
    */

    //==========================================
    // AFTER: New API with global state management
    //==========================================

    // New way - automatic state management:

    // 1. Reduce operation
    // The init function automatically checks current packer edge mask state
    // and only reconfigures if different from REDUCE_ROW configuration
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, intermediate_cb_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, i, 0, i);
    }

    // NO UNINIT REQUIRED! Global state tracks current edge mask configuration

    // 2. Tilize operation
    // The init function automatically checks current packer stride state
    // and only reconfigures if different from TILIZE configuration
    tilize_init(intermediate_cb_id, num_blocks, output_cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        tilize_block(intermediate_cb_id, 1, output_cb_id, block, block);
    }

    // NO UNINIT REQUIRED! Global state tracks current stride configuration

    // 3. Pack untilize operation
    // The init function automatically checks current packer stride state
    // and only reconfigures if different from UNTILIZE configuration
    pack_untilize_init<8>(output_cb_id, intermediate_cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        pack_untilize_block<8>(output_cb_id, 1, intermediate_cb_id);
    }

    // NO UNINIT REQUIRED! Global state tracks current stride configuration

    //==========================================
    // Data format reconfiguration example
    //==========================================

    // Suppose we need to switch to different data formats
    constexpr uint32_t fp16_cb_id = 4;
    constexpr uint32_t fp32_cb_id = 5;

    // First reconfiguration - will actually reconfigure hardware
    reconfig_data_format_srca(fp16_cb_id);
    reconfig_data_format_srcb(fp32_cb_id);

    // Second call with same formats - will be skipped automatically!
    reconfig_data_format_srca(fp16_cb_id);  // No-op due to global state tracking
    reconfig_data_format_srcb(fp32_cb_id);  // No-op due to global state tracking

    // Different formats - will reconfigure only what's needed
    reconfig_data_format_srca(fp32_cb_id);  // Only srcA reconfigured
    // srcB remains unchanged since it's already fp32_cb_id format

    //==========================================
    // Multiple operations with automatic transitions
    //==========================================

    // Sequence of operations with automatic state management:

    // Reduce operation (sets edge mask to REDUCE_COL)
    reduce_init<PoolType::AVG, ReduceDim::REDUCE_COL>(src_cb_id, scaler_cb_id, intermediate_cb_id);
    // ... reduce operations ...

    // Tilize operation (automatically clears edge mask, sets strides to TILIZE)
    tilize_init(intermediate_cb_id, num_blocks, output_cb_id);
    // ... tilize operations ...

    // Another reduce operation (automatically sets strides to CONTIGUOUS, sets edge mask to REDUCE_SCALAR)
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(output_cb_id, scaler_cb_id, intermediate_cb_id);
    // ... reduce operations ...

    // Pack untilize operation (automatically clears edge mask, sets strides to UNTILIZE)
    pack_untilize_init<4>(intermediate_cb_id, output_cb_id);
    // ... untilize operations ...

    // All transitions happen automatically based on global state tracking!

    //==========================================
    // Performance benefits
    //==========================================

    // The global state management provides several performance benefits:
    // 1. Eliminates redundant hardware reconfigurations
    // 2. Reduces code complexity and potential for errors
    // 3. Allows for more efficient operation sequences
    // 4. Simplifies kernel development and maintenance

    // Example: If multiple consecutive reduce operations use the same dimension,
    // only the first one will actually configure the hardware
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, output_cb_id);
    // ... first reduce operation ...

    reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(src_cb_id, scaler_cb_id, output_cb_id);
    // Second init is much faster - edge mask already configured for REDUCE_ROW!
    // ... second reduce operation ...
}

}  // namespace NAMESPACE

//==========================================
// Migration Guide for Existing Code
//==========================================

/*
MIGRATION GUIDE: Converting from old API to new global state API

1. REMOVE all uninit function calls:
   - reduce_uninit() -> DELETE
   - tilize_uninit() -> DELETE
   - pack_untilize_uninit() -> DELETE

2. ADD global state initialization:
   - Add global_config_state_init() at the beginning of your kernel

3. UPDATE includes:
   - #include "compute_kernel_api/reduce_new.h"
   - #include "compute_kernel_api/tilize_new.h"
   - #include "compute_kernel_api/pack_untilize_new.h"
   - #include "compute_kernel_api/reconfig_data_format_new.h"

4. SIMPLIFY data format reconfigurations:
   - Remove old_operand parameters from reconfig_data_format* functions
   - The global state will track previous configurations automatically

5. PERFORMANCE: Your code will automatically benefit from:
   - Reduced hardware reconfigurations
   - Simplified control flow
   - Better optimization opportunities

BACKWARD COMPATIBILITY:
- Old functions are marked deprecated but still work
- They will be removed in a future release
- Migration is straightforward and provides immediate benefits
*/
