// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"
#include "compute_kernel_api/global_config_state.h"

namespace ckernel {

// clang-format off
/**
 * Helper function to reconfigure srca and srcb data formats.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * unpacker data format state and only reconfigures when necessary, improving performance and eliminating redundant operations.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srca_new_operand | New srcA operand/CB identifier   | uint32_t | 0 to 31     | True     |
 * | Function   | srcb_new_operand | New srcB operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    // Get the new data formats and tile sizes from the circular buffers
    const uint32_t srca_new_format = CB_interface[srca_new_operand].data_format;
    const uint32_t srca_new_tile_size = CB_interface[srca_new_operand].tile_size;
    const uint32_t srcb_new_format = CB_interface[srcb_new_operand].data_format;
    const uint32_t srcb_new_tile_size = CB_interface[srcb_new_operand].tile_size;

    // Check if srcA reconfiguration is needed
    bool reconfig_srca = unpack_srca_needs_reconfig(srca_new_format, srca_new_tile_size);

    // Check if srcB reconfiguration is needed
    bool reconfig_srcb = unpack_srcb_needs_reconfig(srcb_new_format, srcb_new_tile_size);

    // Only reconfigure if needed
    if (reconfig_srca || reconfig_srcb) {
        UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(srca_new_operand, srcb_new_operand)));
        MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(srca_new_operand, srcb_new_operand)));

        // Update global state
        if (reconfig_srca) {
            update_unpack_srca_config_state(srca_new_format, srca_new_tile_size);
        }
        if (reconfig_srcb) {
            update_unpack_srcb_config_state(srcb_new_format, srcb_new_tile_size);
        }
    }
}

// clang-format off
/**
 * Helper function to reconfigure srca/srcb data formats, only if they differ from existing formats.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now uses the global
 * state tracking instead of requiring the caller to provide old operand information, simplifying the API.
 *
 * NOTE: The old_operand parameters are now ignored as the global state provides this information automatically.
 * This function is kept for backward compatibility but will use the global state for comparison.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srca_old_operand | Old srcA operand (IGNORED)       | uint32_t | 0 to 31     | True     |
 * | Function   | srca_new_operand | New srcA operand/CB identifier   | uint32_t | 0 to 31     | True     |
 * | Function   | srcb_old_operand | Old srcB operand (IGNORED)       | uint32_t | 0 to 31     | True     |
 * | Function   | srcb_new_operand | New srcB operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format(
    [[maybe_unused]] const uint32_t srca_old_operand,  // Ignored - using global state
    const uint32_t srca_new_operand,
    [[maybe_unused]] const uint32_t srcb_old_operand,  // Ignored - using global state
    const uint32_t srcb_new_operand) {
    // Use the simplified version that leverages global state management
    reconfig_data_format<to_from_int8>(srca_new_operand, srcb_new_operand);
}

// clang-format off
/**
 * Helper function to reconfigure srca data format.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * unpacker srcA data format state and only reconfigures when necessary.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srca_new_operand | New srcA operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    // Get the new data format and tile size from the circular buffer
    const uint32_t srca_new_format = CB_interface[srca_new_operand].data_format;
    const uint32_t srca_new_tile_size = CB_interface[srca_new_operand].tile_size;

    // Check if srcA reconfiguration is needed
    if (unpack_srca_needs_reconfig(srca_new_format, srca_new_tile_size)) {
        UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_new_operand)));
        MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_new_operand)));

        // Update global state
        update_unpack_srca_config_state(srca_new_format, srca_new_tile_size);
    }
}

// clang-format off
/**
 * Helper function to reconfigure srca input data format, only if it differs from existing format.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now uses the global
 * state tracking instead of requiring the caller to provide old operand information.
 *
 * NOTE: The old_operand parameter is now ignored as the global state provides this information automatically.
 * This function is kept for backward compatibility but will use the global state for comparison.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srca_old_operand | Old srcA operand (IGNORED)       | uint32_t | 0 to 31     | True     |
 * | Function   | srca_new_operand | New srcA operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srca(
    [[maybe_unused]] const uint32_t srca_old_operand,  // Ignored - using global state
    const uint32_t srca_new_operand) {
    // Use the simplified version that leverages global state management
    reconfig_data_format_srca<to_from_int8>(srca_new_operand);
}

// clang-format off
/**
 * Helper function to reconfigure srcb input data format.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * unpacker srcB data format state and only reconfigures when necessary.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srcb_new_operand | New srcB operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    // Get the new data format and tile size from the circular buffer
    const uint32_t srcb_new_format = CB_interface[srcb_new_operand].data_format;
    const uint32_t srcb_new_tile_size = CB_interface[srcb_new_operand].tile_size;

    // Check if srcB reconfiguration is needed
    if (unpack_srcb_needs_reconfig(srcb_new_format, srcb_new_tile_size)) {
        UNPACK((llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_new_operand)));
        MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_new_operand)));

        // Update global state
        update_unpack_srcb_config_state(srcb_new_format, srcb_new_tile_size);
    }
}

// clang-format off
/**
 * Helper function to reconfigure srcb input data format, only if it differs from existing format.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now uses the global
 * state tracking instead of requiring the caller to provide old operand information.
 *
 * NOTE: The old_operand parameter is now ignored as the global state provides this information automatically.
 * This function is kept for backward compatibility but will use the global state for comparison.
 *
 * | Param Type | Name             | Description                      | Type     | Valid Range | Required |
 * |------------|------------------|----------------------------------|----------|-------------|----------|
 * | Template   | to_from_int8     | Enable int8 conversion support   | bool     | true/false  | False    |
 * | Function   | srcb_old_operand | Old srcB operand (IGNORED)       | uint32_t | 0 to 31     | True     |
 * | Function   | srcb_new_operand | New srcB operand/CB identifier   | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srcb(
    [[maybe_unused]] const uint32_t srcb_old_operand,  // Ignored - using global state
    const uint32_t srcb_new_operand) {
    // Use the simplified version that leverages global state management
    reconfig_data_format_srcb<to_from_int8>(srcb_new_operand);
}

}  // namespace ckernel
