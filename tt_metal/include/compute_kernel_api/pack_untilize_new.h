// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/global_config_state.h"

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the pack untilize operation for packing tiles from DEST register. Should be called once at the beginning of the kernel.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * packer stride state and only reconfigures when necessary, eliminating the need for explicit `pack_untilize_uninit` calls.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                   | Type      | Valid Range     | Required |
 * |------------|----------------|---------------------------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                              | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim    | Width of a full input in tiles                                | uint32_t  | Divisible by block_ct_dim | False    |
 * | Template   | diagonal       | Whether to use diagonal packing                               | bool      | true/false      | False    |
 * | Template   | narrow_row     | Whether the provided input is narrow                          | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row                                      | uint32_t  | >= 1            | False    |
 * | Function   | icb            | Input circular buffer identifier                               | uint32_t  | 0 to 31         | True     |
 * | Function   | ocb            | Output circular buffer identifier                             | uint32_t  | 0 to 31         | True     |
 * | Function   | face_r_dim     | Face height in rows                                           | uint32_t  | 1, 8 or 16      | False (default=16) |
 * | Function   | num_faces      | Number of faces                                               | uint32_t  | 1, 2 or 4       | False (default=4) |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
    // Check if packer stride reconfiguration is needed for untilize
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_UNTILIZE)) {
        // Initialize packer for untilize operation with appropriate strides
        PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
            ocb, face_r_dim, num_faces)));

        // Update global state to reflect the new configuration
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_UNTILIZE);
    }

    // Ensure edge masking is in default state for untilize operations
    if (pack_edge_mask_needs_reconfig(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE)) {
        // Clear any edge masks that might have been set by previous reduce operations
        PACK((llk_pack_reduce_mask_clear()));
        set_pack_edge_mask_config_state(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE);
    }

    // Initialize packer destination offset registers
    PACK((llk_init_packer_dest_offset_registers<false>()));
}

// clang-format off
/**
 * Initializes the pack untilize operation for packing tiles from DEST register. This version allows configuring
 * the starting destination offset for packing operations.
 *
 * This version incorporates global configuration state management from Issue #22904.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                   | Type      | Valid Range     | Required |
 * |------------|----------------|---------------------------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                              | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim    | Width of a full input in tiles                                | uint32_t  | Divisible by block_ct_dim | False    |
 * | Template   | diagonal       | Whether to use diagonal packing                               | bool      | true/false      | False    |
 * | Template   | narrow_row     | Whether the provided input is narrow                          | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row                                      | uint32_t  | >= 1            | False    |
 * | Function   | ocb            | Output circular buffer identifier                             | uint32_t  | 0 to 31         | True     |
 * | Function   | face_r_dim     | Face height in rows                                           | uint32_t  | 1, 8 or 16      | False (default=16) |
 * | Function   | num_faces      | Number of faces                                               | uint32_t  | 1, 2 or 4       | False (default=4) |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dest_init(uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
    // Check if packer stride reconfiguration is needed for untilize
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_UNTILIZE)) {
        // Initialize packer for untilize operation with appropriate strides
        PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
            ocb, face_r_dim, num_faces)));

        // Update global state to reflect the new configuration
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_UNTILIZE);
    }

    // Ensure edge masking is in default state for untilize operations
    if (pack_edge_mask_needs_reconfig(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE)) {
        // Clear any edge masks that might have been set by previous reduce operations
        PACK((llk_pack_reduce_mask_clear()));
        set_pack_edge_mask_config_state(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE);
    }
}

// clang-format off
/**
 * Performs the pack untilize operation on a block of tiles from the input CB and writes the result to the output CB.
 * In order to properly initialize the operation, a call to `pack_untilize_init` must be made before this function.
 * The width of the block has to be the same as the one provided during the initialization of the pack untilize
 * operation (`pack_untilize_init`).
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                   | Type      | Valid Range     | Required |
 * |------------|----------------|---------------------------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                              | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim    | Width of a full input in tiles                                | uint32_t  | Divisible by block_ct_dim | False    |
 * | Template   | diagonal       | Whether to use diagonal packing                               | bool      | true/false      | False    |
 * | Template   | narrow_row     | Whether the provided input is narrow                          | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row                                      | uint32_t  | >= 1            | False    |
 * | Function   | icb            | Input circular buffer identifier                               | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim   | Height of a single block in tiles                             | uint32_t  | >= 1            | False (default=1) |
 * | Function   | ocb            | Output circular buffer identifier                             | uint32_t  | 0 to 31         | True     |
 * | Function   | block_c_index  | Block column index (used when full_ct_dim > block_ct_dim)     | uint32_t  | >= 0            | False (default=0) |
 * | Function   | face_r_dim     | Face height in rows                                           | uint32_t  | 1, 8 or 16      | False (default=16) |
 * | Function   | num_faces      | Number of faces                                               | uint32_t  | 1, 2 or 4       | False (default=4) |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_block(
    uint32_t icb,
    uint32_t block_rt_dim,
    uint32_t ocb,
    uint32_t block_c_index = 0 /* used when full_ct_dim > block_ct_dim*/,
    uint32_t face_r_dim = 16,
    uint32_t num_faces = 4) {
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
        block_rt_dim, icb, ocb, face_r_dim, num_faces, block_c_index)));
}

// clang-format off
/**
 * Performs the pack untilize operation when PACK input is already in DEST register. In order to properly initialize the operation,
 * a call to `pack_untilize_dest_init` must be made before this function. The width of the block has to be the same
 * as the one provided during the initialization of the pack untilize operation (`pack_untilize_dest_init`). In order for this
 * untilization to be performed correctly, some other function must place the tiles in the DEST register, e.g. `reduce_tile`,
 * `copy_tile`, etc. Similarly as `pack_untilize_block`, this function operates on a whole block that needs to be untilized.
 * Note that the maximum size of the block is limited by the size of the DEST
 * and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                   | Type      | Valid Range     | Required |
 * |------------|----------------|---------------------------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                              | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim    | Width of a full input in tiles                                | uint32_t  | Divisible by block_ct_dim | False    |
 * | Template   | diagonal       | Whether to use diagonal packing                               | bool      | true/false      | False    |
 * | Template   | narrow_row     | Whether the provided input is narrow                          | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row                                      | uint32_t  | >= 1            | False    |
 * | Function   | ocb            | Output circular buffer identifier                             | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim   | Height of a single block in tiles                             | uint32_t  | >= 1            | False (default=1) |
 * | Function   | block_c_index  | Block column index (used when full_ct_dim > block_ct_dim)     | uint32_t  | >= 0            | False (default=0) |
 * | Function   | face_r_dim     | Face height in rows                                           | uint32_t  | 1, 8 or 16      | False (default=16) |
 * | Function   | num_faces      | Number of faces                                               | uint32_t  | 1, 2 or 4       | False (default=4) |
 * | Function   | tile_dst_offset | Index of the tile in the dest from which to pack             | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0) |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dest(
    uint32_t ocb,
    uint32_t block_rt_dim = 1,
    uint32_t block_c_index = 0 /* used when full_ct_dim > block_ct_dim*/,
    uint32_t face_r_dim = 16,
    uint32_t num_faces = 4,
    uint32_t tile_dst_offset = 0) {
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
        block_rt_dim, ocb, face_r_dim, num_faces, block_c_index, tile_dst_offset)));
}

// clang-format off
/**
 * Legacy function for backward compatibility.
 *
 * DEPRECATED: This function is now a no-op as it is no longer needed with the new global configuration
 * state management (Issue #22904). The function is kept for backward compatibility but does nothing.
 *
 * In the new design, the next operation's init function will automatically handle any necessary
 * state transitions, eliminating the need for explicit uninit calls.
 *
 * NOTE: This function will be completely removed in a future release. Please update your code
 * to remove calls to pack_untilize_uninit().
 *
 * Return value: None
 *
 * | Param Type | Name | Description                        | Type     | Valid Range | Required |
 * |------------|------|------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | Output circular buffer identifier  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated(
    "pack_untilize_uninit is no longer needed with global config state management. This function will be removed in a "
    "future release.")]]
ALWI void pack_untilize_uninit(uint32_t ocb) {
    // No-op: Global state management handles transitions automatically
    // The next operation's init function will reconfigure as needed
}

}  // namespace ckernel
