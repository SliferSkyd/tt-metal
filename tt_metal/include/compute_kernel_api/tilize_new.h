// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/global_config_state.h"

#ifdef TRISC_MATH
#include "llk_math_pack_sync_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_tilizeA_B_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the tilize operation. Should be called once at the beginning of a kernel.
 *
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * packer stride state and only reconfigures when necessary, eliminating the need for explicit `tilize_uninit` calls.
 *
 * Return value: None
 *
 * | Param Type | Name  | Description                               | Type     | Valid Range | Required |
 * |------------|-------|-------------------------------------------|----------|-------------|----------|
 * | Function   | icb   | Input circular buffer identifier          | uint32_t | 0 to 31     | True     |
 * | Function   | block | Size of tile block to work on             | uint32_t | > 0         | True     |
 * | Function   | ocb   | Output circular buffer identifier         | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) {
    // Initialize unpacker for tilize operation
    UNPACK((llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb)));
    UNPACK((llk_unpack_tilizeA_B_init<false, true>(icb, block)));

    // Initialize math unit
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));

    // Check if packer stride reconfiguration is needed for tilize
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_TILIZE)) {
        // Configure packer for tilize operation with appropriate strides
        PACK((llk_pack_tilize_init<false /*untilize*/, false /*skip_inputs*/, true /*tilize en*/>(ocb)));

        // Update global state to reflect the new configuration
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_TILIZE);
    }

    // Ensure edge masking is in default state for tilize operations
    if (pack_edge_mask_needs_reconfig(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE)) {
        // Clear any edge masks that might have been set by previous reduce operations
        PACK((llk_pack_reduce_mask_clear()));
        set_pack_edge_mask_config_state(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE);
    }
}

// clang-format off
/**
 * Performs the tilize operation on a block of tiles from the input CB and writes the result to the output CB.
 * This function expects that the packer and unpacker have already been configured via tilize_init.
 *
 * Return value: None
 *
 * | Param Type | Name              | Description                                      | Type     | Valid Range                             | Required |
 * |------------|-------------------|--------------------------------------------------|----------|-----------------------------------------|----------|
 * | Function   | icb               | Input circular buffer identifier                 | uint32_t | 0 to 31                                 | True     |
 * | Function   | block             | Size of tile block to work on                   | uint32_t | > 0                                     | True     |
 * | Function   | ocb               | Output circular buffer identifier                | uint32_t | 0 to 31                                 | True     |
 * | Function   | input_tile_index  | Index of the input tile to start from           | uint32_t | Must be less than the size of the CB    | False    |
 * | Function   | output_tile_index | Index of the output tile to start from          | uint32_t | Must be less than the size of the CB    | False    |
 */
// clang-format on
ALWI void tilize_block(
    uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
    UNPACK((llk_unpack_tilizeA_B_block<false, false, false>(icb, block, input_tile_index)));
    PACK((llk_pack_tilize_block<false>(block, ocb, output_tile_index)));
}

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)

// clang-format off
/**
 * Initializes the tilize operation with reduction. Should be called once at the beginning of a kernel.
 *
 * This version incorporates global configuration state management from Issue #22904.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                              | Type     | Valid Range | Required |
 * |------------|----------------|------------------------------------------|----------|-------------|----------|
 * | Template   | neginf_srcA    | NegInf source A flag                     | bool     | true/false  | False    |
 * | Template   | zero_srcA_reduce| Zero source A for reduce flag           | bool     | true/false  | False    |
 * | Function   | icb0           | Input circular buffer A identifier       | uint32_t | 0 to 31     | True     |
 * | Function   | icb1_scaler    | Input circular buffer for scaler         | uint32_t | 0 to 31     | True     |
 * | Function   | block          | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb            | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 * | Function   | num_faces      | Number of faces per tile                 | uint32_t | 1 to 4      | False    |
 * | Function   | face_r_dim     | Number of rows in each face              | uint32_t | 1 to 16     | False    |
 */
// clang-format on
template <bool neginf_srcA = true, bool zero_srcA_reduce = false>
ALWI void tilizeA_B_reduce_init(
    uint32_t icb0,
    uint32_t icb1_scaler,
    uint32_t block,
    uint32_t ocb,
    uint32_t num_faces = 4,
    uint32_t face_r_dim = 16) {
    // Initialize unpacker for tilize with reduce operation
    UNPACK((llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1_scaler)));
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true, false, zero_srcA_reduce>(
        icb0, icb1_scaler, block, num_faces, face_r_dim, 1)));

    // Initialize math unit for reduce operation
    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>()));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));

    // Check if packer stride reconfiguration is needed for tilize
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_TILIZE)) {
        PACK((llk_pack_tilize_init<false /*untilize*/, false /*skip_inputs*/, true /*tilize en*/>(ocb)));
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_TILIZE);
    }

    // Check if packer edge mask reconfiguration is needed for the reduce operation
    constexpr PackEdgeMaskConfig required_mask_config = reduce_dim_to_edge_mask_config<REDUCE_DIM>();
    if (pack_edge_mask_needs_reconfig(required_mask_config)) {
        PACK((llk_pack_reduce_mask_config<false /*untilize*/, REDUCE_DIM>()));
        set_pack_edge_mask_config_state(required_mask_config);
    }
}

#endif  // (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)

// clang-format off
/**
 * Unpacks a block of tiles for tilize operation with optional parameters.
 *
 * Return value: None
 *
 * | Param Type | Name            | Description                              | Type     | Valid Range | Required |
 * |------------|-----------------|------------------------------------------|----------|-------------|----------|
 * | Template   | neginf_srcA     | NegInf source A flag                     | bool     | true/false  | False    |
 * | Template   | reload_srcB     | Reload source B flag                     | bool     | true/false  | False    |
 * | Template   | zero_srcA       | Zero source A flag                       | bool     | true/false  | False    |
 * | Template   | zero_srcA_reduce| Zero source A for reduce flag           | bool     | true/false  | False    |
 * | Function   | icb0            | Input circular buffer A identifier       | uint32_t | 0 to 31     | True     |
 * | Function   | icb1            | Input circular buffer B identifier       | uint32_t | 0 to 31     | True     |
 * | Function   | block           | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | tile_idx_b      | Index of tile B                          | uint32_t | >= 0        | True     |
 * | Function   | num_faces       | Number of faces per tile                 | uint32_t | 1 to 4      | False    |
 * | Function   | srca_face_r_dim | Number of rows in each face for src A    | uint32_t | 1 to 16     | False    |
 */
// clang-format on
template <bool neginf_srcA = true, bool reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
ALWI void unpack_tilizeA_B_block(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t block,
    uint32_t tile_idx_b,
    uint32_t num_faces = 4,
    uint32_t srca_face_r_dim = 16) {
    UNPACK((llk_unpack_tilizeA_B_block<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        icb0, icb1, block, tile_idx_b, num_faces, srca_face_r_dim)));
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
 * to remove calls to tilize_uninit().
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                              | Type     | Valid Range | Required |
 * |----------- |--------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | ocb    | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated(
    "tilize_uninit is no longer needed with global config state management. This function will be removed in a future "
    "release.")]]
ALWI void tilize_uninit(uint32_t icb, uint32_t ocb) {
    // No-op: Global state management handles transitions automatically
    // The next operation's init function will reconfigure as needed
}

// clang-format off
/**
 * Legacy function for backward compatibility.
 *
 * DEPRECATED: This function is now a no-op as it is no longer needed with the new global configuration
 * state management (Issue #22904). The function is kept for backward compatibility but does nothing.
 *
 * NOTE: This function will be completely removed in a future release. Please update your code
 * to remove calls to tilize_uninit_with_dt().
 *
 * Return value: None
 *
 * | Param Type | Name     | Description                              | Type     | Valid Range | Required |
 * |----------- |----------|------------------------------------------|----------|-------------|----------|
 * | Function   | old_icb  | Previous input circular buffer identifier| uint32_t | 0 to 31     | True     |
 * | Function   | new_icb  | New input circular buffer identifier     | uint32_t | 0 to 31     | True     |
 * | Function   | ocb      | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated(
    "tilize_uninit_with_dt is no longer needed with global config state management. This function will be removed in a "
    "future release.")]]
ALWI void tilize_uninit_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t ocb) {
    // No-op: Global state management handles transitions automatically
    // The next operation's init function will reconfigure as needed
}

}  // namespace ckernel
