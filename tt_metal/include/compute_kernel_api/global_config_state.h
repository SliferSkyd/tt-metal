// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include <cstdint>

namespace ckernel {

/**
 * @brief Global configuration state management for compute kernels
 *
 * This module provides a centralized way to track the current hardware configuration state
 * for packer and unpacker units. This eliminates the need for explicit "uninit" functions
 * by allowing init functions to check the current state and only reconfigure when necessary.
 *
 * Addresses Issue #22904: Create global configuration state
 */

//========================================
// Packer State Enums and Types
//========================================

/**
 * @brief Packer stride configuration states
 *
 * Tracks the current stride configuration of the packer unit.
 * Different operations (contiguous, tilize, untilize) require different stride patterns.
 */
enum class PackStridesConfig : uint8_t {
    PACK_STRIDES_CONTIGUOUS = 0,  ///< Default contiguous packing strides
    PACK_STRIDES_TILIZE = 1,      ///< Tilize operation strides
    PACK_STRIDES_UNTILIZE = 2,    ///< Untilize operation strides
    PACK_STRIDES_UNKNOWN = 255    ///< Unknown/uninitialized state
};

/**
 * @brief Packer edge masking configuration states
 *
 * Tracks the current edge masking configuration for reduce operations.
 * Different reduce dimensions require different edge mask patterns.
 */
enum class PackEdgeMaskConfig : uint8_t {
    PACK_EDGE_MASK_NONE = 0,           ///< No edge masking (default state)
    PACK_EDGE_MASK_REDUCE_ROW = 1,     ///< Row reduction edge masking
    PACK_EDGE_MASK_REDUCE_COL = 2,     ///< Column reduction edge masking
    PACK_EDGE_MASK_REDUCE_SCALAR = 3,  ///< Scalar reduction edge masking
    PACK_EDGE_MASK_UNKNOWN = 255       ///< Unknown/uninitialized state
};

//========================================
// Unpacker State Enums and Types
//========================================

/**
 * @brief Data format configuration for unpacker sources
 *
 * Tracks the current data format configuration for srcA and srcB.
 * This allows init functions to avoid redundant reconfigurations.
 */
struct UnpackDataFormatConfig {
    uint32_t src_a_format;     ///< Current srcA data format
    uint32_t src_a_tile_size;  ///< Current srcA tile size
    uint32_t src_b_format;     ///< Current srcB data format
    uint32_t src_b_tile_size;  ///< Current srcB tile size
    bool is_valid;             ///< Whether the configuration is valid/initialized

    /// Constructor for uninitialized state
    constexpr UnpackDataFormatConfig() :
        src_a_format(0), src_a_tile_size(0), src_b_format(0), src_b_tile_size(0), is_valid(false) {}

    /// Constructor for initialized state
    constexpr UnpackDataFormatConfig(uint32_t a_fmt, uint32_t a_size, uint32_t b_fmt, uint32_t b_size) :
        src_a_format(a_fmt), src_a_tile_size(a_size), src_b_format(b_fmt), src_b_tile_size(b_size), is_valid(true) {}
};

//========================================
// Global State Variables
//========================================

/**
 * @brief Global variables to track current hardware configuration state
 *
 * These variables are updated by init functions and checked to avoid redundant configurations.
 * They are initialized to "unknown" states to ensure proper initialization on first use.
 */
extern PackStridesConfig g_pack_strides_config_state;
extern PackEdgeMaskConfig g_pack_edge_mask_config_state;
extern UnpackDataFormatConfig g_unpack_data_format_config_state;

//========================================
// State Management Functions
//========================================

/**
 * @brief Initialize global configuration state
 *
 * Should be called once at the beginning of kernel execution to set all states to known defaults.
 * This function resets all global state variables to their default values.
 */
ALWI void global_config_state_init();

/**
 * @brief Reset global configuration state
 *
 * Resets all state variables to unknown/uninitialized state.
 * Useful for debugging or when forcing a complete reconfiguration.
 */
ALWI void global_config_state_reset();

//========================================
// Packer State Management
//========================================

/**
 * @brief Get current packer strides configuration state
 * @return Current packer strides configuration
 */
ALWI PackStridesConfig get_pack_strides_config_state();

/**
 * @brief Set packer strides configuration state
 * @param new_state New packer strides configuration state
 */
ALWI void set_pack_strides_config_state(PackStridesConfig new_state);

/**
 * @brief Check if packer strides reconfiguration is needed
 * @param required_state Required packer strides configuration
 * @return true if reconfiguration is needed, false otherwise
 */
ALWI bool pack_strides_needs_reconfig(PackStridesConfig required_state);

/**
 * @brief Get current packer edge mask configuration state
 * @return Current packer edge mask configuration
 */
ALWI PackEdgeMaskConfig get_pack_edge_mask_config_state();

/**
 * @brief Set packer edge mask configuration state
 * @param new_state New packer edge mask configuration state
 */
ALWI void set_pack_edge_mask_config_state(PackEdgeMaskConfig new_state);

/**
 * @brief Check if packer edge mask reconfiguration is needed
 * @param required_state Required packer edge mask configuration
 * @return true if reconfiguration is needed, false otherwise
 */
ALWI bool pack_edge_mask_needs_reconfig(PackEdgeMaskConfig required_state);

//========================================
// Unpacker State Management
//========================================

/**
 * @brief Get current unpacker data format configuration state
 * @return Current unpacker data format configuration
 */
ALWI const UnpackDataFormatConfig& get_unpack_data_format_config_state();

/**
 * @brief Set unpacker data format configuration state
 * @param new_config New unpacker data format configuration
 */
ALWI void set_unpack_data_format_config_state(const UnpackDataFormatConfig& new_config);

/**
 * @brief Check if unpacker srcA reconfiguration is needed
 * @param src_format Required srcA data format
 * @param tile_size Required srcA tile size
 * @return true if reconfiguration is needed, false otherwise
 */
ALWI bool unpack_srca_needs_reconfig(uint32_t src_format, uint32_t tile_size);

/**
 * @brief Check if unpacker srcB reconfiguration is needed
 * @param src_format Required srcB data format
 * @param tile_size Required srcB tile size
 * @return true if reconfiguration is needed, false otherwise
 */
ALWI bool unpack_srcb_needs_reconfig(uint32_t src_format, uint32_t tile_size);

/**
 * @brief Update unpacker srcA configuration state
 * @param src_format New srcA data format
 * @param tile_size New srcA tile size
 */
ALWI void update_unpack_srca_config_state(uint32_t src_format, uint32_t tile_size);

/**
 * @brief Update unpacker srcB configuration state
 * @param src_format New srcB data format
 * @param tile_size New srcB tile size
 */
ALWI void update_unpack_srcb_config_state(uint32_t src_format, uint32_t tile_size);

//========================================
// Utility Functions
//========================================

/**
 * @brief Convert ReduceDim to PackEdgeMaskConfig
 * @param reduce_dim The reduce dimension
 * @return Corresponding PackEdgeMaskConfig
 */
template <ReduceDim reduce_dim>
constexpr PackEdgeMaskConfig reduce_dim_to_edge_mask_config() {
    if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        return PackEdgeMaskConfig::PACK_EDGE_MASK_REDUCE_ROW;
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        return PackEdgeMaskConfig::PACK_EDGE_MASK_REDUCE_COL;
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        return PackEdgeMaskConfig::PACK_EDGE_MASK_REDUCE_SCALAR;
    } else {
        return PackEdgeMaskConfig::PACK_EDGE_MASK_UNKNOWN;
    }
}

//========================================
// Implementation Details
//========================================

// Inline function implementations
inline PackStridesConfig get_pack_strides_config_state() { return g_pack_strides_config_state; }

inline void set_pack_strides_config_state(PackStridesConfig new_state) { g_pack_strides_config_state = new_state; }

inline bool pack_strides_needs_reconfig(PackStridesConfig required_state) {
    return g_pack_strides_config_state != required_state;
}

inline PackEdgeMaskConfig get_pack_edge_mask_config_state() { return g_pack_edge_mask_config_state; }

inline void set_pack_edge_mask_config_state(PackEdgeMaskConfig new_state) { g_pack_edge_mask_config_state = new_state; }

inline bool pack_edge_mask_needs_reconfig(PackEdgeMaskConfig required_state) {
    return g_pack_edge_mask_config_state != required_state;
}

inline const UnpackDataFormatConfig& get_unpack_data_format_config_state() { return g_unpack_data_format_config_state; }

inline void set_unpack_data_format_config_state(const UnpackDataFormatConfig& new_config) {
    g_unpack_data_format_config_state = new_config;
}

inline bool unpack_srca_needs_reconfig(uint32_t src_format, uint32_t tile_size) {
    return !g_unpack_data_format_config_state.is_valid ||
           g_unpack_data_format_config_state.src_a_format != src_format ||
           g_unpack_data_format_config_state.src_a_tile_size != tile_size;
}

inline bool unpack_srcb_needs_reconfig(uint32_t src_format, uint32_t tile_size) {
    return !g_unpack_data_format_config_state.is_valid ||
           g_unpack_data_format_config_state.src_b_format != src_format ||
           g_unpack_data_format_config_state.src_b_tile_size != tile_size;
}

inline void update_unpack_srca_config_state(uint32_t src_format, uint32_t tile_size) {
    g_unpack_data_format_config_state.src_a_format = src_format;
    g_unpack_data_format_config_state.src_a_tile_size = tile_size;
    g_unpack_data_format_config_state.is_valid = true;
}

inline void update_unpack_srcb_config_state(uint32_t src_format, uint32_t tile_size) {
    g_unpack_data_format_config_state.src_b_format = src_format;
    g_unpack_data_format_config_state.src_b_tile_size = tile_size;
    g_unpack_data_format_config_state.is_valid = true;
}

inline void global_config_state_init() {
    g_pack_strides_config_state = PackStridesConfig::PACK_STRIDES_CONTIGUOUS;
    g_pack_edge_mask_config_state = PackEdgeMaskConfig::PACK_EDGE_MASK_NONE;
    g_unpack_data_format_config_state = UnpackDataFormatConfig();
}

inline void global_config_state_reset() {
    g_pack_strides_config_state = PackStridesConfig::PACK_STRIDES_UNKNOWN;
    g_pack_edge_mask_config_state = PackEdgeMaskConfig::PACK_EDGE_MASK_UNKNOWN;
    g_unpack_data_format_config_state = UnpackDataFormatConfig();
}

}  // namespace ckernel
