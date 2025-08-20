// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_config_state.h"

namespace ckernel {

/**
 * @brief Global state variable definitions
 *
 * These variables track the current hardware configuration state across all compute operations.
 * They are initialized to "unknown" states to ensure proper initialization on first use.
 */

// Initialize packer strides configuration to unknown state
PackStridesConfig g_pack_strides_config_state = PackStridesConfig::PACK_STRIDES_UNKNOWN;

// Initialize packer edge mask configuration to unknown state
PackEdgeMaskConfig g_pack_edge_mask_config_state = PackEdgeMaskConfig::PACK_EDGE_MASK_UNKNOWN;

// Initialize unpacker data format configuration to invalid/uninitialized state
UnpackDataFormatConfig g_unpack_data_format_config_state;

}  // namespace ckernel
