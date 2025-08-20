# Global Configuration State Management Solution

**Issue Reference**: [tt-metal#22904](https://github.com/tenstorrent/tt-metal/issues/22904)

## Overview

This document describes the comprehensive solution for implementing global configuration state management in the tt-metal compute kernel APIs. The solution eliminates the need for explicit "uninit" functions by tracking the current hardware configuration state and only performing reconfigurations when necessary.

## Problem Statement

The current compute kernel APIs require explicit "uninit" functions to revert hardware state:

- `reduce_revert_delta()` / `reduce_uninit()`
- `tilize_uninit()`
- `pack_untilize_uninit()`

These functions are needed because different operations (reduce, tilize, untilize) change the hardware state differently from regular contiguous operations. This creates several issues:

1. **API Complexity**: Developers must remember to call uninit functions
2. **Performance Overhead**: Redundant hardware reconfigurations
3. **Error Prone**: Missing uninit calls cause incorrect behavior
4. **Programming Model Inconsistency**: Uninit pattern doesn't fit the preferred model

## Solution Architecture

### Core Components

#### 1. Global State Tracking (`global_config_state.h`)

The solution introduces three main state tracking categories:

```cpp
// Packer stride configuration states
enum class PackStridesConfig : uint8_t {
    PACK_STRIDES_CONTIGUOUS = 0,  // Default contiguous packing
    PACK_STRIDES_TILIZE     = 1,  // Tilize operation strides
    PACK_STRIDES_UNTILIZE   = 2,  // Untilize operation strides
    PACK_STRIDES_UNKNOWN    = 255 // Uninitialized state
};

// Packer edge masking configuration states
enum class PackEdgeMaskConfig : uint8_t {
    PACK_EDGE_MASK_NONE        = 0,  // No edge masking (default)
    PACK_EDGE_MASK_REDUCE_ROW  = 1,  // Row reduction masking
    PACK_EDGE_MASK_REDUCE_COL  = 2,  // Column reduction masking
    PACK_EDGE_MASK_REDUCE_SCALAR = 3, // Scalar reduction masking
    PACK_EDGE_MASK_UNKNOWN     = 255 // Uninitialized state
};

// Unpacker data format configuration
struct UnpackDataFormatConfig {
    uint32_t src_a_format;      // Current srcA data format
    uint32_t src_a_tile_size;   // Current srcA tile size
    uint32_t src_b_format;      // Current srcB data format
    uint32_t src_b_tile_size;   // Current srcB tile size
    bool     is_valid;          // Configuration validity flag
};
```

#### 2. Global State Variables

```cpp
extern PackStridesConfig g_pack_strides_config_state;
extern PackEdgeMaskConfig g_pack_edge_mask_config_state;
extern UnpackDataFormatConfig g_unpack_data_format_config_state;
```

#### 3. State Management Functions

The API provides functions to:
- Query current state
- Check if reconfiguration is needed
- Update state after configuration changes

### Updated API Functions

#### Reduce Operations (`reduce_new.h`)

```cpp
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb) {
    // Initialize unpacker and math units
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim, BroadcastType::NONE, enforce_fp32_accumulation>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, enforce_fp32_accumulation>()));

    // Check if packer edge mask reconfiguration is needed
    constexpr PackEdgeMaskConfig required_mask_config = reduce_dim_to_edge_mask_config<reduce_dim>();

    if (pack_edge_mask_needs_reconfig(required_mask_config)) {
        // Only reconfigure if current state is different
        PACK((llk_pack_reduce_mask_config<false, reduce_dim>()));
        set_pack_edge_mask_config_state(required_mask_config);
    }

    // Ensure packer strides are in contiguous mode
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_CONTIGUOUS)) {
        PACK((llk_pack_init(ocb)));
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_CONTIGUOUS);
    }
}
```

**Key Changes:**
- Checks current edge mask state before reconfiguration
- Only calls hardware configuration functions when necessary
- Updates global state after changes
- `reduce_uninit()` becomes deprecated no-op

#### Tilize Operations (`tilize_new.h`)

```cpp
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) {
    // Initialize unpacker and math units
    UNPACK((llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb)));
    UNPACK((llk_unpack_tilizeA_B_init<false, true>(icb, block)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));

    // Check if packer stride reconfiguration is needed
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_TILIZE)) {
        PACK((llk_pack_tilize_init<false, false, true>(ocb)));
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_TILIZE);
    }

    // Ensure edge masking is cleared
    if (pack_edge_mask_needs_reconfig(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE)) {
        PACK((llk_pack_reduce_mask_clear()));
        set_pack_edge_mask_config_state(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE);
    }
}
```

**Key Changes:**
- Checks current stride configuration before reconfiguration
- Clears edge masks if previously set by reduce operations
- `tilize_uninit()` becomes deprecated no-op

#### Pack Untilize Operations (`pack_untilize_new.h`)

```cpp
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim, bool diagonal = false, bool narrow_row = false, std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
    // Check if packer stride reconfiguration is needed
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_UNTILIZE)) {
        PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(ocb, face_r_dim, num_faces)));
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_UNTILIZE);
    }

    // Ensure edge masking is cleared
    if (pack_edge_mask_needs_reconfig(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE)) {
        PACK((llk_pack_reduce_mask_clear()));
        set_pack_edge_mask_config_state(PackEdgeMaskConfig::PACK_EDGE_MASK_NONE);
    }

    PACK((llk_init_packer_dest_offset_registers<false>()));
}
```

**Key Changes:**
- Checks current stride configuration before reconfiguration
- `pack_untilize_uninit()` becomes deprecated no-op

#### Data Format Reconfiguration (`reconfig_data_format_new.h`)

```cpp
template <bool to_from_int8 = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
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
```

**Key Changes:**
- Tracks current data formats and tile sizes
- Only reconfigures when format or tile size actually changes
- Eliminates need for "old_operand" parameters in API

## Implementation Benefits

### 1. Performance Improvements

- **Reduced Hardware Reconfigurations**: Only reconfigure when state actually changes
- **Better Optimization**: Consecutive operations with same configuration skip hardware setup
- **Lower Latency**: Fewer register writes and hardware stalls

### 2. API Simplification

- **No Uninit Functions**: Eliminates `reduce_uninit()`, `tilize_uninit()`, `pack_untilize_uninit()`
- **Simplified Data Format API**: No need to track previous operand information
- **Automatic State Management**: Developers don't need to manage hardware state manually

### 3. Error Reduction

- **No Missing Uninits**: Impossible to forget uninit calls
- **Consistent State**: Global state ensures hardware always in known configuration
- **Automatic Transitions**: State changes handled automatically between operations

### 4. Code Maintainability

- **Cleaner Kernels**: Less boilerplate code
- **Easier Debugging**: Clear state tracking and logging capabilities
- **Better Testing**: Deterministic state transitions

## Migration Strategy

### Phase 1: Implementation (Current)
- Implement global state tracking infrastructure
- Create new API versions with global state management
- Mark old uninit functions as deprecated but functional

### Phase 2: Adoption
- Update existing kernels to use new APIs
- Provide migration tools and documentation
- Performance validation and optimization

### Phase 3: Cleanup
- Remove deprecated uninit functions
- Finalize API and documentation
- Long-term maintenance and improvements

## Backward Compatibility

The solution maintains full backward compatibility:

1. **Deprecated Functions**: Old uninit functions become no-ops with deprecation warnings
2. **Gradual Migration**: Existing code continues to work unchanged
3. **Optional Adoption**: Kernels can migrate at their own pace
4. **Clear Timeline**: Deprecation warnings indicate future removal

## Example Usage Comparison

### Before (Current API)
```cpp
// Complex state management required
reduce_init<SUM, REDUCE_ROW>(icb, scaler_cb, ocb);
// ... reduce operations ...
reduce_uninit();  // REQUIRED - easy to forget!

tilize_init(icb, block, ocb);
// ... tilize operations ...
tilize_uninit(icb, ocb);  // REQUIRED - easy to forget!

pack_untilize_init<8>(icb, ocb);
// ... untilize operations ...
pack_untilize_uninit(ocb);  // REQUIRED - easy to forget!
```

### After (Global State API)
```cpp
// Automatic state management
global_config_state_init();  // Once at kernel start

reduce_init<SUM, REDUCE_ROW>(icb, scaler_cb, ocb);
// ... reduce operations ...
// NO UNINIT NEEDED!

tilize_init(icb, block, ocb);  // Automatically handles state transition
// ... tilize operations ...
// NO UNINIT NEEDED!

pack_untilize_init<8>(icb, ocb);  // Automatically handles state transition
// ... untilize operations ...
// NO UNINIT NEEDED!
```

## Testing and Validation

### Unit Tests
- State tracking function correctness
- Configuration change detection accuracy
- Performance regression testing

### Integration Tests
- Multi-operation kernel sequences
- State transition validation
- Hardware configuration verification

### Performance Benchmarks
- Before/after performance comparisons
- Configuration overhead measurements
- Real-world kernel performance impact

## Future Enhancements

### Additional State Tracking
- Math unit configuration state
- Destination accumulation mode tracking
- Circular buffer configuration tracking

### Advanced Optimizations
- Predictive state management
- Cross-operation optimization
- Hardware-specific optimizations

### Tooling and Debugging
- State visualization tools
- Configuration change logging
- Performance profiling integration

## Conclusion

The global configuration state management solution provides a comprehensive approach to eliminating uninit functions while improving performance and reducing API complexity. The implementation maintains full backward compatibility while providing a clear migration path toward a simpler, more efficient programming model.

The solution directly addresses all requirements from Issue #22904:
- ✅ Tracks packer strides configuration
- ✅ Tracks packer row edge masking
- ✅ Tracks unpacker source data formats (srcA & srcB)
- ✅ Eliminates need for uninit functions
- ✅ Improves performance through reduced reconfigurations
- ✅ Maintains backward compatibility
- ✅ Provides clear migration path
