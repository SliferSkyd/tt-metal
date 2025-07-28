#pragma once

#include <stdint.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

#include "dataflow_api.h"

namespace {
constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH = 16;
constexpr std::uint32_t FACE_HW = FACE_HEIGHT * FACE_WIDTH;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH = 32;
constexpr std::uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
constexpr std::uint32_t onetile = 1;

// deprecated. minimum read size is not 32
constexpr std::uint32_t NOC_MINIMUM_READ_SIZE = 32;  // 32 Bytes

constexpr std::uint32_t SHARED_EXPONENT_PER_TILE = 64;
constexpr std::uint32_t BFP8_PER_SHARED_EXPONENT = 16;

union Scalar {
    float f;
    uint32_t u;
};
constexpr Scalar one{.f = 1.0f};
constexpr Scalar zero{.f = 0.0f};

constexpr uint16_t u16_one = 0x3F80;   // bfloat16 representation of 1.0
constexpr uint16_t u16_zero = 0x0000;  // bfloat16 representation of 0.0

constexpr uint32_t bfp8_one = 0x3FC00000;

constexpr uint32_t div_up(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

class CommonArgFetcher {
public:
    int idx = 0;

    template <typename T>
    T next() {
        return get_common_arg_val<T>(idx++);
    }
};

FORCE_INLINE bool get_start_id_and_num_unit(
    uint32_t& start_id,
    uint32_t& num_unit,
    uint32_t upcg1,
    uint32_t upcg2,
    uint32_t g1_cores,
    uint32_t g2_cores,
    uint32_t grid_y) {
    uint32_t x = get_absolute_logical_x();
    uint32_t y = get_absolute_logical_y();
    uint32_t core_linear_id = x * grid_y + y;
    uint32_t total_cores = g1_cores + g2_cores;

    if (core_linear_id >= total_cores) {
        return false;
    }

    if (core_linear_id < g1_cores) {
        num_unit = upcg1;
        start_id = core_linear_id * upcg1;
    } else {
        num_unit = upcg2;
        start_id = g1_cores * upcg1 + (core_linear_id - g1_cores) * upcg2;
    }

    return num_unit > 0;
}

#define WORK_PARTITION(start_id, num_unit)                                                                  \
    uint32_t start_id, num_unit;                                                                            \
    const auto __upcg1 = common_arg_fetcher.next<uint32_t>();                                               \
    const auto __upcg2 = common_arg_fetcher.next<uint32_t>();                                               \
    const auto __g1_cores = common_arg_fetcher.next<uint32_t>();                                            \
    const auto __g2_cores = common_arg_fetcher.next<uint32_t>();                                            \
    const auto __grid_y = common_arg_fetcher.next<uint32_t>();                                              \
    if (!get_start_id_and_num_unit(start_id, num_unit, __upcg1, __upcg2, __g1_cores, __g2_cores, __grid_y)) \
        return;

FORCE_INLINE void fill_bf16_cb(uint8_t cb, uint32_t value, uint32_t n) {
    uint16_t* ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb));
    uint16_t casted_value = static_cast<uint16_t>(value >> 16);
    std::fill_n(ptr, n, casted_value);
}

FORCE_INLINE void fill_bfp8_cb(uint8_t cb, uint32_t value, uint32_t n) {
    const uint8_t shared_exponent = static_cast<uint8_t>((value >> 23) & 0xFF);
    const uint8_t sign_mantissa = static_cast<uint8_t>(
        ((value >> 31) << 7) |  // Sign Bit
        ((value >> 16) & 0x7F)  // Mantissa
    );
    const uint32_t n_shared_exponent = div_up(n, BFP8_PER_SHARED_EXPONENT);
    uint8_t* ptr = reinterpret_cast<uint8_t*>(get_write_ptr(cb));
    std::memset(ptr, shared_exponent, n_shared_exponent);
    std::memset(ptr + SHARED_EXPONENT_PER_TILE, sign_mantissa, n);
}

FORCE_INLINE void generate_bfp8_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint8_t*>(get_write_ptr(cb_mask));
    auto blk_head = ptr + 64;

    fill_bfp8_cb(cb_mask, zero.u, 1024);
    memset(ptr, 127, 64);

    const uint8_t exponent = static_cast<uint8_t>((bfp8_one >> 23) & 0xFF);
    const uint8_t sign_mantissa = static_cast<uint8_t>(
        ((bfp8_one >> 31) << 7) |  // Sign Bit
        ((bfp8_one >> 16) & 0x7F)  // Mantissa
    );

    uint32_t h = 0;
    uint32_t index;
    auto populate_subtile = [&](uint32_t offset, uint32_t mask_w) {
        memset(blk_head + offset + h * 16, sign_mantissa, mask_w);
    };

    for (; h < 16; h++) {
        populate_subtile(0, std::min<uint32_t>(mask_w, 16));     // sub 0
        populate_subtile(256, (mask_w < 16) ? 0 : mask_w - 16);  // sub 1
        populate_subtile(512, std::min<uint32_t>(mask_w, 16));   // sub 2
        populate_subtile(768, (mask_w < 16) ? 0 : mask_w - 16);  // sub 3
    }

    cb_push_back(cb_mask, 1);
}

FORCE_INLINE void generate_bf16_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));

    uint32_t h = 0;
    auto populate_subtile = [&](uint32_t offset, uint32_t mask_w) {
        uint32_t w = 0;
        for (; w < mask_w; w++) {
            ptr[h * 16 + w + offset] = u16_one;
        }
        for (; w < 16; w++) {
            ptr[h * 16 + w + offset] = u16_zero;
        }
    };
    for (; h < 16; h++) {
        populate_subtile(0, std::min<uint32_t>(mask_w, 16));     // sub 0
        populate_subtile(256, (mask_w < 16) ? 0 : mask_w - 16);  // sub 1
        populate_subtile(512, std::min<uint32_t>(mask_w, 16));   // sub 2
        populate_subtile(768, (mask_w < 16) ? 0 : mask_w - 16);  // sub 3
    }

    cb_push_back(cb_mask, 1);
}

FORCE_INLINE void generate_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    const DataFormat data_format = get_dataformat(cb_mask);
    switch (data_format) {
        case (DataFormat::Float16_b): generate_bf16_mask_w(cb_mask, mask_w); break;
        case (DataFormat::Bfp8_b): generate_bfp8_mask_w(cb_mask, mask_w); break;
        default: break;
    }
}
}  // namespace
