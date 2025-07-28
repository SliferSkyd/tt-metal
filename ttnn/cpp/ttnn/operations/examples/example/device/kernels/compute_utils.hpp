#pragma once

#include <cstdint>

#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif

#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"

constexpr std::uint32_t onetile = 1;

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace ckernel {

union Scalar {
    float f;
    uint32_t u;
};
constexpr Scalar one{.f = 1.0f};
constexpr Scalar zero{.f = 0.0f};

ALWI void enable_pack_reconfig_l1_acc() { PACK((llk_pack_reconfig_l1_acc(1))); }

ALWI void disable_pack_reconfig_l1_acc() { PACK((llk_pack_reconfig_l1_acc(0))); }

uint32_t cb_srca = 256;  // outside of any cbs
uint32_t cb_srcb = 256;  // outside of any cbs
uint32_t cb_pack = 256;  // outside of any cbs

template <bool to_from_int8 = false>
ALWI void _reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    // if (cb_srca == 256 || cb_srcb == 256)
    reconfig_data_format<to_from_int8>(srca_new_operand, srcb_new_operand);
    // else
    //   reconfig_data_format<to_from_int8>(cb_srca, srca_new_operand, cb_srcb,
    //  srcb_new_operand);

    cb_srca = srca_new_operand;
    cb_srcb = srcb_new_operand;
}

template <bool to_from_int8 = false>
ALWI void _reconfig_data_format_srca(const uint32_t srca_new_operand) {
    // if (cb_srca == 256)
    reconfig_data_format_srca<to_from_int8>(srca_new_operand);
    // else
    //   reconfig_data_format_srca<to_from_int8>(cb_srca, srca_new_operand);
    //

    cb_srca = srca_new_operand;
}

template <bool to_from_int8 = false>
ALWI void _reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    // if (cb_srcb == 256)
    reconfig_data_format_srcb<to_from_int8>(srcb_new_operand);
    // else {
    //   reconfig_data_format_srcb<to_from_int8>(cb_srcb, srcb_new_operand);
    // }

    cb_srcb = srcb_new_operand;
}

ALWI void _pack_reconfig_data_format(const uint32_t new_operand) {
    // if (cb_pack == 256)
    pack_reconfig_data_format(new_operand);
    // else
    //   pack_reconfig_data_format(cb_pack, new_operand);

    cb_pack = new_operand;
}

ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb) {
    _pack_reconfig_data_format(icb);
    pack_tile(ifrom_dst, icb);
}

ALWI void copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0) {
    _reconfig_data_format_srca<DST_ACCUM_MODE>(icb);
    copy_tile_to_dst_init_short(icb, transpose);
}

class ArgFetcher {
private:
    int idx = 0;

public:
    template <typename T>
    T next() {
        return get_arg_val<T>(idx++);
    }
};

class CommonArgFetcher {
private:
    int idx = 0;

public:
    template <typename T>
    T next() {
        return get_common_arg_val<T>(idx++);
    }
};

ALWI void pack_onetile_to_cb(uint32_t icb = 16, uint32_t ifrom_dst = 0) {
    tile_regs_wait();
    cb_reserve_back(icb, onetile);
    pack_tile_with_dt(ifrom_dst, icb);
    cb_push_back(icb, onetile);
    tile_regs_release();
}

ALWI bool get_start_id_and_num_unit(
    uint32_t& start_id,
    uint32_t& num_unit,
    uint32_t upcg1,
    uint32_t upcg2,
    uint32_t g1_cores,
    uint32_t g2_cores,
    uint32_t grid_y) {
    uint32_t x = get_relative_logical_x();
    uint32_t y = get_relative_logical_y();
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

constexpr uint32_t div_up(uint32_t n, uint32_t d) { return (n + d - 1) / d; }
}  // namespace ckernel
