// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../../cb_config.hpp"
#include "../compute_utils.hpp"
#include "ttnn/operations/examples/example/device/unary_op_types.hpp"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

template <int NEWTON_ITERATIONS>
sfpi_inline vFloat recip(vFloat v) {
    vUInt magic = 0x7EF477D3;
    vFloat r = reinterpret<vFloat>(magic - reinterpret<vUInt>(v));

#pragma GCC unroll 0
    for (int i = 0; i < NEWTON_ITERATIONS; i++) {
        r = r * (2.0f - r * v);
    }
    v_if(v == vConst0) { r = std::numeric_limits<float>::infinity(); }
    v_endif;
    return r;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_power_exp_int_backward(int param) {
    constexpr int NEWTON_ITERATIONS = 5;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        uint p = (param - 1) >= 0 ? (param - 1) : -(param - 1);
        vFloat dy = dst_reg[0];
        vFloat x = dst_reg[32];
        vFloat dx = 1.0f;
        while (p > 0) {
            if (p & 1) {
                dx *= x;
            }
            x *= x;
            p >>= 1;
        }
        if (param - 1 < 0) {
            dx = recip<NEWTON_ITERATIONS>(dx);
        }
        dx *= ((float)param) * dy;
        dst_reg[0] = dx;
        dst_reg++;
    }
}

#endif

template <UnaryOpType op_type>
ALWI void elemwise_unary_backward_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::power, false>()));
}

template <UnaryOpType op_type>
ALWI void elemwise_unary_backward(uint32_t num_input_cbs, uint32_t i, uint32_t scalar1, uint32_t scalar2) {
    union {
        uint32_t u;
        int32_t i;
    } c = {.u = scalar1};
    MATH((_llk_math_eltwise_unary_sfpu_params_<false>(
        calculate_power_exp_int_backward<false>, num_input_cbs * i, (int)VectorMode::RC, c.i)));
}

namespace NAMESPACE {
void MAIN {
    // Compile Time Arguments
    constexpr UnaryOpType op_type = static_cast<UnaryOpType>(get_compile_time_arg_val(0));
    constexpr uint32_t max_block_size = get_compile_time_arg_val(1);

    // Common Runtime Arguments
    CommonArgFetcher common_arg_fetcher;
    WORK_PARTITION(tiles_offset, num_tiles);
    const auto requires_input = common_arg_fetcher.next<uint32_t>() == 1;
    const auto requires_output = common_arg_fetcher.next<uint32_t>() == 1;
    const auto scalar1 = common_arg_fetcher.next<uint32_t>();
    const auto scalar2 = common_arg_fetcher.next<uint32_t>();
    const auto do_mask_w = common_arg_fetcher.next<uint32_t>() == 1;
    const auto Wt = common_arg_fetcher.next<uint32_t>();

    // Variables
    const auto num_blocks = div_up(num_tiles, max_block_size);
    const auto last_block_size = num_tiles % max_block_size == 0 ? max_block_size : num_tiles % max_block_size;
    const uint32_t num_input_cbs = 1 + (requires_input ? 1 : 0) + (requires_output ? 1 : 0);

    // Circular Buffers
    namespace cb = cb_backward;

    // Compute
    if (requires_input && requires_output) {
        binary_op_init_common(cb::output_grad, cb::output, cb::input_grad);
    } else if (requires_input) {
        binary_op_init_common(cb::output_grad, cb::input, cb::input_grad);
    } else if (requires_output) {
        binary_op_init_common(cb::output_grad, cb::output, cb::input_grad);
    } else {
        unary_op_init_common(cb::output_grad, cb::input_grad);
    }

    if (do_mask_w) {
        cb_wait_front(cb::mask_w, onetile);
    }

    for (uint32_t b = 0; b < num_blocks; b++) {
        const auto block_size = (b == num_blocks - 1) ? last_block_size : max_block_size;

        tile_regs_acquire();
        cb_wait_front(cb::output_grad, block_size);
        if (requires_input) {
            cb_wait_front(cb::input, block_size);
        }
        if (requires_output) {
            cb_wait_front(cb::output, block_size);
        }
        // copy tiles from input CBs to DEST reg
        copy_tile_init_with_dt(cb::output_grad);
        for (uint32_t i = 0; i < block_size; i++) {
            copy_tile(cb::output_grad, i, num_input_cbs * i);
        }
        uint32_t cnt = 1;
        if (requires_input) {
            copy_tile_init_with_dt(cb::input);
            for (uint32_t i = 0; i < block_size; i++) {
                copy_tile(cb::input, i, num_input_cbs * i + cnt);
            }
            cnt++;
        }
        if (requires_output) {
            copy_tile_init_with_dt(cb::output);
            for (uint32_t i = 0; i < block_size; i++) {
                copy_tile(cb::output, i, num_input_cbs * i + cnt);
            }
        }
        // run elemwise unary backward op
        elemwise_unary_backward_init<op_type>();
        for (uint32_t i = 0; i < block_size; i++) {
            elemwise_unary_backward<op_type>(num_input_cbs, i, scalar1, scalar2);
        }
        if (do_mask_w && (tiles_offset + max_block_size * b + block_size - 1) % Wt == Wt - 1) {
            // output mask (always last tile)
            const auto idst_last = num_input_cbs * (block_size - 1);
            copy_tile_init_with_dt(cb::mask_w);
            copy_tile(cb::mask_w, 0, idst_last + 1);
            mask_tile_init();
            mask_tile(idst_last, idst_last + 1);
        }
        cb_pop_front(cb::output_grad, block_size);
        if (requires_input) {
            cb_pop_front(cb::input, block_size);
        }
        if (requires_output) {
            cb_pop_front(cb::output, block_size);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb::input_grad, block_size);
        pack_reconfig_data_format(cb::input_grad);
        for (uint32_t i = 0; i < block_size; i++) {
            pack_tile(num_input_cbs * i, cb::input_grad);
        }
        cb_push_back(cb::input_grad, block_size);
        tile_regs_release();
    }

    if (do_mask_w) {
        cb_pop_front(cb::mask_w, onetile);
    }
}
}  // namespace NAMESPACE
