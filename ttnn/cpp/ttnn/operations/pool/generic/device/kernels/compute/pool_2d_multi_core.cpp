// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/common.h"
#include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

template <uint32_t num_tiles, uint32_t split_reader, uint32_t num_pages_to_8>
inline void eltwise_mul_tiles(
    const uint32_t in_cb_id_0,
    const uint32_t in_cb_id_1,
    const uint32_t weight_cb_id,
    const uint32_t in_stick_index,
    const uint32_t mul_cb_id) {
    const uint32_t curr_in_cb_id = (split_reader && (in_stick_index & 0x1)) ? in_cb_id_1 : in_cb_id_0;
    cb_reserve_back(mul_cb_id, num_pages_to_8);
    UNPACK((llk_unpack_tilize_uninit(curr_in_cb_id)));
    PACK((pack_untilize_uninit(mul_cb_id)));
    mul_tiles_init(curr_in_cb_id, weight_cb_id);

    tile_regs_acquire();
    for (uint32_t j = 0; j < num_pages_to_8; j++) {
        // uint64_t start_cycles_loop = read_wall_clock();
        cb_wait_front(curr_in_cb_id, 1);

        for (uint32_t i = 0; i < num_tiles; ++i) {
            // UNPACK(( DPRINT << "now it is tile num: " << j * num_tiles + i << ENDL()));
            // UNPACK (( tt::compute::common::print_full_tile(curr_in_cb_id, i) ));
            // UNPACK(( DPRINT << "now it is weight tile num: " << j * num_tiles + i << ENDL()));
            // UNPACK (( tt::compute::common::print_full_tile(weight_cb_id, 0) ));
            mul_tiles(curr_in_cb_id, weight_cb_id, i, 0, j * num_tiles + i);
            // dprint_tensix_dest_reg(j * num_tiles + i);
        }
        cb_pop_front(curr_in_cb_id, 1);
    }

    // dprint_tensix_dest_reg(0);
    // dprint_tensix_dest_reg(1);
    // dprint_tensix_dest_reg(2);
    // dprint_tensix_dest_reg(3);
    // dprint_tensix_dest_reg(4);
    // dprint_tensix_dest_reg(5);
    // dprint_tensix_dest_reg(6);
    // dprint_tensix_dest_reg(7);
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t j = 0; j < num_pages_to_8; j++) {
        // uint64_t start_cycles_loop = read_wall_clock();
        for (uint32_t i = 0; i < num_tiles; ++i) {
            // uint64_t start_cycles_wait = read_wall_clock();
            pack_tile(j * num_tiles + i, mul_cb_id, j * num_tiles + i);  // packer zabo
        }
        cb_push_back(mul_cb_id, 1);
    }

    // PACK((DPRINT << "now it is tile num: " << 0 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 0)));
    // PACK((DPRINT << "now it is curr tile num: " << 1 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 1)));
    // PACK((DPRINT << "now it is tile num: " << 2 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 2)));
    // PACK((DPRINT << "now it is curr tile num: " << 3 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 3)));
    // PACK((DPRINT << "now it is tile num: " << 4 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 4)));
    // PACK((DPRINT << "now it is curr tile num: " << 5 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 5)));
    // PACK((DPRINT << "now it is tile num: " << 6 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 6)));
    // PACK((DPRINT << "now it is curr tile num: " << 7 << ENDL()));
    // PACK((tt::compute::common::print_full_tile(mul_cb_id, 7)));
    // cb_push_back(mul_cb_id, num_pages_to_8);
    tile_regs_release();
}

template <
    uint32_t num_output_tiles,
    bool is_partial_tile,
    uint32_t split_reader,
    uint32_t unpA_face_r_dim,
    uint32_t num_faces_in_tile,
    bool neginf_srca_maxpool,
    bool zero_srca_avgpool,
    uint32_t num_pages_to_8>
inline void reduce_h_fused(
    const uint32_t in_cb_id_0,
    const uint32_t in_cb_id_1,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id,
    const uint32_t mul_cb_id) {
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t num_output_faces = (is_partial_tile ? 1 : 2);

    const uint32_t curr_in_cb_id = mul_cb_id;

    cb_wait_front(curr_in_cb_id, num_pages_to_8);

    // UNPACK((DPRINT << "scaler: " << 0 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(in_scalar_cb_id, 0)));

    // UNPACK((DPRINT << "now it is tile num: " << 0 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 0)));
    // UNPACK((DPRINT << "now it is tile num: " << 1 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 1)));
    // UNPACK((DPRINT << "now it is tile num: " << 2 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 2)));
    // UNPACK((DPRINT << "now it is tile num: " << 3 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 3)));

    // UNPACK((DPRINT << "now it is tile num: " << 4 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 4)));
    // UNPACK((DPRINT << "now it is tile num: " << 5 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 5)));
    // UNPACK((DPRINT << "now it is tile num: " << 6 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 6)));
    // UNPACK((DPRINT << "now it is tile num: " << 7 << ENDL()));
    // UNPACK((tt::compute::common::print_full_tile(curr_in_cb_id, 7)));

    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
        mul_cb_id, in_scalar_cb_id, num_output_tiles, num_faces_in_tile, unpA_face_r_dim, 1)));
    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>()));
    PACK((llk_pack_untilize_init<num_output_tiles, num_output_tiles, false, false, TILE_C_DIM>(
        out_cb_id, num_out_rows, num_faces_in_tile)));
    PACK((llk_init_packer_dest_offset_registers<true, false>()));

    tile_regs_acquire();
    cb_reserve_back(out_cb_id, num_pages_to_8 * num_output_tiles);
    for (uint32_t j = 0; j < num_pages_to_8; j++) {
        unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
            curr_in_cb_id, in_scalar_cb_id, num_output_tiles, 0, num_faces_in_tile, unpA_face_r_dim);
        for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
            // DPRINT << "now it is tile num: " << j * num_output_tiles + c_i << ENDL();
            reduce_tile_math(j * num_output_tiles + c_i, num_faces_in_tile);
        }
        // dprint_tensix_dest_reg(0);
        // dprint_tensix_dest_reg(1);
        // dprint_tensix_dest_reg(2);

        cb_pop_front(curr_in_cb_id, 1);
    }
    // dprint_tensix_dest_reg(0);
    // dprint_tensix_dest_reg(1);
    // dprint_tensix_dest_reg(2);
    // dprint_tensix_dest_reg(3);
    // dprint_tensix_dest_reg(4);
    // dprint_tensix_dest_reg(5);
    // dprint_tensix_dest_reg(6);
    // dprint_tensix_dest_reg(7);
    tile_regs_commit();

    // num_output_tiles takodje govori koja je sirina tensora
    tile_regs_wait();

    // jedan page je velicine 64, sto su 32 podatka. imam 64 page-a. znaci ako radim 8 velikih iteracija, u svakoj
    // popunjavam 8 page-a znaci 1 tile kanala je 1 page, a kada uradim pack_untilize_dest, ja sam zapravo popunio 2
    // page-a

    for (uint32_t i = 0; i < num_pages_to_8; i++) {  // todo: morace sve u jednu petlju
        pack_untilize_dest<num_output_tiles>(
            out_cb_id,
            1 /*out_subblock_h*/,
            0,
            num_out_rows,
            num_output_faces,
            i * num_output_tiles); /* pack 1 row (1x16 or 1x32) */
        // PACK(( DPRINT << "now it is tile num: " << 0 << ENDL()));
        // PACK(( tt::compute::common::print_full_tile(out_cb_id, 0) ));
        // PACK(( DPRINT << "now it is curr tile num: " << 1 << ENDL()));
        // PACK(( tt::compute::common::print_full_tile(out_cb_id, 1) ));
        // PACK(( DPRINT << "packerr: "  << i << ENDL()));
        // PACK(( tt::compute::common::print_full_tile(out_cb_id, 0, true) ));
        cb_push_back(out_cb_id, num_output_tiles);
    }

    tile_regs_release();
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(11);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t weight_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(14);

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = window_size_hw > 16 ? 4 : (is_partial_tile ? 1 : 2);
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    constexpr uint32_t num_pages_to_8 = 8 / in_ntiles_c;

    DPRINT << "num_pages_to_8: " << num_pages_to_8 << ENDL();

    // In case we have <=16 sticks we will use only upper two faces of the tile.
    // In this case we can configure reduce to only process as many rows as needed.
    // In case #sticks > 16 we need bottom two faces as well, and we need to configure reduce to
    // process all rows per face. In the case we rely on reader kernel to put "clear value"
    // in datums which are not used.
    constexpr uint32_t face_r_dim = window_size_hw > 16 ? 16 : window_size_hw;

    // mul_tiles_init(in_cb_id_0, weight_cb_id);
    // if (split_reader) {
    //     mul_tiles_init(in_cb_id_1, weight_cb_id);
    // }

    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        mul_cb_id, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_tile, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_tile);

    // tilize reconfiguration is needed if we have more than one block and the number of tiles
    // is not a multiple of MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig_needed = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0;
    if (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }
    for (uint32_t i = 0; i < nsticks_per_core / num_pages_to_8; ++i) {
        const uint32_t curr_scalar_cb_id =
            (split_reader && (i & 0x1) && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }
        // perform the reduction over the first N - 1 whole chunks
        if constexpr (tilize_reconfig_needed) {
            UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                mul_cb_id, curr_scalar_cb_id, max_tiles_per_iter, num_faces_in_tile, face_r_dim, 1)));
        }

        for (uint32_t b_i = 0; b_i < in_nblocks_c - 1; ++b_i) {
            eltwise_mul_tiles<max_tiles_per_iter, split_reader, num_pages_to_8>(
                in_cb_id_0, in_cb_id_1, weight_cb_id, i, mul_cb_id);

            reduce_h_fused<
                max_tiles_per_iter,
                is_partial_tile,
                split_reader,
                face_r_dim,
                num_faces_in_tile,
                neginf_srca_maxpool,
                zero_srca_avgpool,
                num_pages_to_8>(in_cb_id_0, in_cb_id_1, curr_scalar_cb_id, i, out_cb_id, mul_cb_id);
        }

        if constexpr (tilize_reconfig_needed) {
            UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                mul_cb_id, curr_scalar_cb_id, partial_iter_output_tiles, num_faces_in_tile, face_r_dim, 1)));
        }
        // perform the reduction over the either whole or partial chunk N

        // DPRINT << "partial_iter_output_tiles: " << partial_iter_output_tiles << ENDL();

        // UNPACK((DPRINT << "radi" << ENDL()));

        eltwise_mul_tiles<partial_iter_output_tiles, split_reader, num_pages_to_8>(
            in_cb_id_0, in_cb_id_1, weight_cb_id, i, mul_cb_id);

        reduce_h_fused<
            partial_iter_output_tiles,
            is_partial_tile,
            split_reader,
            face_r_dim,
            num_faces_in_tile,
            neginf_srca_maxpool,
            zero_srca_avgpool,
            num_pages_to_8>(in_cb_id_0, in_cb_id_1, curr_scalar_cb_id, i, out_cb_id, mul_cb_id);

        // UNPACK((DPRINT << "uradio!" << ENDL()));
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
    if constexpr (one_scalar_per_core) {
        cb_pop_front(in_scalar_cb_id_0, 1);
    }
}

}  // namespace NAMESPACE
