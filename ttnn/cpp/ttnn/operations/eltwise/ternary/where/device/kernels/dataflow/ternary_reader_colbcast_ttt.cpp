// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    DPRINT << "KERNEL START" << ENDL();

    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    DPRINT << "Got src0_addr: " << src0_addr << ENDL();

    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    DPRINT << "Got start_tile_id: " << start_tile_id << ENDL();

    const uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    DPRINT << "Got src0_num_tiles: " << src0_num_tiles << ENDL();

    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    DPRINT << "Got dst_num_tiles: " << dst_num_tiles << ENDL();

    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    DPRINT << "Got dst_shard_width: " << dst_shard_width << ENDL();

    const uint32_t nD_stride_0 = get_arg_val<uint32_t>(5);
    DPRINT << "Got nD_stride_0: " << nD_stride_0 << ENDL();

    const uint32_t d_stride_0 = get_arg_val<uint32_t>(6);
    DPRINT << "Got d_stride_0: " << d_stride_0 << ENDL();

    const uint32_t n_stride_0 = get_arg_val<uint32_t>(7);
    DPRINT << "Got n_stride_0: " << n_stride_0 << ENDL();

    const uint32_t c_stride_0 = get_arg_val<uint32_t>(8);
    DPRINT << "Got c_stride_0: " << c_stride_0 << ENDL();

    const uint32_t D = get_arg_val<uint32_t>(9);
    DPRINT << "Got D: " << D << ENDL();

    const uint32_t N = get_arg_val<uint32_t>(10);
    DPRINT << "Got N: " << N << ENDL();

    const uint32_t C = get_arg_val<uint32_t>(11);
    DPRINT << "Got C: " << C << ENDL();

    const uint32_t Ht = get_arg_val<uint32_t>(12);
    DPRINT << "Got Ht: " << Ht << ENDL();

    const uint32_t Wt = get_arg_val<uint32_t>(13);
    DPRINT << "Got Wt: " << Wt << ENDL();

    const uint32_t cND = get_arg_val<uint32_t>(14);  // collapsed dims > 5
    DPRINT << "Got cND: " << cND << ENDL();

    const uint32_t src1_addr = get_arg_val<uint32_t>(15);
    DPRINT << "Got src1_addr: " << src1_addr << ENDL();

    const uint32_t nD_stride_1 = get_arg_val<uint32_t>(16);
    DPRINT << "Got nD_stride_1: " << nD_stride_1 << ENDL();

    const uint32_t d_stride_1 = get_arg_val<uint32_t>(17);
    DPRINT << "Got d_stride_1: " << d_stride_1 << ENDL();

    const uint32_t n_stride_1 = get_arg_val<uint32_t>(18);
    DPRINT << "Got n_stride_1: " << n_stride_1 << ENDL();

    const uint32_t c_stride_1 = get_arg_val<uint32_t>(19);
    DPRINT << "Got c_stride_1: " << c_stride_1 << ENDL();

    const uint32_t src1_num_tiles = get_arg_val<uint32_t>(20);
    DPRINT << "Got src1_num_tiles: " << src1_num_tiles << ENDL();

    const uint32_t src2_addr = get_arg_val<uint32_t>(21);
    DPRINT << "Got src2_addr: " << src2_addr << ENDL();

    const uint32_t nD_stride_2 = get_arg_val<uint32_t>(22);
    DPRINT << "Got nD_stride_2: " << nD_stride_2 << ENDL();

    const uint32_t d_stride_2 = get_arg_val<uint32_t>(23);
    DPRINT << "Got d_stride_2: " << d_stride_2 << ENDL();

    const uint32_t n_stride_2 = get_arg_val<uint32_t>(24);
    DPRINT << "Got n_stride_2: " << n_stride_2 << ENDL();

    const uint32_t c_stride_2 = get_arg_val<uint32_t>(25);
    DPRINT << "Got c_stride_2: " << c_stride_2 << ENDL();

    const uint32_t src2_num_tiles = get_arg_val<uint32_t>(26);
    DPRINT << "Got src2_num_tiles: " << src2_num_tiles << ENDL();

    constexpr auto cb_id_src0 = tt::CBIndex::c_0;
    constexpr auto cb_id_src1 = tt::CBIndex::c_1;
    constexpr auto cb_id_src2 = tt::CBIndex::c_2;
    DPRINT << "Set CB IDs" << ENDL();

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool src2_is_dram = get_compile_time_arg_val(5) == 1;
    // DPRINT << "Got DRAM flags: src0=" << src0_is_dram << " src1=" << src1_is_dram << " src2=" << src2_is_dram <<
    // ENDL();

    const uint32_t src0_tile_bytes = get_tile_size(cb_id_src0);
    DPRINT << "Got src0_tile_bytes: " << src0_tile_bytes << ENDL();

    const uint32_t src1_tile_bytes = get_tile_size(cb_id_src1);
    DPRINT << "Got src1_tile_bytes: " << src1_tile_bytes << ENDL();

    const uint32_t src2_tile_bytes = get_tile_size(cb_id_src2);
    DPRINT << "Got src2_tile_bytes: " << src2_tile_bytes << ENDL();

    const DataFormat src0_data_format = get_dataformat(cb_id_src0);
    DPRINT << "Got src0_data_format" << ENDL();

    const DataFormat src1_data_format = get_dataformat(cb_id_src1);
    DPRINT << "Got src1_data_format" << ENDL();

    const DataFormat src2_data_format = get_dataformat(cb_id_src2);
    DPRINT << "Got src2_data_format" << ENDL();

    const InterleavedAddrGenFast<src0_is_dram> src0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
    DPRINT << "Created src0 addr gen" << ENDL();

    const InterleavedAddrGenFast<src1_is_dram> src1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};
    DPRINT << "Created src1 addr gen" << ENDL();

    const InterleavedAddrGenFast<src2_is_dram> src2 = {
        .bank_base_address = src2_addr, .page_size = src2_tile_bytes, .data_format = src2_data_format};
    DPRINT << "Created src2 addr gen" << ENDL();

    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(1) == 1;
    // DPRINT << "Got has_sharding: " << has_sharding << ENDL();

    const uint32_t HtWt = Ht * Wt;
    // DPRINT << "Calculated HtWt: " << HtWt << ENDL();

    const uint32_t tiles_per_n = C * HtWt;
    // DPRINT << "Calculated tiles_per_n: " << tiles_per_n << ENDL();

    const uint32_t tiles_per_d = N * tiles_per_n;
    // DPRINT << "Calculated tiles_per_d: " << tiles_per_d << ENDL();

    const uint32_t tiles_per_nd = D * tiles_per_d;
    // DPRINT << "Calculated tiles_per_nd: " << tiles_per_nd << ENDL();

    const uint32_t offset_nd = start_tile_id % tiles_per_nd;
    // DPRINT << "Calculated offset_nd: " << offset_nd << ENDL();

    const uint32_t offset_d = offset_nd % tiles_per_d;
    // DPRINT << "Calculated offset_d: " << offset_d << ENDL();

    const uint32_t offset_n = offset_d % tiles_per_n;
    // DPRINT << "Calculated offset_n: " << offset_n << ENDL();

    const uint32_t offset_c = offset_n % HtWt;
    // DPRINT << "Calculated offset_c: " << offset_c << ENDL();

    uint32_t start_nd = start_tile_id / tiles_per_nd;
    // DPRINT << "Calculated start_nd: " << start_nd << ENDL();

    uint32_t start_d = offset_nd / tiles_per_d;
    // DPRINT << "Calculated start_d: " << start_d << ENDL();

    uint32_t start_n = offset_d / tiles_per_n;
    // DPRINT << "Calculated start_n: " << start_n << ENDL();
    uint32_t start_c = offset_n / HtWt;
    // DPRINT << "Calculated start_c: " << start_c << ENDL();

    uint32_t start_th = offset_c / Wt;
    // DPRINT << "Calculated start_th: " << start_th << ENDL();

    uint32_t start_tw = offset_c % Wt;
    // DPRINT << "Calculated start_tw: " << start_tw << ENDL();

    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;
    // DPRINT << "Calculated end_tw: " << end_tw << ENDL();

    DPRINT << "src0_addr === " << src0_addr << ENDL();
    DPRINT << "src1_addr === " << src1_addr << ENDL();
    DPRINT << "src2_addr === " << src2_addr << ENDL();

    // this is the INPUT tile offset for src0
    uint32_t tile_offset_0 =
        start_nd * nD_stride_0 + start_d * d_stride_0 + start_n * n_stride_0 + start_c * c_stride_0;
    // DPRINT << "Initial tile_offset_0: " << tile_offset_0 << ENDL();

#if !SRC0_BCAST
    tile_offset_0 += start_th * Wt;
    // DPRINT << "Updated tile_offset_0 (no bcast): " << tile_offset_0 << ENDL();
#endif

    uint32_t next_c_shift_0 = c_stride_0 - HtWt;
    uint32_t next_n_shift_0 = n_stride_0 - c_stride_0 * C;
    uint32_t next_d_shift_0 = d_stride_0 - n_stride_0 * N;
    uint32_t next_nd_shift_0 = nD_stride_0 - d_stride_0 * D;
    // DPRINT << "SRC0 shifts: next_c=" << next_c_shift_0 << " next_n=" << next_n_shift_0 << " next_d=" <<
    // next_d_shift_0 << " next_nd=" << next_nd_shift_0 << ENDL();

    // this is the INPUT tile offset for src1
    uint32_t tile_offset_1 =
        start_nd * nD_stride_1 + start_d * d_stride_1 + start_n * n_stride_1 + start_c * c_stride_1;
    DPRINT << "Initial tile_offset_1: " << tile_offset_1 << ENDL();

#if !SRC1_BCAST
    tile_offset_1 += start_th * Wt;
    DPRINT << "Updated tile_offset_1 (no bcast): " << tile_offset_1 << ENDL();
#endif

    uint32_t next_c_shift_1 = c_stride_1 - HtWt;
    uint32_t next_n_shift_1 = n_stride_1 - c_stride_1 * C;
    uint32_t next_d_shift_1 = d_stride_1 - n_stride_1 * N;
    uint32_t next_nd_shift_1 = nD_stride_1 - d_stride_1 * D;
    // DPRINT << "SRC1 shifts: next_c=" << next_c_shift_1 << " next_n=" << next_n_shift_1 << " next_d=" <<
    // next_d_shift_1 << " next_nd=" << next_nd_shift_1 << ENDL();

    // this is the INPUT tile offset for src2
    uint32_t tile_offset_2 =
        start_nd * nD_stride_2 + start_d * d_stride_2 + start_n * n_stride_2 + start_c * c_stride_2;
    DPRINT << "Initial tile_offset_2: " << tile_offset_2 << ENDL();

#if !SRC2_BCAST
    tile_offset_2 += start_th * Wt;
    DPRINT << "Updated tile_offset_2 (no bcast): " << tile_offset_2 << ENDL();
#endif

    uint32_t next_c_shift_2 = c_stride_2 - HtWt;
    uint32_t next_n_shift_2 = n_stride_2 - c_stride_2 * C;
    uint32_t next_d_shift_2 = d_stride_2 - n_stride_2 * N;
    uint32_t next_nd_shift_2 = nD_stride_2 - d_stride_2 * D;
    // DPRINT << "SRC2 shifts: next_c=" << next_c_shift_2 << " next_n=" << next_n_shift_2 << " next_d=" <<
    // next_d_shift_2 << " next_nd=" << next_nd_shift_2 << ENDL();

    uint32_t num_tiles_read = 0;
    DPRINT << "Starting main loops with cND=" << cND << " D=" << D << " N=" << N << " C=" << C << " Ht=" << Ht
           << " dst_num_tiles=" << dst_num_tiles << ENDL();

    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        DPRINT << "Loop nd=" << nd << " num_tiles_read=" << num_tiles_read << ENDL();

        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            DPRINT << "  Loop d=" << d << " num_tiles_read=" << num_tiles_read << ENDL();

            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                DPRINT << "    Loop n=" << n << " num_tiles_read=" << num_tiles_read << ENDL();

                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    DPRINT << "      Loop c=" << c << " num_tiles_read=" << num_tiles_read << ENDL();

                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        DPRINT << "        Loop th=" << th << " num_tiles_read=" << num_tiles_read << ENDL();
#if SRC0_BCAST
                        DPRINT << "          SRC0_BCAST: reading tile " << (tile_offset_0 + th) << ENDL();
                        cb_reserve_back(cb_id_src0, onetile);
                        DPRINT << "          SRC0_BCAST: reserved CB" << ENDL();
                        uint32_t l1_write_addr_src0 = get_write_ptr(cb_id_src0);
                        DPRINT << "          SRC0_BCAST: got write ptr" << ENDL();
                        noc_async_read_tile(tile_offset_0 + th, src0, l1_write_addr_src0);
                        DPRINT << "          SRC0_BCAST: started async read" << ENDL();
                        noc_async_read_barrier();
                        DPRINT << "          SRC0_BCAST: finished async read" << ENDL();
                        FILL_TILE_WITH_FIRST_COLUMN(cb_id_src0);
                        DPRINT << "          SRC0_BCAST: filled tile with first column" << ENDL();
                        cb_push_back(cb_id_src0, onetile);
                        DPRINT << "          SRC0_BCAST: pushed to CB" << ENDL();
#endif
#if SRC1_BCAST
                        DPRINT << "          SRC1_BCAST: reading tile " << (tile_offset_1 + th) << ENDL();
                        cb_reserve_back(cb_id_src1, onetile);
                        DPRINT << "          SRC1_BCAST: reserved CB" << ENDL();
                        uint32_t l1_write_addr_src1 = get_write_ptr(cb_id_src1);
                        DPRINT << "          SRC1_BCAST: got write ptr" << ENDL();
                        noc_async_read_tile(tile_offset_1 + th, src1, l1_write_addr_src1);
                        DPRINT << "          SRC1_BCAST: started async read" << ENDL();
                        noc_async_read_barrier();
                        DPRINT << "          SRC1_BCAST: finished async read" << ENDL();
                        FILL_TILE_WITH_FIRST_COLUMN_B(cb_id_src1);
                        DPRINT << "          SRC1_BCAST: filled tile with first column" << ENDL();
                        cb_push_back(cb_id_src1, onetile);
                        DPRINT << "          SRC1_BCAST: pushed to CB" << ENDL();
#endif
#if SRC2_BCAST
                        // DPRINT << "          SRC2_BCAST: reading tile " << (tile_offset_2 + th) << ENDL();
                        cb_reserve_back(cb_id_src2, onetile);
                        // DPRINT << "          SRC2_BCAST: reserved CB" << ENDL();
                        uint32_t l1_write_addr_src2 = get_write_ptr(cb_id_src2);
                        // DPRINT << "          SRC2_BCAST: got write ptr" << ENDL();
                        noc_async_read_tile(tile_offset_2 + th, src2, l1_write_addr_src2);
                        // DPRINT << "          SRC2_BCAST: started async read" << ENDL();
                        noc_async_read_barrier();
                        // DPRINT << "          SRC2_BCAST: finished async read" << ENDL();
                        FILL_TILE_WITH_FIRST_COLUMN_C(cb_id_src2);
                        // DPRINT << "          SRC2_BCAST: filled tile with first column" << ENDL();
                        cb_push_back(cb_id_src2, onetile);
                        // DPRINT << "          SRC2_BCAST: pushed to CB" << ENDL();
#endif
                        // DPRINT << "          Starting tw loop: start_tw=" << start_tw << " end_tw=" << end_tw <<
                        // ENDL();
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            // DPRINT << "            tw=" << tw << " num_tiles_read=" << num_tiles_read << ENDL();
#if !SRC0_BCAST
                            // DPRINT << "            SRC0_NO_BCAST: reading tile " << (tile_offset_0 + tw) << ENDL();
                            cb_reserve_back(cb_id_src0, onetile);
                            uint32_t l1_write_addr_0 = get_write_ptr(cb_id_src0);
                            noc_async_read_tile(tile_offset_0 + tw, src0, l1_write_addr_0);
                            noc_async_read_barrier();
                            cb_push_back(cb_id_src0, onetile);
                            // DPRINT << "            SRC0_NO_BCAST: completed" << ENDL();
#endif
#if !SRC1_BCAST
                            DPRINT << "            SRC1_NO_BCAST: reading tile " << (tile_offset_1 + tw) << ENDL();
                            cb_reserve_back(cb_id_src1, onetile);
                            uint32_t l1_write_addr_1 = get_write_ptr(cb_id_src1);
                            noc_async_read_tile(tile_offset_1 + tw, src1, l1_write_addr_1);
                            noc_async_read_barrier();
                            cb_push_back(cb_id_src1, onetile);
                            DPRINT << "            SRC1_NO_BCAST: completed" << ENDL();
#endif
#if !SRC2_BCAST
                            DPRINT << "            SRC2_NO_BCAST: reading tile " << (tile_offset_2 + tw) << ENDL();
                            cb_reserve_back(cb_id_src2, onetile);
                            uint32_t l1_write_addr_2 = get_write_ptr(cb_id_src2);
                            noc_async_read_tile(tile_offset_2 + tw, src2, l1_write_addr_2);
                            noc_async_read_barrier();
                            cb_push_back(cb_id_src2, onetile);
                            DPRINT << "            SRC2_NO_BCAST: completed" << ENDL();
#endif
                        }
                        DPRINT << "          Finished tw loop" << ENDL();
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                            DPRINT << "        Reset start_tw=0 (no sharding)" << ENDL();
                        }
                        DPRINT << "        Updating tile offsets for next th" << ENDL();
#if !SRC0_BCAST
                        tile_offset_0 += Wt;
                        DPRINT << "        SRC0_NO_BCAST: tile_offset_0 += Wt -> " << tile_offset_0 << ENDL();
#endif
#if !SRC1_BCAST
                        tile_offset_1 += Wt;
                        DPRINT << "        SRC1_NO_BCAST: tile_offset_1 += Wt -> " << tile_offset_1 << ENDL();
#endif
#if !SRC2_BCAST
                        tile_offset_2 += Wt;
                        DPRINT << "        SRC2_NO_BCAST: tile_offset_2 += Wt -> " << tile_offset_2 << ENDL();
#endif
                    }
                    DPRINT << "      Finished th loop, updating tile offsets for next c" << ENDL();
#if SRC0_BCAST
                    // same as following logically
                    // tile_offset_0 += HtWt;
                    // tile_offset_0 += next_c_shift_0;
                    tile_offset_0 += c_stride_0;
                    DPRINT << "      SRC0_BCAST: tile_offset_0 += c_stride_0 -> " << tile_offset_0 << ENDL();
#else
                    tile_offset_0 += next_c_shift_0;
                    DPRINT << "      SRC0_NO_BCAST: tile_offset_0 += next_c_shift_0 -> " << tile_offset_0 << ENDL();
#endif
#if SRC1_BCAST
                    tile_offset_1 += c_stride_1;
                    DPRINT << "      SRC1_BCAST: tile_offset_1 += c_stride_1 -> " << tile_offset_1 << ENDL();
#else
                    tile_offset_1 += next_c_shift_1;
                    DPRINT << "      SRC1_NO_BCAST: tile_offset_1 += next_c_shift_1 -> " << tile_offset_1 << ENDL();
#endif
#if SRC2_BCAST
                    tile_offset_2 += c_stride_2;
                    DPRINT << "      SRC2_BCAST: tile_offset_2 += c_stride_2 -> " << tile_offset_2 << ENDL();
#else
                    tile_offset_2 += next_c_shift_2;
                    DPRINT << "      SRC2_NO_BCAST: tile_offset_2 += next_c_shift_2 -> " << tile_offset_2 << ENDL();
#endif
                }
                DPRINT << "    Finished c loop, updating tile offsets for next n" << ENDL();
                tile_offset_0 += next_n_shift_0;
                tile_offset_1 += next_n_shift_1;
                tile_offset_2 += next_n_shift_2;
                DPRINT << "    Updated tile offsets: 0=" << tile_offset_0 << " 1=" << tile_offset_1
                       << " 2=" << tile_offset_2 << ENDL();
            }
            DPRINT << "  Finished n loop, updating tile offsets for next d" << ENDL();
            tile_offset_0 += next_d_shift_0;
            tile_offset_1 += next_d_shift_1;
            tile_offset_2 += next_d_shift_2;
            DPRINT << "  Updated tile offsets: 0=" << tile_offset_0 << " 1=" << tile_offset_1 << " 2=" << tile_offset_2
                   << ENDL();
        }
        DPRINT << "Finished d loop, updating tile offsets for next nd" << ENDL();
        tile_offset_0 += next_nd_shift_0;
        tile_offset_1 += next_nd_shift_1;
        tile_offset_2 += next_nd_shift_2;
        DPRINT << "Updated tile offsets: 0=" << tile_offset_0 << " 1=" << tile_offset_1 << " 2=" << tile_offset_2
               << ENDL();
    }
    DPRINT << "KERNEL COMPLETE - Total tiles read: " << num_tiles_read << ENDL();
}
