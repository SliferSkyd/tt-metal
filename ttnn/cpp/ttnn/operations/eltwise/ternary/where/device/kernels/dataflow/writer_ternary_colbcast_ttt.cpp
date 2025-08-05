// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(3);
    const uint32_t nD_stride = get_arg_val<uint32_t>(4);
    const uint32_t d_stride = get_arg_val<uint32_t>(5);
    const uint32_t n_stride = get_arg_val<uint32_t>(6);
    const uint32_t c_stride = get_arg_val<uint32_t>(7);
    const uint32_t D = get_arg_val<uint32_t>(8);
    const uint32_t N = get_arg_val<uint32_t>(9);
    const uint32_t C = get_arg_val<uint32_t>(10);
    const uint32_t Ht = get_arg_val<uint32_t>(11);
    const uint32_t Wt = get_arg_val<uint32_t>(12);
    const uint32_t cND = get_arg_val<uint32_t>(13);  // collapsed dims > 5

    constexpr uint32_t cb_id_dst = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool has_sharding = get_compile_time_arg_val(2) == 1;

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = get_tile_size(cb_id_dst), .data_format = get_dataformat(cb_id_dst)};

    uint32_t offset_dst = 0;
    const uint32_t HtWt = Ht * Wt;

    if constexpr (has_sharding) {
        offset_dst = start_tile_id % HtWt;
    } else {
        offset_dst = start_tile_id;
    }

    uint32_t tile_offset = offset_dst;
    uint32_t num_tiles_written = 0;

    // Single-tile ublocks
    constexpr uint32_t onetile = 1;

    uint32_t start_tw = offset_dst % Wt;
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    for (uint32_t nd = 0; nd < cND && num_tiles_written < dst_num_tiles; ++nd) {
        for (uint32_t d = 0; d < D && num_tiles_written < dst_num_tiles; ++d) {
            for (uint32_t n = 0; n < N && num_tiles_written < dst_num_tiles; ++n) {
                for (uint32_t c = 0; c < C && num_tiles_written < dst_num_tiles; ++c) {
                    for (uint32_t th = 0; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            cb_wait_front(cb_id_dst, onetile);
                            uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                            noc_async_write_tile(tile_offset + tw, s, l1_read_addr);
                            cb_pop_front(cb_id_dst, onetile);
                        }

                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                    }
                    tile_offset += c_stride;
                }
                tile_offset += n_stride;
            }
            tile_offset += d_stride;
        }
        tile_offset += nD_stride;
    }

    noc_async_write_barrier();
}
