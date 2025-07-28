// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "../common.hpp"
#include "../../cb_config.hpp"
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t onetile = 1;

    // Common Runtime Arguments
    CommonArgFetcher common_arg_fetcher;
    WORK_PARTITION(start_id, num_tiles);
    const auto buffer_address = common_arg_fetcher.next<uint32_t>();
    const auto cb = common_arg_fetcher.next<uint32_t>();

    const uint32_t tile_size = get_tile_size(cb);
    const DataFormat data_format = get_dataformat(cb);

    constexpr uint32_t base_cta_idx = 0;
    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = buffer_address, .page_size = tile_size, .data_format = data_format};

    // Write
    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb, 1);
        const uint32_t cb_read_ptr = get_read_ptr(cb);
        auto noc_addr = get_noc_addr(i, s);
        noc_async_write(cb_read_ptr, noc_addr, tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
    }
}
