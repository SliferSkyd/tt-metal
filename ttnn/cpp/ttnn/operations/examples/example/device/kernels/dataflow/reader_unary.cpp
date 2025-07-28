// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "../../cb_config.hpp"
#include "../common.hpp"

void kernel_main() {
    constexpr bool output_grad_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool output_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t onetile = 1;

    CommonArgFetcher common_arg_fetcher;
    WORK_PARTITION(tiles_offset, num_tiles);
    const auto output_grad_addr = common_arg_fetcher.next<uint32_t>();
    const auto input_addr = common_arg_fetcher.next<uint32_t>();
    const auto output_addr = common_arg_fetcher.next<uint32_t>();
    const auto requires_input = common_arg_fetcher.next<uint32_t>() == 1;
    const auto requires_output = common_arg_fetcher.next<uint32_t>() == 1;
    const auto do_mask_w = common_arg_fetcher.next<uint32_t>() == 1;
    const auto mask_w = common_arg_fetcher.next<uint32_t>();

    namespace cb = cb_backward;

    if (do_mask_w) {
        generate_mask_w(cb::mask_w, mask_w);
    }

    // Variables
    const uint32_t output_grad_tile_size = get_tile_size(cb::output_grad);
    const uint32_t input_tile_size = get_tile_size(cb::input);
    const uint32_t output_tile_size = get_tile_size(cb::output);
    const DataFormat output_grad_data_format = get_dataformat(cb::output_grad);
    const DataFormat input_data_format = get_dataformat(cb::input);
    const DataFormat output_data_format = get_dataformat(cb::output);

    uint32_t crt_idx = common_arg_fetcher.idx;

    const InterleavedAddrGenFast<output_grad_is_dram> s_output_grad = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_size,
        .data_format = output_grad_data_format};
    const InterleavedAddrGenFast<input_is_dram> s_input = {
        .bank_base_address = input_addr, .page_size = input_tile_size, .data_format = input_data_format};
    const InterleavedAddrGenFast<output_is_dram> s_output = {
        .bank_base_address = output_addr, .page_size = output_tile_size, .data_format = output_data_format};

    for (uint32_t i = tiles_offset; i < tiles_offset + num_tiles; i++) {
        cb_reserve_back(cb::output_grad, onetile);
        const auto cb_output_grad_write_ptr = get_write_ptr(cb::output_grad);

        auto output_grad_noc_addr = get_noc_addr(i, s_output_grad);
        noc_async_read(output_grad_noc_addr, cb_output_grad_write_ptr, output_grad_tile_size);
        noc_async_read_barrier();
        cb_push_back(cb::output_grad, onetile);

        if (requires_input) {
            cb_reserve_back(cb::input, onetile);
            const auto cb_input_write_ptr = get_write_ptr(cb::input);
            noc_async_read_tile(i, s_input, cb_input_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb::input, onetile);
        }

        if (requires_output) {
            cb_reserve_back(cb::output, onetile);
            const auto cb_output_write_ptr = get_write_ptr(cb::output);
            noc_async_read_tile(i, s_output, cb_output_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb::output, onetile);
        }
    }
}
