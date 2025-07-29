#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in1_receiver_run(
    const uint32_t origin_x_coord,
    const uint32_t origin_y_coord,
    const uint32_t phy_x_coord,
    const uint32_t phy_y_coord,
    const uint32_t start_x,
    const uint32_t start_y,
    const uint32_t end_x,
    const uint32_t end_y,
    const uint32_t mhartid) {
    uint64_t sender_id = (phy_y_coord - origin_y_coord) * (end_x - start_x + 1) + (phy_x_coord - origin_x_coord);
    // Aether::wait_for_tensix_sync(tensix_sync_address, tensix_sync_value);

    // InterleavedAddrGenFast<dram_start_address, tile_size> s;
    //  reset_receive_cmdbuf();

    uint32_t cur_x = phy_x_coord == start_x ? end_x : start_x;
    uint32_t cur_y = phy_y_coord == start_y ? end_y : start_y;

    DeviceTimestampedData("Number of transactions", num_subblocks_k_dim * in1_row_block_num_tiles_per_tensix_cluster);
    DeviceTimestampedData("Transaction size in bytes", tile_size);

    // in1 handler4
    // POST_CODE(sender_id);
    uint64_t l1_write_addr_in1 = 0x80000;
    // POST_CODE(l1_write_addr_in1);
    //  copy start address of block, to be used for mcasting
    uint64_t in1_start_address = l1_write_addr_in1;
    uint64_t dram_addr = dram_start_address;
    // Copy in1 block into CB, as the default kernel
    // POST_CODE(0xa);
    uint32_t in1_tensor_start_tile_id = sender_id * subblock_c_dim * num_subblocks_c_dim / num_of_dests;
    // POST_CODE(in1_tensor_start_tile_id);
    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t subblock_k_dim_index = 0; subblock_k_dim_index < num_subblocks_k_dim; subblock_k_dim_index++) {
            l1_write_addr_in1 = 0x80000;
            // for (uint32_t i = 0; i < num_of_workers_in_neo; i++)
            //{
            //     while (fast_llk_intf_get_free_space(i, cb_id_in1) < in1_row_block_num_tiles_per_tensix_cluster)
            //         ;
            // }

            uint32_t in1_tensor_tile_id = in1_tensor_start_tile_id;
            for (uint32_t tile_counter = 0; tile_counter < in1_row_block_num_tiles_per_tensix_cluster; tile_counter++) {
                noc_async_read(get_noc_addr(cur_x, cur_y, 0x90000), l1_write_addr_in1, tile_size);
                // l1_write_addr_in1 += tile_size;
                in1_tensor_tile_id++;
            }
            in1_tensor_start_tile_id += in1_row_block_num_tiles;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();

            // for (uint32_t i = 0; i < num_of_workers_in_neo; i++)
            //{
            //     fast_llk_intf_inc_posted(i, cb_id_in1, in1_row_block_num_tiles_per_tensix_cluster);
            // }
        }
    }
}
