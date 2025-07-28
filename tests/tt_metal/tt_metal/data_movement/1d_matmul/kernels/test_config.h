#pragma once

#include <stdint.h>
#include <array>

namespace TestConfig {
constexpr uint32_t dram_start_address = 0x0;
constexpr uint32_t dram_gather_address = 0x100000;

// Tile parameters
constexpr uint32_t data_format_bytes = 2;
constexpr uint32_t tile_size = 32 * 32 * data_format_bytes;  // Bytes
constexpr uint32_t face_size = 16 * 16 * data_format_bytes;  // Bytes
constexpr uint32_t faces_per_tile = tile_size / face_size;

// Math parameters
constexpr uint32_t tensix_sync_address = 0x80000 - 0x40;
constexpr uint32_t tensix_sync_value = 1;

// Worker parameters
constexpr uint32_t num_of_dests_x = 4;
constexpr uint32_t num_of_dests_y = 4;
constexpr uint32_t num_of_dests = num_of_dests_x * num_of_dests_y;
constexpr uint32_t num_of_workers = num_of_dests;
constexpr uint32_t num_of_workers_in_neo = 4;
constexpr uint32_t num_remote_senders = num_of_dests;
constexpr uint32_t in0_mcast_num_dests = num_of_dests;
constexpr uint32_t in0_mcast_dest_noc_start_x = 0;  // logical x coord
constexpr uint32_t in0_mcast_dest_noc_start_y = 0;  // logical y coord
constexpr uint32_t in0_mcast_dest_noc_end_x = 3;    // logical x coord
constexpr uint32_t in0_mcast_dest_noc_end_y = 0;    // logical y coord

// In0 parameters
constexpr uint32_t subblock_r_dim = 1;
constexpr uint32_t subblock_c_dim = 32;
constexpr uint32_t subblock_k_dim = 32;
constexpr uint32_t num_subblocks_r_dim = 4;
constexpr uint32_t num_subblocks_c_dim = 32;
constexpr uint32_t num_subblocks_k_dim = 32;

constexpr uint32_t num_subblocks_k_dim_per_tensix_cluster = num_subblocks_k_dim / num_of_dests;
constexpr uint32_t num_subblocks_c_dim_per_tensix_cluster = num_subblocks_c_dim / num_of_dests;
static_assert(num_subblocks_k_dim_per_tensix_cluster != 0, "num_subblocks_k_dim not divisible by number of workers");
static_assert(num_subblocks_c_dim_per_tensix_cluster != 0, "num_subblocks_c_dim not divisible by number of workers");

constexpr uint32_t in0_column_block_num_tiles = subblock_r_dim * num_subblocks_r_dim * subblock_k_dim;
constexpr uint32_t in0_column_block_size_bytes = in0_column_block_num_tiles * tile_size;
constexpr uint32_t in0_column_block_num_tiles_per_tensix_core = in0_column_block_num_tiles / num_of_workers_in_neo;
constexpr uint32_t in0_column_block_size_bytes_per_tensix_core = in0_column_block_num_tiles_per_tensix_core * tile_size;

constexpr uint32_t output_num_tiles_per_core = subblock_r_dim * subblock_c_dim;
constexpr uint32_t output_num_tiles_per_row_per_cluster =
    output_num_tiles_per_core * num_subblocks_c_dim / num_of_dests_x;
constexpr uint32_t output_num_tiles_per_cluster = output_num_tiles_per_row_per_cluster * num_subblocks_r_dim;

// In1 parameters
// Width slice of work for each neo
constexpr uint32_t in1_row_block_num_tiles = subblock_k_dim * subblock_c_dim * num_subblocks_c_dim;
constexpr uint32_t in1_row_block_size_bytes = in1_row_block_num_tiles * tile_size;
constexpr uint32_t in1_row_block_num_tiles_per_tensix_cluster = in1_row_block_num_tiles / num_of_dests;
constexpr uint32_t in1_row_block_size_bytes_per_tensix_cluster = in1_row_block_num_tiles_per_tensix_cluster * tile_size;

static_assert(
    in1_row_block_num_tiles_per_tensix_cluster != 0, "in1_row_block_num_tiles not divisible by number of workers");

// Addr of semaphore, which all the receivers will notify when they are ready to receive multicast
constexpr uint32_t in0_mcast_sender_semaphore_addr = 0x80000 - 1 * 0x100;
constexpr uint32_t in0_mcast_sender_semaphore_valid_addr = 0x80000 - 2 * 0x100;
// Addr of all the remote semaphore for the current block
constexpr uint32_t in0_mcast_receiver_semaphore_addr = 0x80000 - 3 * 0x100;

// Circular buffer fake
constexpr uint8_t cb_id_in0 = 0;
constexpr uint8_t cb_id_in1 = 1;
constexpr uint8_t cb_id_in2 = 2;
constexpr uint8_t cb_id_in3 = 3;
constexpr uint8_t cb_id_in4 = 4;
constexpr uint8_t cb_id_in5 = 5;
constexpr uint8_t cb_id_out8 = 8;
constexpr uint32_t cb0_write_addr = 0x80000;
constexpr uint32_t cb1_write_addr = 0x80000;
constexpr uint32_t cb2_write_addr = 0x80000;
constexpr uint32_t cb3_write_addr = 0x80000;
constexpr uint32_t cb4_write_addr = 0x80000;
constexpr uint32_t cb5_write_addr = 0x80000;
constexpr uint32_t cb8_write_addr = 0x80000;

constexpr std::array<uint32_t, 7> cb_write_addr_map = {
    cb0_write_addr,
    cb1_write_addr,
    cb2_write_addr,
    cb3_write_addr,
    cb4_write_addr,
    cb5_write_addr,
    cb8_write_addr,
};
}  // namespace TestConfig
