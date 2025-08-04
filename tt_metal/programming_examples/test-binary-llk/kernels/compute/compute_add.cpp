#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"  // just for binary_op_init_common

#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    uint32_t nTiles = get_arg_val<uint32_t>(0);

    // Initialize the SFPU
    // init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16); // QUESTION: why don't we need this?

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2);  // QUESTION: why don't we need this???
    add_binary_tile_init();  // QUESTION: seems to work without this?

    for (uint32_t i = 0; i < nTiles; i++) {
        // Wait for data to show up in the circular buffer and copy it from
        // the circular buffer to registers so the SFPU can use it.
        // the first 0 in copy_tile is the index into the circular buffer
        // and the second 0 is the offset into the registers. This case
        // we are copying the 0th tile from the circular buffer to the 0th tile
        // in the registers.

        // MATH: acquire DST
        // DPRINT << __FILE__ << ":" << __LINE__ << " tile_regs_acquire" << ENDL();
        tile_regs_acquire();

        // DPRINT << __FILE__ << ":" << __LINE__ << " cb_wait_front(c_0)" << ENDL();

        cb_wait_front(tt::CBIndex::c_0, 1);
        // DPRINT << __FILE__ << ":" << __LINE__ << " cb_wait_front(c_1)" << ENDL();
        cb_wait_front(tt::CBIndex::c_1, 1);

        // UNPACKER (QUESTION: this is okay after tile_regs_acquire?)
        // CB0[0] -> DST[0]
        // CB1[0] -> DST[1]
        ckernel::copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0 /*DST[0]*/);

        ckernel::copy_tile_to_dst_init_short(tt::CBIndex::c_1);
        copy_tile(tt::CBIndex::c_1, 0, 1 /*DST[1]*/);

        // tt-metal/tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h
        // output overwrites the first operand
        // DST[0] + DST[1] -> DST[0]
        {
            const uint32_t op1 = 0;
            const uint32_t op2 = 1;

            add_binary_tile_init();
            add_binary_tile(op1, op2);
        }

        // MATH: release DST
        tile_regs_commit();

        // PACK: acquire DST
        tile_regs_wait();

        // Wait for space in the circular buffer to be available for us to write
        cb_reserve_back(tt::CBIndex::c_2, 1);

        // DST[0] -> CB16[0]
        {
            const uint32_t dsti = 0;
            const uint32_t cb16i = 0;
            pack_tile(dsti, tt::CBIndex::c_2, cb16i);
        }
        // We don't need the input tile anymore, mark it as consumed
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_pop_front(tt::CBIndex::c_1, 1);

        // Mark the tile as ready for the writer kernel to write to DRAM
        cb_push_back(tt::CBIndex::c_2, 1);

        // PACK: release DST
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
