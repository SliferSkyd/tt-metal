# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import torch

import ttnn
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, comp_pcc
from models.utility_functions import is_blackhole, skip_for_wormhole_b0

# @pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65), (1, 6, 256, 20, 50), (6, 20, 50, 1, 256)])
# @pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1), (1, 3, 4, 0, 2), (3, 0, 4, 1, 2)])
# @pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
# @pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65)])
# @pytest.mark.parametrize("shape", [(3, 33, 3, 3, 33)])
# @pytest.mark.parametrize("shape", [(3, 33, 3, 2, 33)]) # Most efficient so far
# @pytest.mark.parametrize("shape", [(3, 33, 2, 2, 33)])
# @pytest.mark.parametrize("shape", [(1, 33, 1, 1, 31)])
# pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1)])


@pytest.mark.parametrize("shape", [(1, 63, 2, 2, 63)])
@pytest.mark.parametrize("perm", [(1, 3, 4, 0, 2), (3, 0, 4, 1, 2)])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    device.disable_and_clear_program_cache()
    # nop_types_sentence = "UNOPS MNOPS PNOPS"
    nop_types_sentence = "MNOPS"
    nop_types = nop_types_sentence.split()

    num_nop = 1
    num_it = 1  # 10
    min_nop = 0

    # torch.manual_seed(520)
    # input_a = random_torch_tensor(dtype, shape)
    tlist = torch.arange(1 * 63 * 2 * 2 * 63, dtype=torch.bfloat16)
    tlist_a = torch.tensor(tlist)
    input_a = torch.reshape(tlist_a, shape)
    torch_output = torch.permute(input_a, perm)

    # for is_risc in range(2):
    for is_risc in range(1):
        print("RISCV ", is_risc)
        os.environ["RISCV"] = str(is_risc)
        for core_nop in nop_types:
            print("NOP TYPE ", core_nop)
            min_it = num_nop
            for nops in range(0, num_nop):
                os.environ[core_nop] = str(nops)
                counter = 0
                for i in range(num_nop):
                    tt_input = ttnn.from_torch(
                        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
                    )

                    tt_output = ttnn.permute(tt_input, perm)
                    tt_output = ttnn.to_torch(tt_output)

                    assert_equal(torch_output, tt_output)
                    if torch.equal(torch_output, tt_output):
                        counter = counter + 1
                    pcc_val = comp_pcc(torch_output, tt_output)
                    print("PCC : ", pcc_val)
                    torch.set_printoptions(profile="full", linewidth=1000, sci_mode=True)
                    # print(torch_output)
                    # print(tt_output)
                    diff_tensor = torch_output - tt_output
                    sum_val = torch.sum(diff_tensor)
                    print("Sum Diff : ", sum_val)
                print("Nops ", nops, " Counter ", counter)
                if min_it > counter:
                    min_nop = nops
                    min_it = counter
            print("Min nops ", min_nop, " Counter ", min_it)
