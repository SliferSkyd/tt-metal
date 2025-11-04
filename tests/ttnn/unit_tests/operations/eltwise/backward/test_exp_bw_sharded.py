# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.sweep_framework.sweeps.eltwise.unary_backward.exp_bw import exp_bw_sharded


def test_exp_bw_sharded_main_vector(device):
    test_vector = {
        "input_spec": {
            "input_shape": [2, 160, 64],
            "X": 8,
            "Y": 6,
            "sharding_strategy": "BLOCK",
            "shard_orientation": "COL_MAJOR",
            "tensor_hw_as_shard_shape": True,
            "input_layout": "TILE_LAYOUT",
            "shard_height_mul_of_32": False,
        },
        "grad_dtype": ttnn.bfloat16,
        "input_a_dtype": ttnn.bfloat16,
    }

    invalidated, reason = exp_bw_sharded.invalidate_vector(test_vector)
    assert not invalidated, reason

    (pcc_passed, pcc_message), e2e_perf = exp_bw_sharded.run(
        test_vector["input_spec"],
        test_vector["grad_dtype"],
        test_vector["input_a_dtype"],
        device=device,
    )

    assert pcc_passed, pcc_message
    assert isinstance(e2e_perf, int) and e2e_perf >= 0
