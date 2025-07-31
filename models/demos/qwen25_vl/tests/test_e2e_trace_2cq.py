# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.qwen25_vl.runner.performant_runner import Qwen25VLPerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("act_dtype, weight_dtype", ((ttnn.bfloat16, ttnn.bfloat8_b),))
@pytest.mark.parametrize("batch_size, seq_len", [(1, 14308)])
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_trace_2cq_qwen25vl(device, batch_size, seq_len, act_dtype, weight_dtype):
    # NOTE: This test is currently skipped due to a fundamental limitation in TTNN's
    # windowed_scaled_dot_product_attention operation. The operation's validate() method
    # calls cu_window_seqlens.cpu().to_vector<>() which performs device-to-host tensor reads
    # that are forbidden during trace capture mode. This is not a bug in our implementation
    # but a limitation at the TTNN operation level that needs to be addressed by the TTNN team.
    # pytest.skip("Qwen25VL trace capture incompatible with windowed attention validation")

    runner = Qwen25VLPerformantRunner(
        device=device,
        device_batch_size=batch_size,
        seq_len=seq_len,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
    )
    runner._capture_qwen25vl_trace_2cqs()
    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = runner.run()
        t1 = time.time()
        inference_times.append(t1 - t0)
    runner.release()
    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_qwen25vl_batch_size: {batch_size}, One inference iteration time (sec): {inference_time_avg}, Images per sec: {round(batch_size/inference_time_avg)}"
    )
