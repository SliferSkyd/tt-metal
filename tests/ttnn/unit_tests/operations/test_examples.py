# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def get_tensor(input_shape, ttnn_dtype, cpu_dtype, output_dtype, device, *, recip=False, sqrt=False):
    ttnn_shape = input_shape
    if len(ttnn_shape) == 0:
        ttnn_shape = [1]

    if ttnn_dtype == ttnn.bfloat8_b and (recip or sqrt):
        torch_input = 4 * torch.rand(input_shape, dtype=cpu_dtype) + 20
    elif ttnn_dtype != ttnn.int32:
        # uniform distribution [-10, 10)
        torch_input = 20 * torch.rand(input_shape, dtype=cpu_dtype) - 10
        if recip:
            torch_input[(0 <= torch_input) & (torch_input < 1e-10)] += 1e-10
            torch_input[(-1e-10 < torch_input) & (torch_input < 0)] -= 1e-10
    else:
        torch_input = torch.randint(-10, 10, input_shape, dtype=cpu_dtype)

    tt_output = ttnn.empty(ttnn_shape, output_dtype, ttnn.TILE_LAYOUT, device=device)

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan") if ttnn_dtype is not ttnn.bfloat8_b else float("0"),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
    )
    return tt_input, tt_output


@pytest.mark.parametrize("input_shape", [[3, 4, 5, 113, 150]])
@pytest.mark.parametrize("ttnn_dtype, cpu_dtype", [[ttnn.bfloat8_b, torch.float32]])
def test_example(device, input_shape, ttnn_dtype, cpu_dtype, output_dtype=None):
    output_dtype = output_dtype or ttnn_dtype

    tt_input, _ = get_tensor(
        input_shape,
        ttnn_dtype,
        cpu_dtype,
        output_dtype,
        device,
    )

    tt_output_grad, tt_input_grad = get_tensor(
        input_shape,
        ttnn_dtype,
        cpu_dtype,
        output_dtype,
        device,
    )

    ttnn.example(
        tt_output_grad,
        -2,
        input=tt_input,
        input_grad=tt_input_grad,
    )
