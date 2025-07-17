# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache, profiler


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_nonzero(
    device,
    reset_seeds,
):
    torch_input = torch.tensor([[[[0, 4, 0, 2, 4, 0, 3]]]])

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device)
    for i in range(2):
        output_indices, output_tensor = ttnn.nonzero(ttnn_input)
        ttnn.deallocate(output_indices)
        ttnn.deallocate(output_tensor)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_index_inplace_add(
    device,
    reset_seeds,
):
    torch_slots = torch.zeros(1, 10000, 256)
    queries = torch.randn(1, 6, 3586, 256).to(dtype=torch.float32)

    indexes = []

    index1 = torch.randint(low=0, high=2968, size=(2968,))
    indexes.append(index1)

    index2 = torch.randint(low=0, high=1263, size=(1263,))
    indexes.append(index2)

    index3 = torch.randint(low=0, high=1295, size=(1295,))
    indexes.append(index3)

    index4 = torch.randint(low=0, high=3586, size=(3586,))
    indexes.append(index4)

    index5 = torch.randint(low=0, high=897, size=(897,))
    indexes.append(index5)

    index6 = torch.randint(low=0, high=941, size=(941,))
    indexes.append(index6)

    for j in range(1):
        for i, index_query_per_img in enumerate(indexes):
            torch_slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_index_add(
    device,
    reset_seeds,
):
    ####### For small dimesion ########
    x = torch.zeros(1, 10, 3)
    y = x.clone()
    t = torch.randn(1, 6, 5, 3).to(dtype=torch.float32)
    index = [[0, 4, 2], [3, 6, 7], [0, 4, 8, 1, 5], [0, 4], [1, 3], [1]]

    for j in range(1):
        for val, ind in enumerate(index):
            x[j, ind] += t[j, val, : len(ind)]
    for j in range(1):
        for val, ind in enumerate(index):
            y[0].index_add_(
                0, torch.tensor(ind, dtype=torch.int32), t[j, val, : len(torch.tensor(ind, dtype=torch.int32))]
            )

    _, msg = assert_with_pcc(x, y, 0.99)
    logger.info(msg)

    ###### Original case ########
    torch_slots = torch.zeros(1, 10000, 256)
    queries = torch.randn(1, 6, 3586, 256).to(dtype=torch.float32)

    indexes = []

    index1 = torch.randint(low=0, high=2968, size=(2968,))
    indexes.append(index1)

    index2 = torch.randint(low=0, high=1263, size=(1263,))
    indexes.append(index2)

    index3 = torch.randint(low=0, high=1295, size=(1295,))
    indexes.append(index3)

    index4 = torch.randint(low=0, high=3586, size=(3586,))
    indexes.append(index4)

    index5 = torch.randint(low=0, high=897, size=(897,))
    indexes.append(index5)

    index6 = torch.randint(low=0, high=941, size=(941,))
    indexes.append(index6)

    t_re_slots = torch_slots.clone()
    ############### Original
    for j in range(1):
        for i, index_query_per_img in enumerate(indexes):
            torch_slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

    ############# TORCH REWRITE
    for j in range(1):
        for i, index_query_per_img in enumerate(indexes):
            t_re_slots[j].index_add_(0, index_query_per_img, queries[j, i, : len(index_query_per_img)])

    _, msg = assert_with_pcc(torch_slots, t_re_slots, 0.99)
    logger.info(msg)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_index_inplace_add_scatter(
    device,
    reset_seeds,
):
    torch_slots = torch.zeros(1, 10000, 256)
    queries = torch.randn(1, 6, 3586, 256).to(dtype=torch.float32)

    indexes = []

    index1 = torch.randint(low=0, high=2968, size=(2968,))
    indexes.append(index1)

    index2 = torch.randint(low=0, high=1263, size=(1263,))
    indexes.append(index2)

    index3 = torch.randint(low=0, high=1295, size=(1295,))
    indexes.append(index3)

    index4 = torch.randint(low=0, high=3586, size=(3586,))
    indexes.append(index4)

    index5 = torch.randint(low=0, high=897, size=(897,))
    indexes.append(index5)

    index6 = torch.randint(low=0, high=941, size=(941,))
    indexes.append(index6)

    torch_slots_original = torch.zeros_like(torch_slots)
    for j in range(1):
        for i, idx in enumerate(indexes):
            torch_slots_original[j, idx] += queries[j, i, : idx.shape[0]]

    torch_slots_scatter = torch.zeros_like(torch_slots)
    for j in range(torch_slots_scatter.size(0)):
        for i, idx in enumerate(indexes):
            idx_exp = idx.unsqueeze(-1).expand(-1, torch_slots_scatter.size(-1))

            # https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/data_movement/gather/gather_pybind.cpp
            gathered = torch.gather(torch_slots_scatter[j], 0, idx_exp)

            q = queries[j, i, : idx.shape[0]]
            updated = gathered + q

            # https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/experimental/scatter/scatter_pybind.cpp
            torch_slots_scatter[j].scatter_(0, idx_exp, updated)

    _, msg = assert_with_pcc(torch_slots_original, torch_slots_scatter, 0.99)
    logger.info(msg)
