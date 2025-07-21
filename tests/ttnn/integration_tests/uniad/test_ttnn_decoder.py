# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from collections import OrderedDict

import ttnn
from models.experimental.uniad.reference.decoder import DetectionTransformerDecoder
from models.experimental.uniad.tt.ttnn_decoder import TtDetectionTransformerDecoder

from models.experimental.uniad.tt.model_preprocessing import extract_sequential_branch


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_decoder(device, reset_seeds):
    weights_path = "models/experimental/uniad/reference/uniad_base_e2e.pth"

    reference_model = DetectionTransformerDecoder(num_layers=6, embed_dim=256, num_heads=8)
    weights = torch.load(weights_path, map_location=torch.device("cpu"))

    prefix = "pts_bbox_head.transformer.decoder"
    filtered = OrderedDict(
        (
            (k[len(prefix) + 1 :], v)  # Remove the prefix from the key
            for k, v in weights["state_dict"].items()
            if k.startswith(prefix)
        )
    )

    reference_model.load_state_dict(filtered)
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
    )

    query = torch.load("models/experimental/uniad/reference/decoder_tensors/query_input.pt")
    kwargs = torch.load("models/experimental/uniad/reference/decoder_tensors/kwargs_updated.pt")
    reference_points = torch.load("models/experimental/uniad/reference/decoder_tensors/reference_points_input.pt")
    reg_branches = torch.load("models/experimental/uniad/reference/decoder_tensors/reg_branches_input.pt")

    parameters_branches = {}
    parameters_branches["reg_branches"] = extract_sequential_branch(reg_branches, dtype=ttnn.bfloat16, device=device)

    parameters_branches = DotDict(parameters_branches)

    ttnn_model = TtDetectionTransformerDecoder(6, 256, 8, parameters, parameters_branches, device)

    output1, output2 = reference_model(
        query=query,
        key=kwargs["key"],
        value=kwargs["value"],
        query_pos=kwargs["query_pos"],
        reference_points=reference_points,
        spatial_shapes=kwargs["spatial_shapes"],
        reg_branches=reg_branches,
    )

    ttnn_output1, ttnn_output2 = ttnn_model(
        query=ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        key=None,
        value=ttnn.from_torch(kwargs["value"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        query_pos=ttnn.from_torch(kwargs["query_pos"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        reference_points=ttnn.from_torch(reference_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        spatial_shapes=ttnn.from_torch(
            kwargs["spatial_shapes"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        reg_branches=reg_branches,
    )

    pcc1, x = assert_with_pcc(output1, ttnn.to_torch(ttnn_output1), 0.99)
    pcc2, y = assert_with_pcc(output2, ttnn.to_torch(ttnn_output2), 0.99)
