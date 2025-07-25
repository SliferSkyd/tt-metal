# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
import pytest

from models.experimental.uniad.tt.ttnn_transformer_encoder_layer import TtTransformerEncoderLayer

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.TransformerEncoderLayer):
        parameters["self_attn"] = model.self_attn

        parameters["linear1"] = {}
        parameters["linear1"]["weight"] = preprocess_linear_weight(model.linear1.weight, dtype=ttnn.bfloat16)
        parameters["linear1"]["bias"] = preprocess_linear_bias(model.linear1.bias, dtype=ttnn.bfloat16)

        parameters["linear2"] = {}
        parameters["linear2"]["weight"] = preprocess_linear_weight(model.linear2.weight, dtype=ttnn.bfloat16)
        parameters["linear2"]["bias"] = preprocess_linear_bias(model.linear2.bias, dtype=ttnn.bfloat16)

        parameters["norm1"] = {}
        parameters["norm1"]["weight"] = preprocess_layernorm_parameter(model.norm1.weight, dtype=ttnn.bfloat16)
        parameters["norm1"]["bias"] = preprocess_layernorm_parameter(model.norm1.bias, dtype=ttnn.bfloat16)

        parameters["norm2"] = {}
        parameters["norm2"]["weight"] = preprocess_layernorm_parameter(model.norm2.weight, dtype=ttnn.bfloat16)
        parameters["norm2"]["bias"] = preprocess_layernorm_parameter(model.norm2.bias, dtype=ttnn.bfloat16)

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_transformer_encoder_layer(device, reset_seeds):
    torch_model = nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dropout=0.1,
        dim_feedforward=512,
        batch_first=True,
    )

    torch_model.eval()

    query = torch.randn(1, 6, 256)

    torch_output = torch_model(query)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_model = TtTransformerEncoderLayer(
        parameters=parameters,
        device=device,
        d_model=256,
        nhead=8,
        dropout=0.1,
        dim_feedforward=512,
        batch_first=True,
    )

    ttnn_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(ttnn_query)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=1)
