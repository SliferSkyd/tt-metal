# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params[0][0].weight  # Changed here
        self.linear2_weight = params[1].weight  # Changed here
        self.linear1_bias = params[0][0].bias  # Changed here
        self.linear2_bias = params[1].bias  # Changed here

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # First linear + ReLU
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)

        return x
