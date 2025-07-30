# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Speculative Decoding for Tenstorrent Hardware

This package provides speculative decoding implementation for LLM inference
on Tenstorrent hardware, using a smaller draft model to generate tokens
quickly and a larger target model to verify them.
"""

from .speculative_generator import SpeculativeGenerator

__all__ = ["SpeculativeGenerator"]
