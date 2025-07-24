# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
)


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def scaled_dot_product_attention_reference(Q, K, V, scale, is_causal=True):
    """
    Full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)
    """

    b, nh, S, d_qk = Q.shape
    _, nkv, _, d_v = V.shape
    # Expand KV to match Q heads
    head_rep = nh // nkv
    K_exp = K.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_qk)
    V_exp = V.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_v)
    # Use PyTorch’s builtin causal attention
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, attn_mask=None, scale=scale, is_causal=is_causal
    )


def page_table_setup(batch_size: int, config: PagedAttentionConfig) -> torch.Tensor:
    """
    Setup the page-related tensors for the attention cache.
    Args:
        batch_size: The number of batches.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        page_table: The page table tensor.
    """
    max_num_blocks = config.max_num_blocks
    assert (
        max_num_blocks % batch_size == 0
    ), f"max_num_blocks {max_num_blocks} must be divisible by batch_size {batch_size}."

    page_table = torch.randperm(max_num_blocks, dtype=torch.int32)
    page_table = page_table.reshape(batch_size, max_num_blocks // batch_size)

    return page_table


def to_paged_cache(
    cache: torch.Tensor,
    mapping: torch.Tensor,
    config: PagedAttentionConfig,
) -> torch.Tensor:
    """
    Convert a cache tensor to a paged cache using the provided mapping.
    Args:
        cache: The original cache tensor.
        mapping: The mapping tensor that defines how to convert the cache.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        paged_cache: The converted paged cache tensor.
    """
    batch_size, nh, seq_len, dim = cache.shape

    block_size, max_num_blocks = config.block_size, config.max_num_blocks
    assert (
        max_num_blocks % batch_size == 0
    ), f"max_num_blocks {max_num_blocks} must be divisible by batch_size {batch_size}."
    assert seq_len == block_size * (
        max_num_blocks // batch_size
    ), f"Sequence length {seq_len} must equal effective paged seq_len {block_size * (max_num_blocks // batch_size)}."

    paged_cache = cache.reshape(batch_size, nh, -1, block_size, dim)  # (B, H, num_blocks // B, block_size, D)
    paged_cache = paged_cache.transpose(1, 2)  # (B, num_blocks // B, H, block_size, D)
    paged_cache = paged_cache.reshape(max_num_blocks, nh, block_size, dim)  # (num_blocks, H, block_size, D)

    # Get the reverse mapping to reorder the paged cache, so that paged cache + mapping = original cache
    # So, paged_cache = original_cache + inverse mapping
    inverse_mapping = torch.argsort(mapping.view(-1))
    paged_cache = paged_cache[inverse_mapping]

    return paged_cache


def from_paged_cache(
    paged_cache: torch.Tensor,
    mapping: torch.Tensor,
    config: PagedAttentionConfig,
) -> torch.Tensor:
    """
    Convert a paged cache back to the original cache format using the provided mapping.
    Args:
        paged_cache: The paged cache tensor.
        mapping: The mapping tensor that defines how to convert the paged cache.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        cache: The converted cache tensor.
    """
    max_num_blocks, nh, block_size, dim = paged_cache.shape  # (max_num_blocks, H, block_size, D)
    assert (
        block_size == config.block_size
    ), f"block_size {block_size} must match the paged attention config block size {config.block_size}."
    assert (
        max_num_blocks == config.max_num_blocks
    ), f"max_num_blocks {max_num_blocks} must match the paged attention config max_num_blocks {config.max_num_blocks}."

    batch, num_blocks_per_batch = mapping.shape

    # Use the mapping to get the original order, paged_cache + mapping = original cache
    cache = paged_cache[mapping.view(-1)]

    cache = cache.reshape(batch, num_blocks_per_batch, nh, block_size, dim)  # (B, num_blocks // B, H, block_size, D)
    cache = cache.transpose(1, 2)  # (B, H, num_blocks // B, block_size, D)
    cache = cache.reshape(batch, nh, -1, dim)  # (B, H, seq_len, D)

    return cache


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention=None,
):
    # Log the test parameters
    logger.info(f"Running FlashMLA Prefill with parameters: ")
    logger.info(f"Batch: {batch}")
    logger.info(f"Sequence Length: {seq_len}")
    logger.info(f"Number of Heads (Q): {nh}")
    logger.info(f"Number of Heads (KV): {nkv}")
    logger.info(f"KV LoRA Rank: {kv_lora_rank}")
    logger.info(f"Dimensionality of RoPE: {d_rope}")
    logger.info(f"Query Data Type: {q_dtype}")
    logger.info(f"Key-Value Data Type: {dtype}")

    # Paged attention configuration
    paged_attention_cfg = None
    if use_paged_attention:
        block_size = ttnn.TILE_SIZE
        assert seq_len % block_size == 0, f"Sequence length must be a multiple of {block_size=} for paged attention."

        max_num_blocks = seq_len // block_size * batch
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)
    ######################
    ### TT Setup
    #######################

    # Page-related setup
    tt_k_torch = k
    page_table = None
    if paged_attention_cfg:
        page_table = page_table_setup(batch, paged_attention_cfg)
        tt_k_torch = to_paged_cache(
            k,
            page_table,
            paged_attention_cfg,
        )
        tt_k_torch_og = from_paged_cache(
            tt_k_torch,
            page_table,
            paged_attention_cfg,
        )
        assert torch.all(tt_k_torch_og == k), "Paged cache conversion for K failed."

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(
        q,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        tt_k_torch,  # (B, H, S, D)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ##########################
    ### FlashMLA Prefill
    ##########################
    tt_out = ttnn.transformer.flash_mla_prefill(
        tt_q,
        tt_k,
        head_dim_v=kv_lora_rank,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=True,
    )
    tt_back = ttnn.to_torch(tt_out)  # now (B, H_padded, S_padded, D)
    print("raw to_torch shape:", tt_back.shape)
    # slice out the padded heads and sequence length; no permute needed
    tt_out_torch = tt_back[:, :nh, :seq_len, :]  # (B, nh, S, D)

    ########################
    ### Validation
    ########################
    out_t = scaled_dot_product_attention_reference(
        q,
        k,
        v,
        scale,
        is_causal=True,
    )

    pcc_threshold = 0.99
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.98

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),
        (2, 8 * 1024, 8, 1, 128, 64),
        (2, 4 * 1024, 64, 1, 256, 0),
        (2, 4 * 1024, 64, 1, 32, 64),
        (8, 4 * 1024, 8, 1, 128, 32),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize(
    "use_paged_attention",
    [
        False,
        # True,
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention,
    function_level_defaults,
    reset_seeds,
):
    # Paged attention configuration
    paged_attention_cfg = None
    if use_paged_attention:
        block_size = ttnn.TILE_SIZE
        assert seq_len % block_size == 0, f"Sequence length must be a multiple of {block_size=} for paged attention."

        max_num_blocks = seq_len // block_size * batch
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        paged_attention_cfg=paged_attention_cfg,
    )


@pytest.mark.parametrize(
    "batch",
    [
        1,  # Single batch
        2,  # Multiple batches
        8,  # Even larger batch size
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [
        1 * 1024,  # Long sequence length
    ],
)
@pytest.mark.parametrize(
    "nh",
    [
        16,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "nkv",
    [
        1,
        8,
        16,
    ],
)
@pytest.mark.parametrize(
    "kv_lora_rank",
    [
        64,
        512,
    ],
)
@pytest.mark.parametrize(
    "d_rope",
    [
        0,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill_stress(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
    )
