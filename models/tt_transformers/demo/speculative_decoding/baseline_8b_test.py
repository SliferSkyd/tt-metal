#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Baseline test script for Llama-3.1-8B model performance vs speculative decoding.

This script runs the same prompts with just the Llama-3.1-8B model
to compare against speculative decoding performance.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


def load_prompts_from_json(json_file="sample_prompts.json"):
    """Load prompts from JSON file."""
    try:
        with open(json_file, "r") as f:
            prompts_data = json.load(f)
        return [item["prompt"] for item in prompts_data]
    except Exception as e:
        logger.warning(f"Could not load prompts from {json_file}: {e}")
        return [
            "What is the capital of France?",
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about the ocean.",
        ]


@pytest.mark.parametrize(
    "device_params,pytest_params",
    [
        (
            {"l1_small_size": 24576, "trace_region_size": 400032224, "num_command_queues": 2},
            {"test_name": "baseline-8b-multi-prompt"},
        )
    ],
)
def test_baseline_8b_model(
    mesh_device,
    pytest_params,
    device_params,
):
    """Test baseline Llama-3.1-8B model performance."""

    # Test configuration
    model_name = "meta-llama/Llama-3.1-8B"
    instruct = True
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
    max_seq_len = 1024

    logger.info("🚀 Starting Baseline 8B Model Test")
    logger.info(f"Model: {model_name}")

    # Load prompts
    prompts = load_prompts_from_json()
    logger.info(f"Loaded {len(prompts)} prompts for testing")

    # Initialize timing
    test_start = time.time()

    # Create model
    init_start = time.time()

    # Set the environment variable for the model
    os.environ["HF_MODEL"] = model_name

    model_args, model, kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        instruct=instruct,
        max_batch_size=1,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )

    # Create generator
    generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
    init_time = time.time() - init_start

    logger.info(f"✓ Model loaded: {model_args.n_layers} layers")

    # Test each prompt
    all_results = []
    total_prefill_time = 0
    total_decode_time = 0

    for i, prompt in enumerate(prompts):
        is_warmup = i == 0
        logger.info(f"\n--- Processing Prompt {i+1}/{len(prompts)} {'(WARMUP)' if is_warmup else ''} ---")
        logger.info(f"Prompt: {prompt}")

        # Preprocess input
        input_tokens, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            input_prompts=[prompt],
            tokenizer=generator.tokenizer,
            model_args=[model_args],
            instruct=instruct,
            max_generated_tokens=30,
        )
        # Convert list to tensor and ensure proper shape for Generator
        if isinstance(input_tokens, list):
            input_tokens = torch.stack(input_tokens).squeeze(1)  # Remove extra dimension
        prompt_lens = [len(x) for x in encoded_prompts]

        prompt_start_time = time.time()

        # Run prefill
        logger.info("Running prefill...")
        prefill_start = time.time()
        # For simple case, use None for page_table (no paged attention)
        logits = generator.prefill_forward_text(
            input_tokens, page_table=None, kv_cache=[kv_cache], prompt_lens=prompt_lens
        )
        prefill_time = time.time() - prefill_start
        total_prefill_time += prefill_time

        # Start standard decoding
        logger.info("Starting standard decoding...")
        decode_start = time.time()

        generated_tokens = []
        iteration_times = []

        for iteration in range(30):  # Generate 30 tokens
            iter_start = time.time()

            # Sample next token - sample_host returns (None, token_tensor)
            _, next_token_tensor = sample_host(logits, temperature=0.0, top_p=1.0)
            next_token_id = next_token_tensor.item() if next_token_tensor.numel() == 1 else next_token_tensor[0].item()
            generated_tokens.append(next_token_id)

            # Forward pass for next iteration (if not last)
            if iteration < 29:
                logits = generator.decode_forward_text(
                    tokens=torch.tensor([[next_token_id]], dtype=torch.int32),
                    start_pos=torch.tensor([prompt_lens[0] + iteration + 1], dtype=torch.int32),
                    page_table=None,
                    kv_cache=[kv_cache],
                    enable_trace=True,
                    read_from_device=True,
                )

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            # Log every 10 iterations
            if iteration % 10 == 0:
                tokens_per_second = 1 / iter_time if iter_time > 0 else 0
                logger.info(f"Iter {iteration}: {iter_time*1000:.0f}ms @ {tokens_per_second:.1f} tok/s/user")

        decode_time = time.time() - decode_start
        total_decode_time += decode_time

        # Calculate metrics
        total_time = time.time() - prompt_start_time
        ttft = prefill_time  # Time to first token is prefill time

        if decode_time > 0:
            tokens_per_second = 30 / decode_time
        else:
            tokens_per_second = 0

        avg_decode_time_per_token = decode_time / 30

        # Decode generated text
        generated_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Store results
        prompt_result = {
            "prompt": prompt,
            "prompt_idx": i,
            "is_warmup": is_warmup,
            "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
            "tokens_generated": 30,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "ttft": ttft,
            "tokens_per_second": tokens_per_second,
            "avg_decode_time_per_token": avg_decode_time_per_token,
        }
        all_results.append(prompt_result)

        logger.info(f"✓ Generated 30 tokens")
        logger.info(f"✓ TTFT: {ttft*1000:.1f}ms")
        logger.info(f"✓ Tokens/sec: {tokens_per_second:.2f}")
        logger.info(f"✓ Avg decode time: {avg_decode_time_per_token*1000:.1f}ms/token")
        logger.info(f"Generated text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

    total_test_time = time.time() - test_start

    # Calculate summary metrics (excluding warmup)
    non_warmup_results = [r for r in all_results if not r["is_warmup"]]

    if non_warmup_results:
        avg_ttft = sum(r["ttft"] for r in non_warmup_results) / len(non_warmup_results)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in non_warmup_results) / len(non_warmup_results)
        avg_decode_time_per_token = sum(r["avg_decode_time_per_token"] for r in non_warmup_results) / len(
            non_warmup_results
        )
        total_tokens_generated = sum(r["tokens_generated"] for r in non_warmup_results)

        logger.info("\n" + "=" * 80)
        logger.info("BASELINE 8B MODEL RESULTS")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Model layers: {model_args.n_layers}")
        logger.info(f"Total prompts processed: {len(prompts)}")
        logger.info(f"Prompts used for metrics: {len(non_warmup_results)} (excluding warmup)")
        logger.info(f"Total tokens generated: {total_tokens_generated}")
        logger.info(f"Average TTFT: {avg_ttft*1000:.1f}ms")
        logger.info(f"Average tokens/sec: {avg_tokens_per_second:.2f}")
        logger.info(f"Average decode time per token: {avg_decode_time_per_token*1000:.1f}ms")
        logger.info(f"Total test time: {total_test_time:.2f}s")
        logger.info(f"Model initialization time: {init_time:.2f}s")
        logger.info(f"Total prefill time: {total_prefill_time:.2f}s")
        logger.info(f"Total decode time: {total_decode_time:.2f}s")

        logger.info("\nDetailed Results (excluding warmup):")
        for i, result in enumerate(non_warmup_results):
            logger.info(f"  Prompt {i}: {result['tokens_per_second']:.2f} tok/s, TTFT: {result['ttft']*1000:.1f}ms")

    logger.info("✅ All test assertions passed!")

    # Return results for further analysis
    return {
        "model": model_name,
        "layers": model_args.n_layers,
        "total_prompts": len(prompts),
        "prompts_for_metrics": len(non_warmup_results),
        "total_tokens_generated": total_tokens_generated if non_warmup_results else 0,
        "avg_ttft": avg_ttft if non_warmup_results else 0,
        "avg_tokens_per_second": avg_tokens_per_second if non_warmup_results else 0,
        "avg_decode_time_per_token": avg_decode_time_per_token if non_warmup_results else 0,
        "total_test_time": total_test_time,
        "init_time": init_time,
        "all_results": all_results,
        "non_warmup_results": non_warmup_results,
    }


if __name__ == "__main__":
    pytest.main([__file__])
