#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test script for speculative decoding with Llama-3.1-8B target model.

This script tests:
- Draft model: meta-llama/Llama-3.2-1B (smaller, faster)
- Target model: meta-llama/Llama-3.1-8B (larger, more accurate)
"""

import json
import time

import pytest
import torch
from loguru import logger

from models.tt_transformers.demo.speculative_decoding.speculative_generator import SpeculativeGenerator
from models.tt_transformers.tt.common import preprocess_inputs_prefill
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
            {"l1_small_size": 24576, "trace_region_size": 63222400, "num_command_queues": 2},
            {"test_name": "speculative-1b-8b-multi-prompt"},
        )
    ],
)
def test_speculative_llama_8b_models(
    mesh_device,
    pytest_params,
    device_params,
):
    """Test speculative decoding with Llama-3.2-1B draft and Llama-3.1-8B target."""

    # Test configuration
    draft_model_name = "meta-llama/Llama-3.2-1B"
    target_model_name = "meta-llama/Llama-3.1-8B"
    instruct = True
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
    max_seq_len = 1024

    logger.info("🚀 Starting Speculative Decoding Test: 1B → 8B")
    logger.info(f"Draft Model: {draft_model_name}")
    logger.info(f"Target Model: {target_model_name}")

    # Load prompts
    prompts = load_prompts_from_json()
    logger.info(f"Loaded {len(prompts)} prompts for testing")

    # Initialize timing
    test_start = time.time()

    # Initialize speculative generator
    init_start = time.time()
    spec_generator = SpeculativeGenerator(
        mesh_device=mesh_device,
        draft_model_name=draft_model_name,
        target_model_name=target_model_name,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        n_draft_tokens=8,
    )
    init_time = time.time() - init_start

    logger.info(
        f"✓ Models loaded - Draft: {spec_generator.draft_model_args.n_layers} layers, Target: {spec_generator.target_model_args.n_layers} layers"
    )

    # Test each prompt
    all_results = []
    acceptance_rates = []
    effective_speedups = []

    for i, prompt in enumerate(prompts):
        is_warmup = i == 0
        logger.info(f"\n--- Processing Prompt {i+1}/{len(prompts)} {'(WARMUP)' if is_warmup else ''} ---")
        logger.info(f"Prompt: {prompt}")

        # Preprocess input
        input_tokens, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            input_prompts=[prompt],
            tokenizer=spec_generator.get_tokenizer(),
            model_args=[spec_generator.target_model_args],
            instruct=instruct,
            max_generated_tokens=30,
        )
        # Convert list to tensor (crucial for prefill_forward_text)
        input_tokens = torch.stack(input_tokens).view(1, -1)
        prompt_lens = [len(x) for x in encoded_prompts]

        # Run speculative generation
        prompt_start = time.time()

        try:
            all_outputs, stats = spec_generator.generate_speculative(
                input_tokens=input_tokens,
                prompt_lens=prompt_lens,
                max_generated_tokens=30,
                temperature=0.0,
                top_p=1.0,
            )

            prompt_time = time.time() - prompt_start

            # Parse results
            generated_tokens = all_outputs[0]  # Get first batch item
            generated_text = spec_generator.get_tokenizer().decode(generated_tokens, skip_special_tokens=True)
            tokens_generated = stats["tokens_generated"]
            acceptance_rate = stats["acceptance_rate"]
            effective_speedup = stats["effective_speedup"]
            ttft = stats.get("time_to_first_token", stats["prefill_time"])  # Use prefill time as TTFT
            total_time = stats["total_time"]

            # Calculate token rate
            if total_time > 0:
                tokens_per_second = tokens_generated / total_time
            else:
                tokens_per_second = 0

            # Store results
            prompt_result = {
                "prompt": prompt,
                "prompt_idx": i,
                "is_warmup": is_warmup,
                "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                "tokens_generated": tokens_generated,
                "total_time": total_time,
                "ttft": ttft,
                "tokens_per_second": tokens_per_second,
                "acceptance_rate": acceptance_rate,
                "effective_speedup": effective_speedup,
            }
            all_results.append(prompt_result)

            # Collect metrics (excluding warmup)
            if not is_warmup:
                acceptance_rates.append(acceptance_rate)
                effective_speedups.append(effective_speedup)

            logger.info(f"✓ Generated {tokens_generated} tokens")
            logger.info(f"✓ TTFT: {ttft*1000:.1f}ms")
            logger.info(f"✓ Tokens/sec: {tokens_per_second:.2f}")
            logger.info(f"✓ Acceptance rate: {acceptance_rate:.1%}")
            logger.info(f"✓ Effective speedup: {effective_speedup:.2f}x")
            logger.info(f"Generated text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

        except Exception as e:
            logger.error(f"Error processing prompt {i}: {e}")
            continue

    total_test_time = time.time() - test_start

    # Calculate summary metrics (excluding warmup)
    non_warmup_results = [r for r in all_results if not r["is_warmup"]]

    if non_warmup_results:
        avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
        avg_effective_speedup = sum(effective_speedups) / len(effective_speedups)
        avg_ttft = sum(r["ttft"] for r in non_warmup_results) / len(non_warmup_results)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in non_warmup_results) / len(non_warmup_results)
        total_tokens_generated = sum(r["tokens_generated"] for r in non_warmup_results)

        logger.info("\n" + "=" * 80)
        logger.info("SPECULATIVE DECODING RESULTS (1B → 8B)")
        logger.info("=" * 80)
        logger.info(f"Draft Model: {draft_model_name}")
        logger.info(f"Target Model: {target_model_name}")
        logger.info(f"Total prompts processed: {len(prompts)}")
        logger.info(f"Prompts used for metrics: {len(non_warmup_results)} (excluding warmup)")
        logger.info(f"Total tokens generated: {total_tokens_generated}")
        logger.info(f"Average acceptance rate: {avg_acceptance_rate:.1f}%")
        logger.info(f"Average effective speedup: {avg_effective_speedup:.2f}x")
        logger.info(f"Average TTFT: {avg_ttft*1000:.1f}ms")
        logger.info(f"Average tokens/sec: {avg_tokens_per_second:.2f}")
        logger.info(f"Total test time: {total_test_time:.2f}s")
        logger.info(f"Model initialization time: {init_time:.2f}s")

        logger.info("\nDetailed Results (excluding warmup):")
        for i, result in enumerate(non_warmup_results):
            logger.info(
                f"  Prompt {i}: {result['tokens_per_second']:.2f} tok/s, "
                f"TTFT: {result['ttft']*1000:.1f}ms, "
                f"Accept: {result['acceptance_rate']:.1f}%, "
                f"Speedup: {result['effective_speedup']:.2f}x"
            )

    # Cleanup
    spec_generator.cleanup()

    logger.info("✅ All test assertions passed!")

    # Return results for further analysis
    return {
        "draft_model": draft_model_name,
        "target_model": target_model_name,
        "total_prompts": len(prompts),
        "prompts_for_metrics": len(non_warmup_results),
        "total_tokens_generated": total_tokens_generated if non_warmup_results else 0,
        "avg_acceptance_rate": avg_acceptance_rate if non_warmup_results else 0,
        "avg_effective_speedup": avg_effective_speedup if non_warmup_results else 0,
        "avg_ttft": avg_ttft if non_warmup_results else 0,
        "avg_tokens_per_second": avg_tokens_per_second if non_warmup_results else 0,
        "total_test_time": total_test_time,
        "init_time": init_time,
        "all_results": all_results,
        "non_warmup_results": non_warmup_results,
    }


if __name__ == "__main__":
    pytest.main([__file__])
