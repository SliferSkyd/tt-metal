#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Baseline comparison script for 3B model performance vs speculative decoding.

This script runs the same prompts with just the Llama-3.2-3B model
to compare against speculative decoding performance.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


@pytest.mark.parametrize(
    "target_model_name, max_seq_len, max_generated_tokens",
    [
        (  # Baseline 3B model test
            "meta-llama/Llama-3.2-3B",  # target_model_name
            1024,  # max_seq_len
            30,  # max_generated_tokens (same as speculative test)
        ),
    ],
    ids=[
        "baseline-3b",
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, 1)  # Default to single device if MESH_DEVICE not set
        )
    ],
    indirect=True,
)
def test_baseline_3b_model(
    target_model_name,
    max_seq_len,
    max_generated_tokens,
    optimizations,
    mesh_device,
    reset_seeds,
):
    """
    Test baseline 3B model performance for comparison with speculative decoding.
    """

    instruct = True
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about the ocean.",
    ]

    logger.info("=" * 80)
    logger.info("BASELINE 3B MODEL PERFORMANCE TEST")
    logger.info("=" * 80)
    logger.info(f"Model: {target_model_name}")
    logger.info(f"Max generated tokens: {max_generated_tokens}")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info("=" * 80)

    # Start profiler
    profiler = BenchmarkProfiler()
    profiler.start("total_test")

    # Initialize model
    logger.info("Initializing baseline model...")
    profiler.start("model_initialization")

    # Temporarily set HF_MODEL for target model
    original_hf_model = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = target_model_name

    try:
        model_args, model, kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=instruct,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat8_b,
        )

        generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
        tokenizer = model_args.tokenizer

    finally:
        # Restore original HF_MODEL environment variable
        if original_hf_model:
            os.environ["HF_MODEL"] = original_hf_model
        elif "HF_MODEL" in os.environ:
            del os.environ["HF_MODEL"]

    profiler.end("model_initialization")

    # Display model information
    layers = model_args.n_layers
    logger.info(f"✓ Model loaded: {layers} layers")

    # Process each prompt
    all_results = []
    total_prefill_time = 0
    total_decode_time = 0
    total_tokens_generated = 0

    for prompt_idx, prompt in enumerate(test_prompts):
        logger.info(f"\n--- Processing Prompt {prompt_idx + 1}/{len(test_prompts)} ---")
        logger.info(f"Prompt: {prompt}")

        # Preprocess prompt
        profiler.start(f"preprocess_prompt_{prompt_idx}")
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            [prompt], tokenizer, [model_args], instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        profiler.end(f"preprocess_prompt_{prompt_idx}")

        # Run prefill
        logger.info("Running prefill...")
        profiler.start(f"prefill_{prompt_idx}")

        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=None,
            kv_cache=[kv_cache],
            prompt_lens=decoding_pos,
        )

        target_token = torch.argmax(logits, dim=-1)
        profiler.end(f"prefill_{prompt_idx}")

        prefill_time = profiler.get_duration(f"prefill_{prompt_idx}")
        total_prefill_time += prefill_time

        # Initialize for decoding
        all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
        all_outputs[0].append(int(target_token[0].item()))

        current_pos = torch.tensor([decoding_pos[0]])
        out_tok = target_token

        # Standard decoding loop
        iteration = 0
        tokens_generated = 0

        logger.info("Starting standard decoding...")
        profiler.start(f"decode_{prompt_idx}")

        decode_start_time = time.time()
        iteration_times = []

        while tokens_generated < max_generated_tokens:
            iter_start_time = time.time()

            # Standard decode forward
            logits = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=True,
                kv_cache=[kv_cache],
            )

            # Sample next token
            _, next_token = sample_host(logits, temperature=0.0, top_p=1.0, on_host=True)

            iter_end_time = time.time()
            iteration_time = iter_end_time - iter_start_time
            iteration_times.append(iteration_time)

            # Update position and token
            current_pos += 1
            out_tok = next_token
            tokens_generated += 1

            # Save output token
            user_tok = next_token.item()
            if user_tok not in tokenizer.stop_tokens:
                all_outputs[0].append(user_tok)
            else:
                logger.info(f"Hit stop token at iteration {iteration}")
                break

            # Log progress every few iterations
            if iteration % 5 == 0:
                tokens_per_second = 1.0 / iteration_time
                logger.info(f"Iter {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second:.1f} tok/s/user")

            iteration += 1

            # Check for stop tokens
            if user_tok in tokenizer.stop_tokens:
                break

        profiler.end(f"decode_{prompt_idx}")

        decode_time = profiler.get_duration(f"decode_{prompt_idx}")
        total_decode_time += decode_time
        total_tokens_generated += tokens_generated

        # Generate final output
        final_text = tokenizer.decode(all_outputs[0])
        prompt_with_tags = tokenizer.decode(model_args.encode_prompt(prompt, instruct=instruct))
        generated_text = final_text.replace(prompt_with_tags, "", 1).strip()

        # Calculate metrics for this prompt
        ttft = prefill_time  # Time to first token
        avg_decode_time = decode_time / tokens_generated if tokens_generated > 0 else 0
        tokens_per_second = tokens_generated / decode_time if decode_time > 0 else 0

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "ttft": ttft,
            "tokens_per_second": tokens_per_second,
            "avg_decode_time_per_token": avg_decode_time,
        }
        all_results.append(result)

        logger.info(f"✓ Generated {tokens_generated} tokens")
        logger.info(f"✓ TTFT: {ttft*1000:.1f}ms")
        logger.info(f"✓ Tokens/sec: {tokens_per_second:.2f}")
        logger.info(f"✓ Avg decode time: {avg_decode_time*1000:.1f}ms/token")
        logger.info(f"Generated text: {generated_text[:100]}...")

    profiler.end("total_test")

    # Calculate overall metrics
    total_test_time = profiler.get_duration("total_test")
    init_time = profiler.get_duration("model_initialization")

    avg_ttft = total_prefill_time / len(test_prompts)
    avg_tokens_per_second = total_tokens_generated / total_decode_time if total_decode_time > 0 else 0
    avg_decode_time_per_token = total_decode_time / total_tokens_generated if total_tokens_generated > 0 else 0

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE MODEL SUMMARY RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {target_model_name}")
    logger.info(f"Model layers: {layers}")
    logger.info(f"Prompts processed: {len(test_prompts)}")
    logger.info(f"Total tokens generated: {total_tokens_generated}")
    logger.info(f"Average TTFT: {avg_ttft*1000:.1f}ms")
    logger.info(f"Average tokens/sec: {avg_tokens_per_second:.2f}")
    logger.info(f"Average decode time per token: {avg_decode_time_per_token*1000:.1f}ms")
    logger.info(f"Total test time: {total_test_time:.2f}s")
    logger.info(f"Model initialization time: {init_time:.2f}s")
    logger.info(f"Total prefill time: {total_prefill_time:.2f}s")
    logger.info(f"Total decode time: {total_decode_time:.2f}s")

    # Detailed results
    logger.info(f"\nDetailed Results:")
    for i, result in enumerate(all_results):
        logger.info(f"  Prompt {i+1}: {result['tokens_per_second']:.2f} tok/s, TTFT: {result['ttft']*1000:.1f}ms")

    # Test assertions
    assert total_tokens_generated > 0, "No tokens were generated"
    assert avg_tokens_per_second > 0, "Invalid token generation rate"
    assert all(len(r["generated_text"]) > 0 for r in all_results), "Some prompts generated no text"

    logger.info("✅ All test assertions passed!")

    return {
        "model": target_model_name,
        "layers": layers,
        "total_tokens_generated": total_tokens_generated,
        "avg_ttft": avg_ttft,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_decode_time_per_token": avg_decode_time_per_token,
        "total_test_time": total_test_time,
        "init_time": init_time,
        "results_per_prompt": all_results,
    }


if __name__ == "__main__":
    # Run with pytest
    logger.info("Use pytest to run this test:")
    logger.info("pytest baseline_comparison.py::test_baseline_3b_model -v")
