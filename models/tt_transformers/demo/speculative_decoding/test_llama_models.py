#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test script for speculative decoding with different Llama model sizes.

This script tests:
- Draft model: meta-llama/Llama-3.2-1B (smaller, faster)
- Target model: meta-llama/Llama-3.2-3B (larger, more accurate)
"""

import os

import pytest
import torch
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.speculative_decoding.speculative_generator import SpeculativeGenerator
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import DecodersPrecision


@pytest.mark.parametrize(
    "draft_model_name, target_model_name, max_seq_len, max_generated_tokens, n_draft_tokens",
    [
        (  # Llama-3.2-1B → Llama-3.2-3B test
            "meta-llama/Llama-3.2-1B",  # draft_model_name
            "meta-llama/Llama-3.2-3B",  # target_model_name
            1024,  # max_seq_len
            30,  # max_generated_tokens
            4,  # n_draft_tokens
        ),
    ],
    ids=[
        "llama-1b-3b",
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
def test_speculative_llama_models(
    draft_model_name,
    target_model_name,
    max_seq_len,
    max_generated_tokens,
    n_draft_tokens,
    optimizations,
    mesh_device,
    reset_seeds,
):
    """
    Test speculative decoding with Llama-3.2-1B as draft and Llama-3.2-3B as target.
    """

    instruct = True
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about the ocean.",
    ]

    logger.info("=" * 80)
    logger.info("SPECULATIVE DECODING TEST WITH PYTEST")
    logger.info("=" * 80)
    logger.info(f"Draft model: {draft_model_name}")
    logger.info(f"Target model: {target_model_name}")
    logger.info(f"Draft tokens per step: {n_draft_tokens}")
    logger.info(f"Max generated tokens: {max_generated_tokens}")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info("=" * 80)

    # Start profiler
    profiler = BenchmarkProfiler()
    profiler.start("total_test")

    # Initialize speculative generator
    logger.info("Initializing speculative generator...")
    profiler.start("model_initialization")

    speculative_generator = SpeculativeGenerator(
        mesh_device=mesh_device,
        draft_model_name=draft_model_name,
        target_model_name=target_model_name,
        draft_model_config={"max_batch_size": 1},
        target_model_config={"max_batch_size": 1},
        instruct=instruct,
        max_seq_len=max_seq_len,
        n_draft_tokens=n_draft_tokens,
        optimizations=optimizations,
    )

    tokenizer = speculative_generator.get_tokenizer()
    profiler.end("model_initialization")

    # Display model information
    draft_layers = speculative_generator.draft_model_args.n_layers
    target_layers = speculative_generator.target_model_args.n_layers

    logger.info(f"✓ Draft model loaded: {draft_layers} layers")
    logger.info(f"✓ Target model loaded: {target_layers} layers")
    logger.info(f"✓ Layer ratio: {target_layers/draft_layers:.1f}x")

    # Process each prompt
    all_results = []

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
            [prompt],
            tokenizer,
            [speculative_generator.target_model_args],
            instruct,
            max_generated_tokens,
            max_prefill_len=max_seq_len,
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        profiler.end(f"preprocess_prompt_{prompt_idx}")

        # Run prefill on both models
        logger.info("Running prefill...")
        profiler.start(f"prefill_{prompt_idx}")

        draft_logits, target_logits = speculative_generator.prefill_both_models(
            input_tokens_prefill_pt,
            page_table=None,
            prompt_lens=decoding_pos,
        )

        target_token = torch.argmax(target_logits, dim=-1)
        profiler.end(f"prefill_{prompt_idx}")

        # Initialize for decoding
        all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
        all_outputs[0].append(int(target_token[0].item()))

        current_pos = torch.tensor([decoding_pos[0]])
        out_tok = target_token

        # Speculative decoding loop
        iteration = 0
        total_tokens_generated = 0
        total_draft_tokens = 0
        total_accepted_tokens = 0

        logger.info("Starting speculative decoding...")
        profiler.start(f"speculative_decode_{prompt_idx}")

        while total_tokens_generated < max_generated_tokens:
            # Perform speculative decoding step
            accepted_tokens, num_accepted, next_token, next_pos = speculative_generator.speculative_decode_step(
                out_tok, current_pos
            )

            # Update statistics
            total_draft_tokens += n_draft_tokens
            total_accepted_tokens += num_accepted
            total_tokens_generated += num_accepted

            # Update position and token
            current_pos = next_pos
            out_tok = next_token

            # Save output tokens
            for token in accepted_tokens:
                user_tok = token.item()
                if user_tok not in tokenizer.stop_tokens:
                    all_outputs[0].append(user_tok)
                else:
                    logger.info(f"Hit stop token at iteration {iteration}")
                    break

            # Log progress
            acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
            logger.info(
                f"Iter {iteration}: {num_accepted}/{n_draft_tokens} tokens accepted (running rate: {acceptance_rate:.1%})"
            )

            iteration += 1

            # Check for stop tokens
            if any(token.item() in tokenizer.stop_tokens for token in accepted_tokens):
                break

        profiler.end(f"speculative_decode_{prompt_idx}")

        # Generate final output
        final_text = tokenizer.decode(all_outputs[0])
        model_args = speculative_generator.target_model_args
        prompt_with_tags = tokenizer.decode(model_args.encode_prompt(prompt, instruct=instruct))
        generated_text = final_text.replace(prompt_with_tags, "", 1).strip()

        # Calculate metrics
        final_acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        effective_speedup = total_tokens_generated / iteration if iteration > 0 else 0

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "total_iterations": iteration,
            "total_tokens_generated": total_tokens_generated,
            "acceptance_rate": final_acceptance_rate,
            "effective_speedup": effective_speedup,
        }
        all_results.append(result)

        logger.info(f"✓ Generated {total_tokens_generated} tokens in {iteration} iterations")
        logger.info(f"✓ Acceptance rate: {final_acceptance_rate:.1%}")
        logger.info(f"✓ Effective speedup: {effective_speedup:.2f}x")
        logger.info(f"Generated text: {generated_text[:100]}...")

    profiler.end("total_test")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS")
    logger.info("=" * 80)

    avg_acceptance_rate = sum(r["acceptance_rate"] for r in all_results) / len(all_results)
    avg_speedup = sum(r["effective_speedup"] for r in all_results) / len(all_results)
    total_test_time = profiler.get_duration("total_test")
    init_time = profiler.get_duration("model_initialization")

    logger.info(f"Models: {draft_model_name} → {target_model_name}")
    logger.info(f"Model layers: {draft_layers} → {target_layers} ({target_layers/draft_layers:.1f}x)")
    logger.info(f"Prompts processed: {len(test_prompts)}")
    logger.info(f"Average acceptance rate: {avg_acceptance_rate:.1%}")
    logger.info(f"Average effective speedup: {avg_speedup:.2f}x")
    logger.info(f"Total test time: {total_test_time:.2f}s")
    logger.info(f"Model initialization time: {init_time:.2f}s")

    # Detailed results
    logger.info(f"\nDetailed Results:")
    for i, result in enumerate(all_results):
        logger.info(
            f"  Prompt {i+1}: {result['acceptance_rate']:.1%} acceptance, {result['effective_speedup']:.2f}x speedup"
        )

    # Cleanup
    speculative_generator.cleanup()

    # Assertions for test validation
    assert avg_acceptance_rate > 0, "No draft tokens were accepted"
    assert avg_speedup > 1.0, "No effective speedup achieved"
    assert all(len(r["generated_text"]) > 0 for r in all_results), "Some prompts generated no text"

    logger.info("✅ All test assertions passed!")

    return all_results


if __name__ == "__main__":
    # Run with pytest
    logger.info("Use pytest to run this test:")
    logger.info("pytest test_llama_models.py::test_speculative_llama_models -v")
