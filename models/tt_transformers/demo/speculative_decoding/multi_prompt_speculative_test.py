#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Multi-prompt speculative decoding test using sample_prompts.json
Skips first prompt for warmup and calculates accurate performance metrics.
"""

import json
import os

import pytest
import torch
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.speculative_decoding.speculative_generator import SpeculativeGenerator
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import DecodersPrecision


def load_prompts_from_json(json_file="sample_prompts.json"):
    """Load prompts from JSON file."""
    with open(json_file, "r") as f:
        prompts_data = json.load(f)
    return [item["prompt"] for item in prompts_data]


@pytest.mark.parametrize(
    "draft_model_name, target_model_name, max_seq_len, max_generated_tokens",
    [
        (
            "meta-llama/Llama-3.2-1B",  # draft_model_name
            "meta-llama/Llama-3.2-3B",  # target_model_name
            1024,  # max_seq_len
            30,  # max_generated_tokens
        ),
    ],
    ids=[
        "speculative-1b-3b-multi-prompt",
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
            os.environ.get("MESH_DEVICE"), (1, 1)
        )
    ],
    indirect=True,
)
def test_multi_prompt_speculative_decoding(
    draft_model_name,
    target_model_name,
    max_seq_len,
    max_generated_tokens,
    optimizations,
    mesh_device,
    reset_seeds,
):
    """
    Test speculative decoding with multiple prompts, skipping first for warmup.
    """

    instruct = True

    # Load all prompts from JSON
    all_prompts = load_prompts_from_json()
    logger.info(f"Loaded {len(all_prompts)} prompts from sample_prompts.json")

    logger.info("=" * 80)
    logger.info("MULTI-PROMPT SPECULATIVE DECODING TEST")
    logger.info("=" * 80)
    logger.info(f"Draft Model: {draft_model_name}")
    logger.info(f"Target Model: {target_model_name}")
    logger.info(f"Total prompts: {len(all_prompts)}")
    logger.info(f"Prompts for metrics: {len(all_prompts)-1} (skipping first for warmup)")
    logger.info(f"Max generated tokens: {max_generated_tokens}")
    logger.info("=" * 80)

    # Start profiler
    profiler = BenchmarkProfiler()
    profiler.start("total_test")

    # Initialize speculative generator
    logger.info("Initializing speculative decoding models...")
    profiler.start("model_initialization")

    spec_generator = SpeculativeGenerator(
        mesh_device=mesh_device,
        draft_model_name=draft_model_name,
        target_model_name=target_model_name,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        n_draft_tokens=4,
    )

    profiler.end("model_initialization")

    # Display model information
    draft_layers = spec_generator.draft_model_args.n_layers
    target_layers = spec_generator.target_model_args.n_layers
    logger.info(f"✓ Draft model loaded: {draft_layers} layers")
    logger.info(f"✓ Target model loaded: {target_layers} layers")

    # Process all prompts
    all_results = []
    total_prefill_time = 0
    total_decode_time = 0
    total_tokens_generated = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0

    for prompt_idx, prompt in enumerate(all_prompts):
        is_warmup = prompt_idx == 0
        logger.info(
            f"\n--- Processing Prompt {prompt_idx + 1}/{len(all_prompts)} {'(WARMUP)' if is_warmup else ''} ---"
        )
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
            spec_generator.tokenizer,
            [spec_generator.target_model_args],
            instruct,
            max_generated_tokens,
            max_prefill_len=max_seq_len,
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        profiler.end(f"preprocess_prompt_{prompt_idx}")

        # Run speculative decoding
        logger.info("Running speculative decoding...")
        profiler.start(f"speculative_generation_{prompt_idx}")

        outputs, stats = spec_generator.generate_speculative(
            input_tokens_prefill_pt,
            prompt_lens=decoding_pos,
            max_generated_tokens=max_generated_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        profiler.end(f"speculative_generation_{prompt_idx}")

        generation_time = profiler.get_duration(f"speculative_generation_{prompt_idx}")

        # Extract metrics
        prefill_time = stats.get("prefill_time", 0)
        decode_time = stats.get("decode_time", 0)
        tokens_generated = stats.get("tokens_generated", 0)
        draft_tokens = stats.get("total_draft_tokens", 0)
        accepted_tokens = stats.get("total_accepted_tokens", 0)
        acceptance_rate = stats.get("acceptance_rate", 0)
        effective_speedup = stats.get("effective_speedup", 0)

        # Decode output
        final_text = spec_generator.tokenizer.decode(outputs[0])
        prompt_with_tags = spec_generator.tokenizer.decode(
            spec_generator.target_model_args.encode_prompt(prompt, instruct=instruct)
        )
        generated_text = final_text.replace(prompt_with_tags, "", 1).strip()

        # Calculate metrics
        ttft = prefill_time  # Time to first token
        tokens_per_second = tokens_generated / decode_time if decode_time > 0 else 0

        result = {
            "prompt": prompt,
            "prompt_idx": prompt_idx,
            "is_warmup": is_warmup,
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "generation_time": generation_time,
            "ttft": ttft,
            "tokens_per_second": tokens_per_second,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "effective_speedup": effective_speedup,
        }
        all_results.append(result)

        # Add to totals (skip warmup for final metrics)
        if not is_warmup:
            total_prefill_time += prefill_time
            total_decode_time += decode_time
            total_tokens_generated += tokens_generated
            total_draft_tokens += draft_tokens
            total_accepted_tokens += accepted_tokens

        logger.info(f"✓ Generated {tokens_generated} tokens")
        logger.info(f"✓ TTFT: {ttft*1000:.1f}ms")
        logger.info(f"✓ Tokens/sec: {tokens_per_second:.2f}")
        logger.info(f"✓ Acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"✓ Effective speedup: {effective_speedup:.2f}x")
        logger.info(f"Generated text: {generated_text[:100]}...")

    profiler.end("total_test")

    # Calculate metrics excluding warmup
    non_warmup_results = [r for r in all_results if not r["is_warmup"]]
    num_prompts_for_metrics = len(non_warmup_results)

    # Calculate averages (excluding warmup)
    avg_ttft = total_prefill_time / num_prompts_for_metrics if num_prompts_for_metrics > 0 else 0
    avg_tokens_per_second = total_tokens_generated / total_decode_time if total_decode_time > 0 else 0
    avg_acceptance_rate = total_accepted_tokens / total_draft_tokens * 100 if total_draft_tokens > 0 else 0
    avg_effective_speedup = (
        sum(r["effective_speedup"] for r in non_warmup_results) / num_prompts_for_metrics
        if num_prompts_for_metrics > 0
        else 0
    )

    total_test_time = profiler.get_duration("total_test")
    init_time = profiler.get_duration("model_initialization")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-PROMPT SPECULATIVE DECODING RESULTS")
    logger.info("=" * 80)
    logger.info(f"Draft Model: {draft_model_name} ({draft_layers} layers)")
    logger.info(f"Target Model: {target_model_name} ({target_layers} layers)")
    logger.info(f"Total prompts processed: {len(all_prompts)}")
    logger.info(f"Prompts used for metrics: {num_prompts_for_metrics} (excluding warmup)")
    logger.info(f"Total tokens generated: {total_tokens_generated}")
    logger.info(f"Average TTFT: {avg_ttft*1000:.1f}ms")
    logger.info(f"Average tokens/sec: {avg_tokens_per_second:.2f}")
    logger.info(f"Average acceptance rate: {avg_acceptance_rate:.1f}%")
    logger.info(f"Average effective speedup: {avg_effective_speedup:.2f}x")
    logger.info(f"Total test time: {total_test_time:.2f}s")
    logger.info(f"Model initialization time: {init_time:.2f}s")

    # Detailed results (excluding warmup)
    logger.info(f"\nDetailed Results (excluding warmup):")
    for i, result in enumerate(non_warmup_results):
        logger.info(
            f"  Prompt {result['prompt_idx']}: {result['tokens_per_second']:.2f} tok/s, TTFT: {result['ttft']*1000:.1f}ms, Accept: {result['acceptance_rate']:.1f}%"
        )

    # Test assertions
    assert total_tokens_generated > 0, "No tokens were generated"
    assert avg_tokens_per_second > 0, "Invalid token generation rate"
    assert avg_acceptance_rate > 50, "Acceptance rate too low"
    assert all(len(r["generated_text"]) > 0 for r in all_results), "Some prompts generated no text"

    logger.info("✅ All test assertions passed!")

    return {
        "draft_model": draft_model_name,
        "target_model": target_model_name,
        "draft_layers": draft_layers,
        "target_layers": target_layers,
        "total_prompts": len(all_prompts),
        "prompts_for_metrics": num_prompts_for_metrics,
        "total_tokens_generated": total_tokens_generated,
        "avg_ttft": avg_ttft,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_acceptance_rate": avg_acceptance_rate,
        "avg_effective_speedup": avg_effective_speedup,
        "total_test_time": total_test_time,
        "init_time": init_time,
        "all_results": all_results,
        "non_warmup_results": non_warmup_results,
    }


if __name__ == "__main__":
    # Run with pytest
    logger.info("Use pytest to run this test:")
    logger.info("pytest multi_prompt_speculative_test.py::test_multi_prompt_speculative_decoding -v")
