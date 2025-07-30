"""
Baseline 3B Model Test - Using EXACT Same Timing as simple_text_demo.py
========================================================================

This test uses IDENTICAL timing methodology as simple_text_demo.py to run
ONLY the 3B model for direct comparison with speculative decoding results.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


@pytest.mark.timeout(420)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 62914560, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}[
            "N150"  # Using same device config as simple_text_demo
        ],
    ],
    indirect=True,
)
def test_baseline_timing_3b(device_params, mesh_device):
    """
    Test baseline 3B model using EXACT same timing methodology as simple_text_demo.py
    """

    # === EXACT SAME SETUP AS simple_text_demo.py ===
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Model configuration
    model_name = "meta-llama/Llama-3.1-8B"

    # Test configuration - matching simple_text_demo defaults
    max_generated_tokens = 30  # Same as speculative test
    max_seq_len = 2048
    instruct = True
    global_batch_size = 1

    # Test prompt - EXACT same as speculative test
    test_prompt = "The future of artificial intelligence will"

    logger.info(f"🚀 Starting Baseline 3B Test (Exact simple_text_demo.py timing)")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max tokens: {max_generated_tokens}")

    # === MODEL LOADING (same timing as simple_text_demo) ===
    profiler.start("loading_inputs")

    # Set environment variable for model loading
    original_hf_model = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = model_name

    try:
        # Create model exactly like simple_text_demo
        model_args, model, kv_cache, state_dict = create_tt_model(
            mesh_device=mesh_device,
            instruct=instruct,
            max_batch_size=1,
            optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
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

    profiler.end("loading_inputs")

    # === PREPROCESSING (same as simple_text_demo) ===
    batch_idx = 0  # Following simple_text_demo pattern

    profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)

    # Preprocess inputs exactly like simple_text_demo
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts=[test_prompt],
        tokenizer=tokenizer,
        model_args=[model_args],
        instruct=instruct,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    # Convert to proper tensor format
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).squeeze(1)

    profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)

    # === PREFILL (same timing as simple_text_demo) ===
    profiler.start(f"compile_prefill", iteration=batch_idx)
    profiler.start(f"inference_prefill", iteration=batch_idx)

    logger.info("Running prefill...")

    # Run prefill
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=None,
        kv_cache=[kv_cache],
        prompt_lens=decoding_pos,
    )

    prefilled_token = torch.argmax(logits, dim=-1)

    profiler.end(f"compile_prefill", iteration=batch_idx)
    profiler.end(f"inference_prefill", iteration=batch_idx)

    # === DECODE LOOP (EXACT same structure as simple_text_demo) ===

    # Initialize decode state exactly like simple_text_demo
    all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
    all_outputs[0].append(int(prefilled_token[0].item()))

    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token

    # Decode loop variables
    iteration = 0
    users_decoding = True
    user_done = [False] * global_batch_size
    num_tokens_generated_decode = [1]  # Starting with prefilled token

    logger.info(f"Starting decode loop...")

    # === EXACT TIMING PATTERN FROM simple_text_demo.py ===
    profiler.start(f"inference_decode", iteration=batch_idx)

    while users_decoding and num_tokens_generated_decode[0] <= max_generated_tokens:
        # EXACT timing pattern from simple_text_demo
        if iteration == 0:  # First iteration accounts for compile time
            profiler.start(f"compile_decode", iteration=batch_idx)

        profiler.start(f"inference_decode_time_{iteration}", iteration=batch_idx)

        # === STANDARD DECODE STEP (like simple_text_demo) ===
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=True,
            kv_cache=[kv_cache],
        )

        # Sample next token (exactly like simple_text_demo)
        _, next_token = sample_host(
            logits,
            temperature=0.0,
            top_p=1.0,
            on_host=True,
        )

        # End timing exactly like simple_text_demo
        if iteration == 0:  # First iteration accounts for compile time
            profiler.end(f"compile_decode", iteration=batch_idx)
            decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)

        profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
        decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

        # === EXACT PERFORMANCE LOGGING FROM simple_text_demo ===
        tokens_per_second_per_user = 1 / decode_iteration_time  # Standard: 1 token per iteration
        logger.info(
            f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

        # Update state exactly like simple_text_demo
        current_pos += 1
        out_tok = next_token

        # Save output token
        user_tok = next_token.item()
        if user_tok not in tokenizer.stop_tokens and user_done[0] == False:
            all_outputs[0].append(user_tok)
            num_tokens_generated_decode[0] += 1
        else:
            user_done[0] = True
            logger.info(f"Hit stop token at iteration {iteration}")
            break

        if user_done[0] or num_tokens_generated_decode[0] > max_generated_tokens:
            users_decoding = False

        iteration += 1

    profiler.end(f"inference_decode", iteration=batch_idx)
    profiler.end("run")

    # === EXACT PERFORMANCE CALCULATION FROM simple_text_demo.py ===

    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")

    total_inference_prefill_time = profiler.get_duration("inference_prefill")
    total_inference_decode_time = 0
    for i in range(1, iteration):  # Iteration 0 is compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    # EXACT same calculations as simple_text_demo
    avg_time_to_first_token = total_inference_prefill_time / global_batch_size
    avg_decode_iteration_time = total_inference_decode_time / (iteration - 1) if iteration > 1 else 0

    prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * global_batch_size
    decode_tok_s_user = (
        (num_tokens_generated_decode[0] - 1) / total_inference_decode_time if total_inference_decode_time > 0 else 0
    )
    decode_tok_s = decode_tok_s_user * global_batch_size

    # EXACT same measurements dict as simple_text_demo
    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,  # tokens/s
        "decode_t/s/u": decode_tok_s_user,  # tokens/s/u
        "decode_t/s": decode_tok_s,  # tokens/s
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # === RESULTS LOGGING ===

    generated_text = tokenizer.decode(all_outputs[0])
    tokens_generated = num_tokens_generated_decode[0] - 1  # Subtract prefilled token

    logger.info("=" * 80)
    logger.info("BASELINE 8B MODEL RESULTS (simple_text_demo.py timing)")
    logger.info("=" * 80)
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Generated: {generated_text}")
    logger.info(f"Tokens generated: {tokens_generated}")
    logger.info(f"Total iterations: {iteration}")
    logger.info("")
    logger.info("=== TIMING METRICS (same as simple_text_demo.py) ===")
    for key, value in measurements.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}s")
        else:
            logger.info(f"{key}: {value}")
    logger.info("")
    logger.info("=== KEY PERFORMANCE METRICS ===")
    logger.info(f"⏱️  Time to First Token (TTFT): {avg_time_to_first_token:.4f}s")
    logger.info(f"🚀 Decode Speed (tok/s/user): {decode_tok_s_user:.2f}")
    logger.info(f"📈 Decode Speed (tok/s): {decode_tok_s:.2f}")
    logger.info("=" * 80)

    # Return measurements for further analysis
    return measurements, tokens_generated, decode_tok_s_user, avg_time_to_first_token


if __name__ == "__main__":
    pytest.main([__file__])
