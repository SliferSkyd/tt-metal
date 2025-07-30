"""
Speculative Decoding Test - Using EXACT Same Timing as simple_text_demo.py
===========================================================================

This test uses IDENTICAL timing methodology as simple_text_demo.py to ensure
we're comparing apples-to-apples and determine if speedup is real or measurement artifact.
"""

import pytest
import torch
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.speculative_decoding.speculative_generator import SpeculativeGenerator
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import DecodersPrecision


@pytest.mark.timeout(420)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 62914560, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}[
            "N150"  # Using same device config as simple_text_demo
        ],
    ],
    indirect=True,
)
def test_speculative_timing_baseline(device_params, mesh_device):
    """
    Test speculative decoding using EXACT same timing methodology as simple_text_demo.py
    """

    # === EXACT SAME SETUP AS simple_text_demo.py ===
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Model configurations - same as your previous tests
    draft_model_name = "meta-llama/Llama-3.2-1B"
    target_model_name = "meta-llama/Llama-3.2-3B"
    n_draft_tokens = 4  # Using n=4 for direct comparison

    # Test configuration - matching simple_text_demo defaults
    max_generated_tokens = 30  # Same as your previous tests
    max_seq_len = 2048
    instruct = True
    global_batch_size = 1

    # Test prompt - using one from your sample prompts
    test_prompt = "The future of artificial intelligence will"

    logger.info(f"🚀 Starting Speculative Decoding Test (Exact simple_text_demo.py timing)")
    logger.info(f"Draft model: {draft_model_name}")
    logger.info(f"Target model: {target_model_name}")
    logger.info(f"n_draft_tokens: {n_draft_tokens}")
    logger.info(f"Max tokens: {max_generated_tokens}")

    # === MODEL LOADING (same timing as simple_text_demo) ===
    profiler.start("loading_inputs")

    # Create speculative generator (this loads both models)
    spec_generator = SpeculativeGenerator(
        draft_model_name=draft_model_name,
        target_model_name=target_model_name,
        mesh_device=mesh_device,
        n_draft_tokens=n_draft_tokens,
        max_seq_len=max_seq_len,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        instruct=instruct,
    )

    tokenizer = spec_generator.get_tokenizer()
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
        model_args=[spec_generator.target_model_args],
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

    logger.info("Running prefill on both models...")

    # Run prefill on both models
    draft_logits, target_logits = spec_generator.prefill_both_models(
        input_tokens_prefill_pt,
        page_table=None,
        prompt_lens=decoding_pos,
    )

    prefilled_token = torch.argmax(target_logits, dim=-1)

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

        # === SPECULATIVE DECODE STEP ===
        accepted_tokens, num_accepted, next_token, next_pos = spec_generator.speculative_decode_step(
            out_tok, current_pos
        )

        # End timing exactly like simple_text_demo
        if iteration == 0:  # First iteration accounts for compile time
            profiler.end(f"compile_decode", iteration=batch_idx)
            decode_iteration_time = profiler.get_duration("compile_decode", iteration=batch_idx)

        profiler.end(f"inference_decode_time_{iteration}", iteration=batch_idx)
        decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=batch_idx)

        # === EXACT PERFORMANCE LOGGING FROM simple_text_demo ===
        tokens_per_second_per_user = num_accepted / decode_iteration_time  # Account for multiple tokens per iteration
        logger.info(
            f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput) [Accepted: {num_accepted}/{n_draft_tokens}]"
        )

        # Update state exactly like simple_text_demo
        current_pos = next_pos
        out_tok = next_token

        # Save output tokens
        for token in accepted_tokens:
            user_tok = token.item()
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
    logger.info("SPECULATIVE DECODING RESULTS (simple_text_demo.py timing)")
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

    # Cleanup
    spec_generator.cleanup()

    # Return measurements for further analysis
    return measurements, tokens_generated, decode_tok_s_user, avg_time_to_first_token


if __name__ == "__main__":
    pytest.main([__file__])
