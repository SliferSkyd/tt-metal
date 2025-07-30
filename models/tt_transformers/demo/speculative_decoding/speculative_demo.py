# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.speculative_decoding.speculative_generator import SpeculativeGenerator
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import DecodersPrecision


def load_inputs(user_input, batch, instruct):
    """Load input prompts from json file or list."""
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)

    if len(user_input) < batch:
        logger.warning(
            f"Number of users in the file is less than the provided batch={batch}. Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch

    in_prompt = []
    for i in range(batch):
        prompt = user_input[i]["prompt"]
        in_prompt.append(prompt)
    return in_prompt


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    """Create page table for paged attention."""
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


@pytest.mark.parametrize(
    "input_prompts, instruct, max_seq_len, batch_size, max_generated_tokens, n_draft_tokens, draft_layers, target_layers",
    [
        (  # Basic speculative decoding test
            "models/tt_transformers/demo/speculative_decoding/sample_prompts.json",  # input_prompts
            True,  # instruct mode
            1024,  # max_seq_len
            1,  # batch_size
            50,  # max_generated_tokens
            4,  # n_draft_tokens
            16,  # draft_layers (smaller model)
            32,  # target_layers (larger model)
        ),
    ],
    ids=[
        "speculative-basic",
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
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_speculative_demo(
    input_prompts,
    instruct,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    n_draft_tokens,
    draft_layers,
    target_layers,
    optimizations,
    mesh_device,
    is_ci_env,
    reset_seeds,
    request,
):
    """
    Speculative decoding demo that uses a smaller draft model to generate tokens
    and a larger target model to verify them for improved inference speed.
    """

    # Override parameters from command line if they are provided
    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [0, 1]:
        instruct = request.config.getoption("--instruct")
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens

    # Speculative decoding specific parameters
    n_draft_tokens = request.config.getoption("--n_draft_tokens") or n_draft_tokens
    draft_layers = request.config.getoption("--draft_layers") or draft_layers
    target_layers = request.config.getoption("--target_layers") or target_layers

    # Set up configurations for draft and target models
    draft_model_config = {
        "max_batch_size": batch_size,
        "num_layers": draft_layers,
    }

    target_model_config = {
        "max_batch_size": batch_size,
        "num_layers": target_layers,
    }

    logger.info(f"Speculative Decoding Configuration:")
    logger.info(f"  Draft model layers: {draft_layers}")
    logger.info(f"  Target model layers: {target_layers}")
    logger.info(f"  Draft tokens per step: {n_draft_tokens}")
    logger.info(f"  Max sequence length: {max_seq_len}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max generated tokens: {max_generated_tokens}")

    # Start profiler
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info("Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * batch_size
    else:  # Inputs from file
        input_prompts = load_inputs(input_prompts, batch_size, instruct)
    profiler.end("loading_inputs")

    # Initialize speculative generator
    logger.info("Initializing speculative generator...")
    profiler.start("model_initialization")

    speculative_generator = SpeculativeGenerator(
        mesh_device=mesh_device,
        draft_model_config=draft_model_config,
        target_model_config=target_model_config,
        instruct=instruct,
        max_seq_len=max_seq_len,
        n_draft_tokens=n_draft_tokens,
        optimizations=optimizations,
    )

    tokenizer = speculative_generator.get_tokenizer()
    profiler.end("model_initialization")

    logger.info("Starting inference...")

    # Preprocess initial prompt inputs
    profiler.start("preprocess_prefill_inputs")
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        [speculative_generator.target_model_args],
        instruct,
        max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
    assert (
        max_generated_tokens + max_encoded_prompt_len <= max_seq_len
    ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)
    profiler.end("preprocess_prefill_inputs")

    # Run prefill on both models
    logger.info("Starting prefill for both models...")
    profiler.start("inference_prefill")

    draft_logits, target_logits = speculative_generator.prefill_both_models(
        input_tokens_prefill_pt,
        page_table=None,  # Not using paged attention for simplicity
        prompt_lens=decoding_pos,
    )

    # Get initial tokens from both models (should be the same)
    draft_token = torch.argmax(draft_logits, dim=-1)
    target_token = torch.argmax(target_logits, dim=-1)

    profiler.end("inference_prefill")
    logger.info("Prefill finished")

    # Keep track of generated outputs
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(target_token[user].item())  # Use target token for consistency
        all_outputs[user].append(user_tok)

    user_done = [False] * batch_size
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    # Start speculative decoding loop
    iteration = 0
    users_decoding = True
    total_generated_tokens = 0
    total_draft_tokens_generated = 0
    total_accepted_tokens = 0

    out_tok = target_token  # Start with target token

    logger.info("Starting speculative decode loop...")
    profiler.start("inference_decode")

    while users_decoding:
        if iteration == 0:
            profiler.start("compile_decode", iteration=0)
        else:
            profiler.start(f"inference_decode_time_{iteration}", iteration=0)

        # Perform speculative decoding step
        accepted_tokens, num_accepted, next_token, next_pos = speculative_generator.speculative_decode_step(
            out_tok, current_pos
        )

        # Update statistics
        total_draft_tokens_generated += n_draft_tokens
        total_accepted_tokens += num_accepted
        total_generated_tokens += num_accepted

        if iteration == 0:
            profiler.end("compile_decode", iteration=0)
            decode_iteration_time = profiler.get_duration("compile_decode", iteration=0)
        else:
            profiler.end(f"inference_decode_time_{iteration}", iteration=0)
            decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=0)

        # Calculate effective token generation rate
        tokens_per_second_per_user = num_accepted / decode_iteration_time
        logger.info(
            f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user "
            f"({batch_size*tokens_per_second_per_user:.1f} tok/s throughput), "
            f"accepted {num_accepted}/{n_draft_tokens} draft tokens"
        )

        # Update position and token
        current_pos = next_pos
        out_tok = next_token

        # Save output tokens
        for user in range(batch_size):
            if not user_done[user]:
                for token in accepted_tokens:
                    user_tok = token.item()
                    if user_tok not in tokenizer.stop_tokens:
                        all_outputs[user].append(user_tok)
                    else:
                        user_done[user] = True
                        logger.info(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
                        break

        # Print current outputs
        if not is_ci_env:
            for user in range(batch_size):
                text = "".join(tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.info("[User {}] {}".format(user, text))

        iteration += 1

        # Check stopping conditions
        if total_generated_tokens >= max_generated_tokens:
            users_decoding = False

        # Final print
        if not users_decoding:
            logger.info("Finished decoding, printing the final outputs...\n")
            for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
                text = tokenizer.decode(output)
                model_args = speculative_generator.target_model_args
                prompt_including_assistant_tags = tokenizer.decode(model_args.encode_prompt(prompt, instruct=instruct))
                text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)

                short_prompt = (
                    (prompt[:100] + "\n<long prompt not printed in full>\n" + prompt[-100:])
                    if len(prompt) > 200
                    else prompt
                )
                logger.info(
                    f"\n==USER {i} - PROMPT\n{short_prompt} \n==USER {i} - OUTPUT\n{text_after_prompt.strip()}\n"
                )

    profiler.end("inference_decode")
    profiler.end("run")

    # Calculate and log performance metrics
    total_time = profiler.get_duration("run")
    acceptance_rate = total_accepted_tokens / total_draft_tokens_generated if total_draft_tokens_generated > 0 else 0
    effective_speedup = total_generated_tokens / iteration  # Tokens per decode step

    logger.info("\n=== Speculative Decoding Performance ===")
    logger.info(f"Total decode iterations: {iteration}")
    logger.info(f"Total tokens generated: {total_generated_tokens}")
    logger.info(f"Total draft tokens generated: {total_draft_tokens_generated}")
    logger.info(f"Total draft tokens accepted: {total_accepted_tokens}")
    logger.info(f"Draft token acceptance rate: {acceptance_rate:.2%}")
    logger.info(f"Average tokens per decode step: {effective_speedup:.2f}")
    logger.info(f"Total runtime: {total_time:.2f}s")
    logger.info(f"Average effective token rate: {total_generated_tokens/total_time:.2f} tokens/s")

    # Cleanup
    speculative_generator.cleanup()

    return {
        "total_iterations": iteration,
        "total_tokens_generated": total_generated_tokens,
        "acceptance_rate": acceptance_rate,
        "effective_speedup": effective_speedup,
        "total_runtime": total_time,
    }


if __name__ == "__main__":
    # Simple test when run directly
    import ttnn

    # Get device
    device = ttnn.open_device(device_id=0)
    mesh_device = ttnn.create_mesh_device(device, x=1, y=1)

    # Test configuration
    test_config = {
        "input_prompts": ["What is the capital of France?"],
        "instruct": True,
        "max_seq_len": 512,
        "batch_size": 1,
        "max_generated_tokens": 20,
        "n_draft_tokens": 4,
        "draft_layers": 8,
        "target_layers": 16,
        "optimizations": lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    }

    try:
        logger.info("Running simple speculative decoding test...")

        speculative_generator = SpeculativeGenerator(
            mesh_device=mesh_device,
            draft_model_config={"max_batch_size": 1, "num_layers": 8},
            target_model_config={"max_batch_size": 1, "num_layers": 16},
            instruct=True,
            max_seq_len=512,
            n_draft_tokens=4,
            optimizations=test_config["optimizations"],
        )

        logger.info("Speculative generator initialized successfully!")

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.close_device(device)
