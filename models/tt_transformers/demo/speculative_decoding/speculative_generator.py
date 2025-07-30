# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import create_tt_model, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


class SpeculativeGenerator:
    """
    Speculative Decoding Generator that manages both draft and target models.

    The draft model generates multiple tokens quickly, and the target model
    verifies these tokens in parallel for improved inference speed.
    """

    def __init__(
        self,
        mesh_device,
        draft_model_name=None,
        target_model_name=None,
        draft_model_config=None,
        target_model_config=None,
        instruct=True,
        max_seq_len=1024,
        n_draft_tokens=4,
        optimizations=None,
    ):
        """
        Initialize speculative generator with draft and target models.

        Args:
            mesh_device: TT mesh device
            draft_model_name: HuggingFace model name for draft model (e.g., "meta-llama/Llama-3.2-1B")
            target_model_name: HuggingFace model name for target model (e.g., "meta-llama/Llama-3.2-3B")
            draft_model_config: Configuration for draft model (smaller, faster)
            target_model_config: Configuration for target model (larger, more accurate)
            instruct: Whether to use instruct mode
            max_seq_len: Maximum sequence length
            n_draft_tokens: Number of tokens to generate speculatively
            optimizations: Model optimization settings
        """
        self.mesh_device = mesh_device
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name
        self.instruct = instruct
        self.max_seq_len = max_seq_len
        self.n_draft_tokens = n_draft_tokens
        self.optimizations = optimizations or (
            lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
        )

        # Set default configurations if not provided
        if draft_model_config is None:
            draft_model_config = {
                "max_batch_size": 1,
                "num_layers": None,  # Will use default from model
            }

        if target_model_config is None:
            target_model_config = {
                "max_batch_size": 1,
                "num_layers": None,  # Will use default from model
            }

        self.draft_config = draft_model_config
        self.target_config = target_model_config

        # Store original HF_MODEL env var to restore later
        self.original_hf_model = os.environ.get("HF_MODEL")

        # Initialize models
        self._setup_models()

    def _setup_models(self):
        """Setup both draft and target models with different HF model names."""
        logger.info("Setting up draft model...")

        # Temporarily set HF_MODEL for draft model
        if self.draft_model_name:
            os.environ["HF_MODEL"] = self.draft_model_name
            logger.info(f"Using draft model: {self.draft_model_name}")

        # Create draft model (smaller/faster)
        self.draft_model_args, self.draft_model, self.draft_kv_cache, draft_state_dict = create_tt_model(
            mesh_device=self.mesh_device,
            instruct=self.instruct,
            max_batch_size=self.draft_config["max_batch_size"],
            optimizations=self.optimizations,
            max_seq_len=self.max_seq_len,
            dtype=ttnn.bfloat8_b,
            num_layers=self.draft_config.get("num_layers"),
        )

        logger.info("Setting up target model...")

        # Temporarily set HF_MODEL for target model
        if self.target_model_name:
            os.environ["HF_MODEL"] = self.target_model_name
            logger.info(f"Using target model: {self.target_model_name}")

        # Create target model (larger/more accurate)
        # Don't reuse state_dict since the models are different sizes
        self.target_model_args, self.target_model, self.target_kv_cache, target_state_dict = create_tt_model(
            mesh_device=self.mesh_device,
            instruct=self.instruct,
            max_batch_size=self.target_config["max_batch_size"],
            optimizations=self.optimizations,
            max_seq_len=self.max_seq_len,
            dtype=ttnn.bfloat8_b,
            state_dict=None,  # Different model sizes need separate state dicts
            num_layers=self.target_config.get("num_layers"),
        )

        # Restore original HF_MODEL environment variable
        if self.original_hf_model:
            os.environ["HF_MODEL"] = self.original_hf_model
        elif "HF_MODEL" in os.environ:
            del os.environ["HF_MODEL"]

        # Create generators for both models
        self.draft_generator = Generator(
            [self.draft_model], [self.draft_model_args], self.mesh_device, tokenizer=self.draft_model_args.tokenizer
        )
        self.target_generator = Generator(
            [self.target_model], [self.target_model_args], self.mesh_device, tokenizer=self.target_model_args.tokenizer
        )

        # Use the target model tokenizer (should be the same for both Llama models anyway)
        self.tokenizer = self.target_model_args.tokenizer

        logger.info(f"Draft model: {self.draft_model_args.model_name} with {self.draft_model_args.n_layers} layers")
        logger.info(f"Target model: {self.target_model_args.model_name} with {self.target_model_args.n_layers} layers")

        # Verify the models are different sizes
        if self.draft_model_args.n_layers >= self.target_model_args.n_layers:
            logger.warning(
                f"Draft model ({self.draft_model_args.n_layers} layers) should be smaller than target model ({self.target_model_args.n_layers} layers) for optimal performance"
            )

    def prefill_both_models(self, input_tokens, page_table=None, prompt_lens=None):
        """
        Run prefill on both draft and target models.

        Args:
            input_tokens: Input token tensor
            page_table: Page table for paged attention
            prompt_lens: Length of prompts for each batch item

        Returns:
            Tuple of (draft_logits, target_logits)
        """
        logger.info("Running prefill on draft model...")
        draft_logits = self.draft_generator.prefill_forward_text(
            input_tokens,
            page_table=page_table,
            kv_cache=[self.draft_kv_cache],
            prompt_lens=prompt_lens,
        )

        logger.info("Running prefill on target model...")
        target_logits = self.target_generator.prefill_forward_text(
            input_tokens,
            page_table=page_table,
            kv_cache=[self.target_kv_cache],
            prompt_lens=prompt_lens,
        )

        return draft_logits, target_logits

    def generate_draft_tokens(self, current_token, current_pos, n_tokens=None):
        """
        Generate n_draft_tokens from the draft model.

        Args:
            current_token: Current token tensor
            current_pos: Current position tensor
            n_tokens: Number of tokens to generate (defaults to self.n_draft_tokens)

        Returns:
            List of generated draft tokens
        """
        if n_tokens is None:
            n_tokens = self.n_draft_tokens

        draft_tokens = []
        token = current_token.clone()
        pos = current_pos.clone()

        logger.debug(f"Generating {n_tokens} draft tokens...")

        for i in range(n_tokens):
            # Generate next token with draft model
            logits = self.draft_generator.decode_forward_text(
                token,
                pos,
                enable_trace=True,
                kv_cache=[self.draft_kv_cache],
            )

            # Sample next token (greedy for now)
            _, next_token = sample_host(logits, temperature=0.0, top_p=1.0, on_host=True)

            draft_tokens.append(next_token.clone())
            token = next_token
            pos += 1

        return draft_tokens

    def verify_draft_tokens(self, draft_tokens, current_token, current_pos):
        """
        Verify draft tokens using the target model.

        Args:
            draft_tokens: List of draft tokens to verify
            current_token: Current token before draft generation
            current_pos: Current position before draft generation

        Returns:
            Tuple of (accepted_tokens, num_accepted, next_token)
        """
        logger.debug(f"Verifying {len(draft_tokens)} draft tokens with target model...")

        accepted_tokens = []
        token = current_token.clone()
        pos = current_pos.clone()

        for i, draft_token in enumerate(draft_tokens):
            # Get target model prediction for current position
            target_logits = self.target_generator.decode_forward_text(
                token,
                pos,
                enable_trace=True,
                kv_cache=[self.target_kv_cache],
            )

            # Sample from target model (greedy for now)
            _, target_token = sample_host(target_logits, temperature=0.0, top_p=1.0, on_host=True)

            # Check if draft token matches target prediction
            if draft_token.item() == target_token.item():
                # Accept the draft token
                accepted_tokens.append(draft_token)
                token = draft_token
                pos += 1
                logger.debug(f"Accepted draft token {i+1}/{len(draft_tokens)}: {draft_token.item()}")
            else:
                # Reject draft token, use target prediction instead
                accepted_tokens.append(target_token)
                logger.debug(f"Rejected draft token {i+1}/{len(draft_tokens)}, using target prediction instead")
                return accepted_tokens, len(accepted_tokens), target_token

        # All draft tokens were accepted, generate one more token from target
        final_pos = pos
        final_logits = self.target_generator.decode_forward_text(
            token,
            final_pos,
            enable_trace=True,
            kv_cache=[self.target_kv_cache],
        )
        _, final_token = sample_host(final_logits, temperature=0.0, top_p=1.0, on_host=True)

        return accepted_tokens, len(accepted_tokens), final_token

    def speculative_decode_step(self, current_token, current_pos):
        """
        Perform one step of speculative decoding.

        Args:
            current_token: Current token tensor
            current_pos: Current position tensor

        Returns:
            Tuple of (generated_tokens, num_tokens_generated, next_token, next_pos)
        """
        # Generate draft tokens
        draft_tokens = self.generate_draft_tokens(current_token, current_pos)

        # Verify draft tokens with target model
        accepted_tokens, num_accepted, next_token = self.verify_draft_tokens(draft_tokens, current_token, current_pos)

        # Calculate new position
        next_pos = current_pos + len(accepted_tokens)

        logger.info(f"Speculative step: {num_accepted}/{len(draft_tokens)} draft tokens accepted")

        return accepted_tokens, len(accepted_tokens), next_token, next_pos

    def generate_speculative(
        self,
        input_tokens,
        prompt_lens,
        max_generated_tokens=30,
        temperature=0.0,
        top_p=1.0,
    ):
        """
        Complete speculative generation for a batch of prompts.

        Args:
            input_tokens: Input token tensor
            prompt_lens: List of prompt lengths
            max_generated_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Tuple of (output_tokens, statistics)
        """
        import time

        logger.info(f"Starting speculative generation for {max_generated_tokens} tokens")

        # Initialize timing
        start_time = time.time()

        # Do prefill for both models
        logger.info("Running prefill...")
        prefill_start = time.time()
        draft_logits, target_logits = self.prefill_both_models(input_tokens, prompt_lens=prompt_lens)
        prefill_time = time.time() - prefill_start

        # Get first token from target model prefill
        target_token = torch.argmax(target_logits, dim=-1)

        # Initialize outputs with prompt tokens
        batch_size = len(prompt_lens)
        all_outputs = []
        for i in range(batch_size):
            prompt_len = prompt_lens[i]
            prompt_tokens = input_tokens[i, :prompt_len].tolist()
            all_outputs.append(prompt_tokens + [int(target_token[i].item())])

        current_pos = torch.tensor(prompt_lens, dtype=torch.long)
        out_tok = target_token

        # Initialize statistics
        total_tokens_generated = 0
        total_draft_tokens = 0
        total_accepted_tokens = 0
        iteration_times = []

        # Start decode loop
        decode_start = time.time()
        iteration = 0

        logger.info("Starting speculative decode loop...")

        while total_tokens_generated < max_generated_tokens:
            iter_start = time.time()

            # Perform one speculative decode step
            accepted_tokens, num_accepted, next_token, next_pos = self.speculative_decode_step(out_tok, current_pos)

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            # Update statistics
            total_draft_tokens += self.n_draft_tokens
            total_accepted_tokens += num_accepted
            total_tokens_generated += num_accepted + 1  # +1 for the final token

            # Add accepted tokens and final token to outputs
            for i in range(batch_size):
                # Add accepted draft tokens
                for token in accepted_tokens:
                    if isinstance(token, torch.Tensor):
                        all_outputs[i].append(int(token.item()))
                    else:
                        all_outputs[i].append(int(token))

                # Add final token
                if isinstance(next_token, torch.Tensor):
                    final_token_val = int(next_token[i].item() if next_token.dim() > 0 else next_token.item())
                else:
                    final_token_val = int(next_token)
                all_outputs[i].append(final_token_val)

            # Update for next iteration
            current_pos = next_pos.clone() if isinstance(next_pos, torch.Tensor) else torch.tensor([next_pos])
            out_tok = next_token
            iteration += 1

            # Check if we've generated enough tokens
            if total_tokens_generated >= max_generated_tokens:
                break

            # Progress logging
            if iteration % 5 == 0:
                acceptance_rate = (total_accepted_tokens / total_draft_tokens * 100) if total_draft_tokens > 0 else 0
                logger.info(
                    f"Iteration {iteration}: {num_accepted}/{self.n_draft_tokens} accepted, "
                    f"overall acceptance: {acceptance_rate:.1f}%"
                )

        decode_time = time.time() - decode_start
        total_time = time.time() - start_time

        # Calculate final statistics
        acceptance_rate = (total_accepted_tokens / total_draft_tokens * 100) if total_draft_tokens > 0 else 0
        effective_speedup = (total_accepted_tokens + iteration) / iteration if iteration > 0 else 1.0

        stats = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "tokens_generated": total_tokens_generated,
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "effective_speedup": effective_speedup,
            "iterations": iteration,
            "avg_iteration_time": sum(iteration_times) / len(iteration_times) if iteration_times else 0,
        }

        logger.info(f"Speculative generation complete:")
        logger.info(f"  Tokens generated: {total_tokens_generated}")
        logger.info(f"  Acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"  Effective speedup: {effective_speedup:.2f}x")
        logger.info(f"  Total time: {total_time:.2f}s")

        return all_outputs, stats

    def get_tokenizer(self):
        """Get the tokenizer used by both models."""
        return self.tokenizer

    def cleanup(self):
        """Cleanup resources."""
        # Restore original environment if needed
        if self.original_hf_model:
            os.environ["HF_MODEL"] = self.original_hf_model
        elif "HF_MODEL" in os.environ:
            del os.environ["HF_MODEL"]
