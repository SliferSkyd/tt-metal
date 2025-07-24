#!/usr/bin/env python3
"""
Speculative Decoding API using demo_api with 1B draft and 3B target models.
Implements greedy speculative decoding with configurable N tokens.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.tt_transformers.demo.demo_api import create_demo_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    draft_model_name: str = "meta-llama/Llama-3.2-1B"
    target_model_name: str = "meta-llama/Llama-3.2-3B"
    max_batch_size: int = 1  # Reduced to avoid core allocation issues
    max_seq_len: int = 512  # Reduced sequence length
    data_parallel: int = 1
    paged_attention: bool = True
    page_params: Dict = None
    sampling_params: Dict = None
    max_generated_tokens: int = 200
    stop_at_eos: bool = True
    eos_token_id: int = 2

    def __post_init__(self):
        if self.page_params is None:
            self.page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
        if self.sampling_params is None:
            self.sampling_params = {"temperature": 0, "top_p": 0.08}


class SpeculativeDecodingAPI:
    """
    Speculative decoding implementation using 1B draft and 3B target models.
    """

    def __init__(self, config: SpeculativeConfig, mesh_device: ttnn.MeshDevice):
        self.config = config
        self.mesh_device = mesh_device

        # Initialize draft and target models
        self.draft_api = None
        self.target_api = None

        # Performance metrics
        self.metrics = {
            "total_tokens_generated": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "draft_tokens_generated": 0,
            "target_forward_passes": 0,
            "speculative_rounds": 0,
            "total_time": 0.0,
            "draft_time": 0.0,
            "target_time": 0.0,
        }

    def initialize_models(self):
        """Initialize both draft and target models."""
        logger.info("Initializing draft model (1B)...")

        # Create draft model API
        self.draft_api = create_demo_api(
            mesh_device=self.mesh_device,
            instruct=True,
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            data_parallel=self.config.data_parallel,
            paged_attention=False,  # Disable paged attention to avoid batch size issues
            page_params=None,
        )

        # Set draft model environment
        os.environ["HF_MODEL"] = self.config.draft_model_name
        self.draft_api.initialize_model()

        logger.info("Initializing target model (3B)...")

        # Create target model API
        self.target_api = create_demo_api(
            mesh_device=self.mesh_device,
            instruct=True,
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            data_parallel=self.config.data_parallel,
            paged_attention=False,  # Disable paged attention to avoid batch size issues
            page_params=None,
        )

        # Set target model environment
        os.environ["HF_MODEL"] = self.config.target_model_name
        self.target_api.initialize_model()

        logger.info("Both models initialized successfully!")

    def greedy_speculative_decode(self, input_prompts: List[str], max_tokens: int, n_draft_tokens: int = 4) -> Dict:
        """
        Perform greedy speculative decoding.

        Args:
            input_prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            n_draft_tokens: Number of tokens to generate with draft model

        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()

        # Initialize generation state
        batch_size = len(input_prompts)
        generated_tokens_per_batch = [[] for _ in range(batch_size)]
        current_positions = [0] * batch_size
        active_sequences = list(range(batch_size))

        # Run encoder (prefill) for both models
        logger.info("Running encoder for draft model...")
        draft_prefill_start = time.time()
        draft_sampled_tokens, draft_decoding_pos, draft_prefill_lens = self.draft_api.run_encoder(
            input_prompts=input_prompts, max_generated_tokens=max_tokens
        )
        self.metrics["draft_time"] += time.time() - draft_prefill_start

        logger.info("Running encoder for target model...")
        target_prefill_start = time.time()
        target_sampled_tokens, target_decoding_pos, target_prefill_lens = self.target_api.run_encoder(
            input_prompts=input_prompts, max_generated_tokens=max_tokens
        )
        self.metrics["target_time"] += time.time() - target_prefill_start

        # For speculative decoding, we'll use both models' KV caches
        # The draft model will be used for generation but we'll verify with target
        draft_kv_cache = self.draft_api.tt_kv_cache
        draft_page_table = self.draft_api.tt_page_table
        target_kv_cache = self.target_api.tt_kv_cache
        target_page_table = self.target_api.tt_page_table

        logger.info(f"Starting speculative decoding with N={n_draft_tokens}")

        while len(active_sequences) > 0 and self.metrics["total_tokens_generated"] < max_tokens:
            self.metrics["speculative_rounds"] += 1

            # Step 1: Generate N tokens with draft model
            draft_start = time.time()
            draft_tokens = self._generate_draft_tokens(
                active_sequences, n_draft_tokens, draft_kv_cache, draft_page_table, draft_prefill_lens
            )
            self.metrics["draft_time"] += time.time() - draft_start

            if not draft_tokens:
                break

            # Step 2: Verify with target model
            target_start = time.time()
            verification_results = self._verify_draft_tokens(
                active_sequences, draft_tokens, target_kv_cache, target_page_table, target_prefill_lens
            )
            self.metrics["target_time"] += time.time() - target_start

            # Step 3: Accept/reject tokens
            accepted_tokens, rejected_positions = self._process_verification_results(
                active_sequences, draft_tokens, verification_results
            )

            # Step 4: Update generation state
            self._update_generation_state(
                active_sequences, accepted_tokens, rejected_positions, generated_tokens_per_batch, current_positions
            )

            # Step 5: Handle rejected sequences
            if rejected_positions:
                self._handle_rejected_sequences(
                    rejected_positions,
                    active_sequences,
                    draft_kv_cache,
                    target_kv_cache,
                    draft_page_table,
                    target_page_table,
                    draft_prefill_lens,
                    target_prefill_lens,
                )

        self.metrics["total_time"] = time.time() - start_time

        # Decode final outputs
        outputs = []
        for user in range(batch_size):
            output_text = self.target_api.tokenizer.decode(generated_tokens_per_batch[user], skip_special_tokens=True)
            outputs.append(output_text)

        return {
            "outputs": outputs,
            "metrics": self._calculate_metrics(),
            "generated_tokens": generated_tokens_per_batch,
            "num_prompts": len(input_prompts),
            "timing": {
                "total_time": self.metrics["total_time"],
                "draft_time": self.metrics["draft_time"],
                "target_time": self.metrics["target_time"],
            },
        }

    def cleanup(self):
        """Clean up resources."""
        if self.draft_api:
            self.draft_api.cleanup()
        if self.target_api:
            self.target_api.cleanup()

    def _generate_draft_tokens(
        self, active_sequences: List[int], n_tokens: int, kv_cache, page_table, prefill_lens: List[int]
    ) -> List[List[int]]:
        """Generate N tokens using the draft model."""
        if not active_sequences:
            return []

        draft_tokens = []
        current_pos = torch.tensor([prefill_lens[i] for i in active_sequences])

        for _ in range(n_tokens):
            # Use the last generated token as input
            if draft_tokens:
                input_tokens = torch.tensor([[tokens[-1]] for tokens in draft_tokens])
            else:
                # For first iteration, use the last token from prefill
                input_tokens = torch.tensor([[0]] * len(active_sequences))

            # Forward pass through draft model
            output_logits = self.draft_api.generator.decode_forward_text(
                input_tokens,
                current_pos,
                page_table=None,  # No paged attention
                kv_cache=self.draft_api.tt_kv_cache,
                enable_trace=False,
                sampling_params=self.config.sampling_params,
            )

            # Greedy decoding
            next_tokens = torch.argmax(output_logits, dim=-1)

            # Store tokens
            if not draft_tokens:
                draft_tokens = [[token.item()] for token in next_tokens]
            else:
                for i, token in enumerate(next_tokens):
                    draft_tokens[i].append(token.item())

            current_pos += 1
            self.metrics["draft_tokens_generated"] += len(active_sequences)

        return draft_tokens

    def _verify_draft_tokens(
        self, active_sequences: List[int], draft_tokens: List[List[int]], kv_cache, page_table, prefill_lens: List[int]
    ) -> List[float]:
        """Verify draft tokens using the target model."""
        if not draft_tokens:
            return []

        # Create input sequence for verification
        max_len = max(len(tokens) for tokens in draft_tokens)
        batch_size = len(active_sequences)

        # Pad sequences to same length
        padded_tokens = []
        for tokens in draft_tokens:
            padded = tokens + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)

        input_tokens = torch.tensor(padded_tokens)
        current_pos = torch.tensor([prefill_lens[i] for i in active_sequences])

        # Forward pass through target model
        output_logits = self.target_api.generator.decode_forward_text(
            input_tokens,
            current_pos,
            page_table=None,  # No paged attention
            kv_cache=self.target_api.tt_kv_cache,
            enable_trace=False,
            sampling_params=self.config.sampling_params,
        )

        self.metrics["target_forward_passes"] += 1

        # Calculate acceptance probabilities
        acceptance_probs = []
        for i, draft_seq in enumerate(draft_tokens):
            seq_probs = []
            for j, draft_token in enumerate(draft_seq):
                if j < len(output_logits[i]):
                    target_probs = torch.softmax(output_logits[i][j], dim=-1)
                    draft_prob = target_probs[draft_token].item()
                    seq_probs.append(draft_prob)
            acceptance_probs.append(seq_probs)

        return acceptance_probs

    def _process_verification_results(
        self, active_sequences: List[int], draft_tokens: List[List[int]], verification_results: List[List[float]]
    ) -> Tuple[List[List[int]], List[int]]:
        """Process verification results and determine accepted/rejected tokens."""
        accepted_tokens = []
        rejected_positions = []

        for i, (draft_seq, verif_probs) in enumerate(zip(draft_tokens, verification_results)):
            accepted_seq = []
            rejected = False

            for j, (token, prob) in enumerate(zip(draft_seq, verif_probs)):
                if not rejected and prob > 0.5:  # Simple acceptance threshold
                    accepted_seq.append(token)
                    self.metrics["accepted_tokens"] += 1
                else:
                    rejected = True
                    rejected_positions.append(i)
                    break

            accepted_tokens.append(accepted_seq)
            self.metrics["rejected_tokens"] += len(draft_seq) - len(accepted_seq)

        return accepted_tokens, rejected_positions

    def _update_generation_state(
        self,
        active_sequences: List[int],
        accepted_tokens: List[List[int]],
        rejected_positions: List[int],
        generated_tokens_per_batch: List[List[int]],
        current_positions: List[int],
    ):
        """Update the generation state with accepted tokens."""
        for i, accepted_seq in enumerate(accepted_tokens):
            if i < len(active_sequences):
                seq_idx = active_sequences[i]
                generated_tokens_per_batch[seq_idx].extend(accepted_seq)
                current_positions[seq_idx] += len(accepted_seq)
                self.metrics["total_tokens_generated"] += len(accepted_seq)

    def _handle_rejected_sequences(
        self,
        rejected_positions: List[int],
        active_sequences: List[int],
        draft_kv_cache,
        target_kv_cache,
        draft_page_table,
        target_page_table,
        draft_prefill_lens: List[int],
        target_prefill_lens: List[int],
    ):
        """Handle sequences that were rejected during verification."""
        # For simplicity, we'll just remove rejected sequences from active list
        # In a more sophisticated implementation, you might want to retry with different tokens
        for pos in reversed(rejected_positions):
            if pos < len(active_sequences):
                del active_sequences[pos]

    def _calculate_metrics(self) -> Dict:
        """Calculate final performance metrics."""
        total_tokens = self.metrics["total_tokens_generated"]
        accepted_tokens = self.metrics["accepted_tokens"]
        rejected_tokens = self.metrics["rejected_tokens"]

        acceptance_rate = (
            accepted_tokens / (accepted_tokens + rejected_tokens) if (accepted_tokens + rejected_tokens) > 0 else 0
        )
        tokens_per_second = total_tokens / self.metrics["total_time"] if self.metrics["total_time"] > 0 else 0

        return {
            "total_tokens_generated": total_tokens,
            "accepted_tokens": accepted_tokens,
            "rejected_tokens": rejected_tokens,
            "acceptance_rate": acceptance_rate,
            "tokens_per_second": tokens_per_second,
            "total_time": self.metrics["total_time"],
            "draft_time": self.metrics["draft_time"],
            "target_time": self.metrics["target_time"],
            "speculative_rounds": self.metrics["speculative_rounds"],
            "target_forward_passes": self.metrics["target_forward_passes"],
            "draft_tokens_generated": self.metrics["draft_tokens_generated"],
        }


def create_speculative_decoding_api(
    mesh_device: ttnn.MeshDevice, config: Optional[SpeculativeConfig] = None
) -> SpeculativeDecodingAPI:
    """
    Create a speculative decoding API instance.

    Args:
        mesh_device: TT-Metal mesh device
        config: Configuration for speculative decoding

    Returns:
        SpeculativeDecodingAPI instance
    """
    if config is None:
        config = SpeculativeConfig()

    return SpeculativeDecodingAPI(config, mesh_device)


def run_speculative_decoding_demo(
    input_prompts: List[str],
    n_draft_tokens: int = 4,
    max_tokens: int = 100,
    mesh_device: Optional[ttnn.MeshDevice] = None,
) -> Dict:
    """
    Run speculative decoding demo.

    Args:
        input_prompts: List of input prompts
        n_draft_tokens: Number of draft tokens to generate
        max_tokens: Maximum tokens to generate
        mesh_device: TT-Metal mesh device

    Returns:
        Dictionary with results and metrics
    """
    if mesh_device is None:
        # Use the same approach as the existing demo_api
        device_ids = ttnn.get_device_ids()
        mesh_shape = ttnn.MeshShape(1, len(device_ids))
        # Create mesh device with proper parameters
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=mesh_shape, physical_device_ids=device_ids, trace_region_size=30000000, num_command_queues=1
        )

    # Create speculative decoding API
    api = create_speculative_decoding_api(mesh_device)

    # Initialize models
    api.initialize_models()

    # Run speculative decoding
    results = api.greedy_speculative_decode(
        input_prompts=input_prompts, max_tokens=max_tokens, n_draft_tokens=n_draft_tokens
    )

    return results


if __name__ == "__main__":
    # Example usage
    input_prompts = ["What is the capital of France?", "Explain quantum computing in simple terms."]

    results = run_speculative_decoding_demo(input_prompts=input_prompts, n_draft_tokens=4, max_tokens=50)

    print("Speculative Decoding Results:")
    print(f"Outputs: {results['outputs']}")
    print(f"Metrics: {results['metrics']}")
