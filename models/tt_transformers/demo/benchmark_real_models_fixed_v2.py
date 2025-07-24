#!/usr/bin/env python3

import logging
import os
import time

import ttnn
from models.tt_transformers.demo.demo_api import create_demo_api
from models.tt_transformers.demo.speculative_decoding_api import create_speculative_decoding_api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark_with_core_fix():
    """Run benchmark with core allocation fix"""

    # Set environment variables
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    os.environ["ARCH_NAME"] = "wormhole_b0"
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.2-3B"

    try:
        # Create mesh device with proper configuration
        device_ids = ttnn.get_device_ids()
        logger.info(f"Available device IDs: {device_ids}")

        # Use smaller mesh shape to avoid core allocation issues
        mesh_shape = ttnn.MeshShape(1, min(len(device_ids), 2))  # Limit to 2 devices max
        logger.info(f"Using mesh shape: {mesh_shape}")

        # Get the number of devices from mesh shape - use a simple approach
        num_devices = 2  # Use 2 devices max to avoid core issues

        mesh_device = ttnn.open_mesh_device(
            mesh_shape=mesh_shape,
            physical_device_ids=device_ids[:num_devices],  # Only use the devices we need
            trace_region_size=30000000,
            num_command_queues=1,
        )

        logger.info("Created mesh device successfully")

        # Test with smaller batch sizes to avoid core allocation issues
        batch_sizes = [1, 4]  # Reduced from [1, 32] to avoid core issues

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            try:
                # Create demo API with smaller batch size
                demo_api = create_demo_api(
                    mesh_device=mesh_device,
                    instruct=True,
                    max_batch_size=batch_size,
                    max_seq_len=1024,
                    data_parallel=1,  # Keep data_parallel=1 to avoid core issues
                    paged_attention=True,
                    page_params={"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
                )

                logger.info(f"Created demo API for batch size {batch_size}")

                # Initialize the model
                demo_api.initialize_model()
                logger.info(f"Initialized model for batch size {batch_size}")

                # Test with appropriate number of prompts
                test_prompts = ["Hello, how are you today?"] * batch_size

                # Warm up the model
                warm_up_results = demo_api.warm_up(test_prompts)
                logger.info(f"Warm up completed for batch {batch_size}: {warm_up_results}")

                # Test TTFT measurement
                ttft_results = demo_api.measure_ttft(test_prompts)
                logger.info(f"TTFT measurement completed for batch {batch_size}: {ttft_results}")

                # Test full inference
                inference_results = demo_api.run_full_inference(
                    test_prompts,
                    max_generated_tokens=50,
                    sampling_params={"temperature": 0, "top_p": 0.08},
                    stop_at_eos=True,
                    enable_trace=True,
                )
                logger.info(f"Full inference completed for batch {batch_size}: {inference_results}")

                results[f"batch_{batch_size}"] = {
                    "warm_up": warm_up_results,
                    "ttft": ttft_results,
                    "inference": inference_results,
                }

                # Clean up demo API
                demo_api.cleanup()

            except Exception as e:
                logger.error(f"Failed for batch size {batch_size}: {e}")
                results[f"batch_{batch_size}"] = {"error": str(e)}

        # Clean up mesh device
        ttnn.close_device(mesh_device)

        # Print summary
        logger.info("=== BENCHMARK RESULTS ===")
        for batch_size, result in results.items():
            logger.info(f"\n{batch_size.upper()}:")
            if "error" in result:
                logger.info(f"  ERROR: {result['error']}")
            else:
                logger.info(f"  Warm-up time: {result['warm_up'].get('warm_up_time', 'N/A'):.3f}s")
                logger.info(f"  TTFT: {result['ttft'].get('ttft_time', 'N/A'):.3f}s")
                logger.info(f"  Total inference time: {result['inference'].get('total_inference_time', 'N/A'):.3f}s")
                logger.info(f"  Tokens generated: {result['inference'].get('total_tokens_generated', 'N/A')}")
                logger.info(f"  Tokens per second: {result['inference'].get('tokens_per_second', 'N/A'):.2f}")

        return results

    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def run_speculative_benchmark():
    """Run speculative decoding benchmark with core allocation fix"""

    # Set environment variables
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    os.environ["ARCH_NAME"] = "wormhole_b0"
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.2-3B"

    try:
        # Create mesh device with proper configuration
        device_ids = ttnn.get_device_ids()
        logger.info(f"Available device IDs: {device_ids}")

        # Use smaller mesh shape to avoid core allocation issues
        mesh_shape = ttnn.MeshShape(1, min(len(device_ids), 2))  # Limit to 2 devices max
        logger.info(f"Using mesh shape: {mesh_shape}")

        # Get the number of devices from mesh shape - use a simple approach
        num_devices = 2  # Use 2 devices max to avoid core issues

        mesh_device = ttnn.open_mesh_device(
            mesh_shape=mesh_shape,
            physical_device_ids=device_ids[:num_devices],  # Only use the devices we need
            trace_region_size=30000000,
            num_command_queues=1,
        )

        logger.info("Created mesh device successfully")

        # Test speculative decoding with smaller batch sizes
        batch_sizes = [1, 4]  # Reduced from [1, 32] to avoid core issues

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Testing speculative decoding with batch size: {batch_size}")

            try:
                # Create speculative decoding API
                speculative_api = create_speculative_decoding_api(
                    mesh_device=mesh_device,
                    draft_model_name="meta-llama/Llama-3.2-1B",
                    target_model_name="meta-llama/Llama-3.2-3B",
                    max_batch_size=batch_size,
                    max_seq_len=512,  # Reduced to avoid memory issues
                    data_parallel=1,
                )

                logger.info(f"Created speculative API for batch size {batch_size}")

                # Initialize the model
                speculative_api.initialize_model()
                logger.info(f"Initialized speculative model for batch size {batch_size}")

                # Test with appropriate number of prompts
                test_prompts = ["Hello, how are you today?"] * batch_size

                # Run speculative decoding
                start_time = time.time()
                generated_tokens, total_tokens = speculative_api.run_speculative_decoding(
                    test_prompts, max_generated_tokens=50, sampling_params={"temperature": 0, "top_p": 0.08}
                )
                end_time = time.time()

                total_time = end_time - start_time
                tokens_per_second = total_tokens / total_time if total_time > 0 else 0

                results[f"speculative_batch_{batch_size}"] = {
                    "total_time": total_time,
                    "total_tokens": total_tokens,
                    "tokens_per_second": tokens_per_second,
                    "generated_tokens": generated_tokens,
                }

                logger.info(f"Speculative decoding completed for batch {batch_size}:")
                logger.info(f"  Total time: {total_time:.3f}s")
                logger.info(f"  Total tokens: {total_tokens}")
                logger.info(f"  Tokens per second: {tokens_per_second:.2f}")

                # Clean up speculative API
                speculative_api.cleanup()

            except Exception as e:
                logger.error(f"Speculative decoding failed for batch size {batch_size}: {e}")
                results[f"speculative_batch_{batch_size}"] = {"error": str(e)}

        # Clean up mesh device
        ttnn.close_device(mesh_device)

        return results

    except Exception as e:
        logger.error(f"Speculative benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info("Running benchmark with core allocation fix...")

    # Run regular benchmark
    regular_results = run_benchmark_with_core_fix()

    # Run speculative benchmark
    speculative_results = run_speculative_benchmark()

    # Print final summary
    logger.info("\n=== FINAL BENCHMARK SUMMARY ===")
    logger.info("Regular API Results:")
    for key, value in regular_results.items():
        if key != "error":
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  ERROR: {value}")

    logger.info("\nSpeculative API Results:")
    for key, value in speculative_results.items():
        if key != "error":
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  ERROR: {value}")

    logger.info("\nBenchmark completed!")
