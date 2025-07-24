#!/usr/bin/env python3
"""
Simple speculative decoding performance test.
This test measures performance metrics we can get without hitting core allocation issues.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.tt_transformers.demo.speculative_decoding_api import SpeculativeConfig, create_speculative_decoding_api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the test environment."""
    # Set environment variables
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    os.environ["ARCH_NAME"] = "wormhole_b0"
    os.environ["TT_METAL_HOME"] = str(project_root)
    os.environ["PYTHONPATH"] = str(project_root)
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.2-1B"
    os.environ["CI"] = "true"

    logger.info("Environment variables set:")
    logger.info(f"  - WH_ARCH_YAML: {os.environ.get('WH_ARCH_YAML')}")
    logger.info(f"  - ARCH_NAME: {os.environ.get('ARCH_NAME')}")
    logger.info(f"  - TT_METAL_HOME: {os.environ.get('TT_METAL_HOME')}")
    logger.info(f"  - PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    logger.info(f"  - HF_MODEL: {os.environ.get('HF_MODEL')}")
    logger.info(f"  - CI: {os.environ.get('CI')}")


def test_model_loading_performance():
    """Test model loading performance."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL LOADING PERFORMANCE")
    logger.info("=" * 80)

    import ttnn

    # Create mesh device
    device_id = ttnn.get_device_ids()[0]
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[device_id],
        trace_region_size=30000000,
        num_command_queues=1,
    )

    # Create config
    config = SpeculativeConfig(
        draft_model_name="meta-llama/Llama-3.2-1B",
        target_model_name="meta-llama/Llama-3.2-3B",
        max_seq_len=128,
        max_batch_size=1,
    )

    # Measure API creation time
    start_time = time.time()
    api = create_speculative_decoding_api(mesh_device, config)
    api_creation_time = time.time() - start_time

    logger.info(f"API creation time: {api_creation_time:.2f}s")

    # Measure model initialization time
    start_time = time.time()
    try:
        api.initialize_models()
        model_init_time = time.time() - start_time
        logger.info(f"Model initialization time: {model_init_time:.2f}s")

        # Calculate total setup time
        total_setup_time = api_creation_time + model_init_time
        logger.info(f"Total setup time: {total_setup_time:.2f}s")

        # Log model sizes
        logger.info(f"Draft model: {config.draft_model_name}")
        logger.info(f"Target model: {config.target_model_name}")
        logger.info(f"Max sequence length: {config.max_seq_len}")
        logger.info(f"Max batch size: {config.max_batch_size}")

        # Test basic API functionality
        logger.info("Testing API functionality...")
        logger.info(f"Draft API: {api.draft_api is not None}")
        logger.info(f"Target API: {api.target_api is not None}")

        # Cleanup
        api.cleanup()
        try:
            mesh_device.close()
        except AttributeError:
            # Device doesn't have close method, use ttnn.close_device instead
            ttnn.close_device(mesh_device)

        return {
            "api_creation_time": api_creation_time,
            "model_init_time": model_init_time,
            "total_setup_time": total_setup_time,
            "success": True,
        }

    except Exception as e:
        model_init_time = time.time() - start_time
        logger.error(f"Model initialization failed after {model_init_time:.2f}s: {e}")

        # Cleanup
        try:
            api.cleanup()
        except:
            pass
        try:
            mesh_device.close()
        except AttributeError:
            # Device doesn't have close method, use ttnn.close_device instead
            ttnn.close_device(mesh_device)

        return {
            "api_creation_time": api_creation_time,
            "model_init_time": model_init_time,
            "total_setup_time": api_creation_time + model_init_time,
            "success": False,
            "error": str(e),
        }


def test_multiple_configurations():
    """Test performance with different configurations."""
    logger.info("=" * 80)
    logger.info("TESTING MULTIPLE CONFIGURATIONS")
    logger.info("=" * 80)

    import ttnn

    # Test different configurations
    configs = [
        {
            "name": "Small Config",
            "draft_model": "meta-llama/Llama-3.2-1B",
            "target_model": "meta-llama/Llama-3.2-3B",
            "max_seq_len": 64,
            "max_batch_size": 1,
        },
        {
            "name": "Medium Config",
            "draft_model": "meta-llama/Llama-3.2-1B",
            "target_model": "meta-llama/Llama-3.2-3B",
            "max_seq_len": 128,
            "max_batch_size": 1,
        },
        {
            "name": "Large Config",
            "draft_model": "meta-llama/Llama-3.2-1B",
            "target_model": "meta-llama/Llama-3.2-3B",
            "max_seq_len": 256,
            "max_batch_size": 1,
        },
    ]

    results = []

    for config_info in configs:
        logger.info(f"Testing {config_info['name']}...")

        # Create mesh device
        device_id = ttnn.get_device_ids()[0]
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[device_id],
            trace_region_size=30000000,
            num_command_queues=1,
        )

        # Create config
        config = SpeculativeConfig(
            draft_model_name=config_info["draft_model"],
            target_model_name=config_info["target_model"],
            max_seq_len=config_info["max_seq_len"],
            max_batch_size=config_info["max_batch_size"],
        )

        # Measure setup time
        start_time = time.time()
        api = create_speculative_decoding_api(mesh_device, config)
        api_creation_time = time.time() - start_time

        start_time = time.time()
        try:
            api.initialize_models()
            model_init_time = time.time() - start_time
            success = True
        except Exception as e:
            model_init_time = time.time() - start_time
            success = False
            error = str(e)

        # Cleanup
        try:
            api.cleanup()
        except:
            pass
        mesh_device.close()

        result = {
            "config_name": config_info["name"],
            "api_creation_time": api_creation_time,
            "model_init_time": model_init_time,
            "total_setup_time": api_creation_time + model_init_time,
            "success": success,
        }

        if not success:
            result["error"] = error

        results.append(result)

        logger.info(f"  - API creation: {api_creation_time:.2f}s")
        logger.info(f"  - Model init: {model_init_time:.2f}s")
        logger.info(f"  - Total setup: {result['total_setup_time']:.2f}s")
        logger.info(f"  - Success: {success}")

    return results


def test_memory_usage():
    """Test memory usage patterns."""
    logger.info("=" * 80)
    logger.info("TESTING MEMORY USAGE")
    logger.info("=" * 80)

    import psutil

    import ttnn

    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Create mesh device
    device_id = ttnn.get_device_ids()[0]
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[device_id],
        trace_region_size=30000000,
        num_command_queues=1,
    )

    # Create config
    config = SpeculativeConfig(
        draft_model_name="meta-llama/Llama-3.2-1B",
        target_model_name="meta-llama/Llama-3.2-3B",
        max_seq_len=128,
        max_batch_size=1,
    )

    # Measure memory after API creation
    api = create_speculative_decoding_api(mesh_device, config)
    memory_after_api = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory after API creation: {memory_after_api:.2f} MB")
    logger.info(f"Memory increase from API: {memory_after_api - initial_memory:.2f} MB")

    # Measure memory after model initialization
    try:
        api.initialize_models()
        memory_after_init = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after model init: {memory_after_init:.2f} MB")
        logger.info(f"Memory increase from models: {memory_after_init - memory_after_api:.2f} MB")
        logger.info(f"Total memory increase: {memory_after_init - initial_memory:.2f} MB")

        success = True
    except Exception as e:
        memory_after_init = process.memory_info().rss / 1024 / 1024
        logger.error(f"Model initialization failed: {e}")
        logger.info(f"Memory at failure: {memory_after_init:.2f} MB")
        success = False

    # Cleanup
    try:
        api.cleanup()
    except:
        pass
    mesh_device.close()

    # Measure memory after cleanup
    memory_after_cleanup = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory after cleanup: {memory_after_cleanup:.2f} MB")
    logger.info(f"Memory leak: {memory_after_cleanup - initial_memory:.2f} MB")

    return {
        "initial_memory": initial_memory,
        "memory_after_api": memory_after_api,
        "memory_after_init": memory_after_init,
        "memory_after_cleanup": memory_after_cleanup,
        "api_memory_increase": memory_after_api - initial_memory,
        "model_memory_increase": memory_after_init - memory_after_api,
        "total_memory_increase": memory_after_init - initial_memory,
        "memory_leak": memory_after_cleanup - initial_memory,
        "success": success,
    }


def main():
    """Main performance test function."""
    logger.info("Starting speculative decoding performance tests...")

    # Set up environment
    setup_environment()

    # Run tests
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE TEST RESULTS")
    logger.info("=" * 80)

    # Test 1: Model loading performance
    logger.info("\n1. MODEL LOADING PERFORMANCE")
    loading_result = test_model_loading_performance()

    # Test 2: Multiple configurations
    logger.info("\n2. MULTIPLE CONFIGURATIONS")
    config_results = test_multiple_configurations()

    # Test 3: Memory usage
    logger.info("\n3. MEMORY USAGE")
    memory_result = test_memory_usage()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    logger.info(f"Model Loading Performance:")
    logger.info(f"  - API creation: {loading_result['api_creation_time']:.2f}s")
    logger.info(f"  - Model initialization: {loading_result['model_init_time']:.2f}s")
    logger.info(f"  - Total setup: {loading_result['total_setup_time']:.2f}s")
    logger.info(f"  - Success: {loading_result['success']}")

    logger.info(f"\nConfiguration Performance:")
    for result in config_results:
        logger.info(
            f"  - {result['config_name']}: {result['total_setup_time']:.2f}s ({'SUCCESS' if result['success'] else 'FAILED'})"
        )

    logger.info(f"\nMemory Usage:")
    logger.info(f"  - Initial memory: {memory_result['initial_memory']:.2f} MB")
    logger.info(f"  - API memory increase: {memory_result['api_memory_increase']:.2f} MB")
    logger.info(f"  - Model memory increase: {memory_result['model_memory_increase']:.2f} MB")
    logger.info(f"  - Total memory increase: {memory_result['total_memory_increase']:.2f} MB")
    logger.info(f"  - Memory leak: {memory_result['memory_leak']:.2f} MB")

    # Calculate averages
    successful_configs = [r for r in config_results if r["success"]]
    if successful_configs:
        avg_setup_time = sum(r["total_setup_time"] for r in successful_configs) / len(successful_configs)
        logger.info(f"\nAverage setup time (successful configs): {avg_setup_time:.2f}s")

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
