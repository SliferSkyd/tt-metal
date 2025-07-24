#!/usr/bin/env python3
"""
Test runner for real speculative decoding pytest tests.
This script runs actual speculative decoding on TT hardware.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the test environment."""
    # Set environment variables
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    os.environ["ARCH_NAME"] = "wormhole_b0"
    os.environ["TT_METAL_HOME"] = str(Path(__file__).parent.parent.parent.parent)
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent)
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.2-1B"
    os.environ["CI"] = "true"

    logger.info("Environment variables set:")
    logger.info(f"  - WH_ARCH_YAML: {os.environ.get('WH_ARCH_YAML')}")
    logger.info(f"  - ARCH_NAME: {os.environ.get('ARCH_NAME')}")
    logger.info(f"  - TT_METAL_HOME: {os.environ.get('TT_METAL_HOME')}")
    logger.info(f"  - PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    logger.info(f"  - HF_MODEL: {os.environ.get('HF_MODEL')}")
    logger.info(f"  - CI: {os.environ.get('CI')}")


def run_basic_test():
    """Run basic speculative decoding test."""
    logger.info("=" * 80)
    logger.info("RUNNING BASIC SPECULATIVE DECODING TEST")
    logger.info("=" * 80)

    # Test file path
    test_file = Path(__file__).parent / "test_speculative_decoding_real.py"

    # Run pytest for basic test
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-k",
        "test_speculative_decoding_basic",
        "-v",
        "--tb=short",
        "--disable-warnings",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        logger.info("=" * 80)
        logger.info("BASIC TEST RESULTS")
        logger.info("=" * 80)

        if result.stdout:
            logger.info("STDOUT:")
            logger.info(result.stdout)

        if result.stderr:
            logger.info("STDERR:")
            logger.info(result.stderr)

        logger.info(f"Return code: {result.returncode}")

        if result.returncode == 0:
            logger.info("✅ Basic speculative decoding test passed!")
        else:
            logger.info("❌ Basic speculative decoding test failed!")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("❌ Test timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running test: {e}")
        return False


def run_all_integration_tests():
    """Run all integration tests for speculative decoding."""
    logger.info("=" * 80)
    logger.info("RUNNING ALL SPECULATIVE DECODING INTEGRATION TESTS")
    logger.info("=" * 80)

    # Test file path
    test_file = Path(__file__).parent / "test_speculative_decoding_real.py"

    # Run pytest for all integration tests
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-m",
        "integration",
        "-v",
        "--tb=short",
        "--disable-warnings",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        logger.info("=" * 80)
        logger.info("INTEGRATION TEST RESULTS")
        logger.info("=" * 80)

        if result.stdout:
            logger.info("STDOUT:")
            logger.info(result.stdout)

        if result.stderr:
            logger.info("STDERR:")
            logger.info(result.stderr)

        logger.info(f"Return code: {result.returncode}")

        if result.returncode == 0:
            logger.info("✅ All integration tests passed!")
        else:
            logger.info("❌ Some integration tests failed!")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("❌ Tests timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False


def run_performance_test():
    """Run performance test for speculative decoding."""
    logger.info("=" * 80)
    logger.info("RUNNING SPECULATIVE DECODING PERFORMANCE TEST")
    logger.info("=" * 80)

    # Test file path
    test_file = Path(__file__).parent / "test_speculative_decoding_real.py"

    # Run pytest for performance test
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-k",
        "test_speculative_decoding_performance",
        "-v",
        "--tb=short",
        "--disable-warnings",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

        logger.info("=" * 80)
        logger.info("PERFORMANCE TEST RESULTS")
        logger.info("=" * 80)

        if result.stdout:
            logger.info("STDOUT:")
            logger.info(result.stdout)

        if result.stderr:
            logger.info("STDERR:")
            logger.info(result.stderr)

        logger.info(f"Return code: {result.returncode}")

        if result.returncode == 0:
            logger.info("✅ Performance test passed!")
        else:
            logger.info("❌ Performance test failed!")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("❌ Test timed out after 15 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running test: {e}")
        return False


def main():
    """Main function to run tests."""
    logger.info("Starting real speculative decoding tests...")

    # Set up environment
    setup_environment()

    # Parse command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "basic":
            success = run_basic_test()
        elif test_type == "performance":
            success = run_performance_test()
        elif test_type == "all":
            success = run_all_integration_tests()
        else:
            logger.error(f"Unknown test type: {test_type}")
            logger.info("Available options: basic, performance, all")
            sys.exit(1)
    else:
        # Default to running basic test
        success = run_basic_test()

    # Exit with appropriate code
    if success:
        logger.info("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
