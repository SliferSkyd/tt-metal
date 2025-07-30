# Speculative Decoding for Tenstorrent Hardware

This implementation provides speculative decoding for LLM inference on Tenstorrent hardware. Speculative decoding uses a smaller "draft" model to generate multiple tokens quickly, then verifies them with a larger "target" model for improved inference speed.

## Key Components

### 1. SpeculativeGenerator (`speculative_generator.py`)
- Manages both draft and target models
- Handles prefill for both models
- Implements the speculative decoding algorithm
- Provides token verification and acceptance logic

### 2. Demo Script (`speculative_demo.py`)
- Main demo script similar to `simple_text_demo.py`
- Supports pytest for testing and benchmarking
- Provides performance metrics and acceptance rate tracking

### 3. Sample Prompts (`sample_prompts.json`)
- Example prompts for testing the implementation

## Usage

### Basic Usage

Run the speculative decoding demo with default parameters:

```bash
python models/tt_transformers/demo/speculative_decoding/speculative_demo.py
```

### Using pytest

Run with specific configurations:

```bash
pytest models/tt_transformers/demo/speculative_decoding/speculative_demo.py::test_speculative_demo -v
```

### Command Line Parameters

You can override default parameters:

```bash
pytest models/tt_transformers/demo/speculative_decoding/speculative_demo.py::test_speculative_demo \
    --input_prompts "models/tt_transformers/demo/speculative_decoding/sample_prompts.json" \
    --max_generated_tokens 100 \
    --n_draft_tokens 4 \
    --draft_layers 16 \
    --target_layers 32 \
    -v
```

### Available Parameters

- `--input_prompts`: Path to JSON file with input prompts
- `--instruct`: Use instruct mode (0 or 1)
- `--max_seq_len`: Maximum sequence length
- `--batch_size`: Batch size for inference
- `--max_generated_tokens`: Maximum tokens to generate
- `--n_draft_tokens`: Number of draft tokens to generate per step
- `--draft_layers`: Number of layers for the draft model
- `--target_layers`: Number of layers for the target model

## How Speculative Decoding Works

1. **Model Setup**: Creates two model instances:
   - Draft model: Smaller/faster (fewer layers)
   - Target model: Larger/more accurate (more layers)

2. **Prefill Phase**: Both models process the input prompt

3. **Speculative Decoding Loop**:
   - Draft model generates `n_draft_tokens` (default: 4) tokens quickly
   - Target model verifies each draft token
   - Accepted tokens are kept, rejected tokens trigger fallback to target prediction
   - Process continues until max tokens reached or EOS token

4. **Benefits**:
   - When draft tokens are accepted, multiple tokens are generated in one step
   - Effective speedup depends on the acceptance rate
   - Higher acceptance rates lead to better performance improvements

## Configuration Examples

### Small vs. Large Model
```python
# Use 8-layer draft model and 32-layer target model
draft_config = {"max_batch_size": 1, "num_layers": 8}
target_config = {"max_batch_size": 1, "num_layers": 32}
```

### Aggressive Speculation
```python
# Generate more draft tokens per step
n_draft_tokens = 8
```

### Conservative Speculation
```python
# Generate fewer draft tokens per step for higher acceptance rate
n_draft_tokens = 2
```

## Performance Metrics

The demo provides several key metrics:

- **Acceptance Rate**: Percentage of draft tokens accepted by target model
- **Effective Speedup**: Average tokens generated per decode step
- **Total Runtime**: End-to-end execution time
- **Token Generation Rate**: Effective tokens per second

### Expected Results

With well-matched draft and target models, you should see:
- Acceptance rates of 50-80%
- Effective speedup of 2-4x for token generation
- Overall inference speedup (depends on hardware utilization)

## Model Compatibility

The implementation works with:
- Llama models (3.1-8B, 3.1-70B, 3.2-1B, 3.2-3B, etc.)
- Mistral models
- Any model supported by the base TT-transformers framework

The draft and target models should be:
- The same model architecture (e.g., both Llama)
- Different sizes (draft < target for optimal performance)
- Use the same tokenizer

## Environment Setup

Ensure you have:
- TT-Metal environment configured
- Required model weights available
- Environment variables set (LLAMA_DIR or HF_MODEL)

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure both draft and target model sizes are supported
2. **Memory Issues**: Reduce batch size or sequence length
3. **Low Acceptance Rates**: Try using more similar model sizes
4. **Performance Issues**: Check device utilization and tracing settings

### Debug Tips

- Set log level to DEBUG for detailed token acceptance information
- Monitor acceptance rates - consistently low rates indicate model mismatch
- Compare with baseline non-speculative performance

## Example Output

```
INFO     | Speculative Decoding Configuration:
INFO     |   Draft model layers: 16
INFO     |   Target model layers: 32
INFO     |   Draft tokens per step: 4
INFO     |   Max sequence length: 1024
INFO     |   Batch size: 1

INFO     | Iteration 0: 150ms @ 2.7 tok/s/user (2.7 tok/s throughput), accepted 4/4 draft tokens
INFO     | Iteration 1: 120ms @ 3.3 tok/s/user (3.3 tok/s throughput), accepted 3/4 draft tokens

INFO     | === Speculative Decoding Performance ===
INFO     | Draft token acceptance rate: 75.00%
INFO     | Average tokens per decode step: 3.50
INFO     | Average effective token rate: 15.2 tokens/s
```

This implementation demonstrates the potential of speculative decoding on Tenstorrent hardware, showing how smaller draft models can accelerate inference from larger target models through intelligent token speculation and verification.
