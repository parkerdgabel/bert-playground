# CLAUDE.md - MLX ModernBERT Project Guide

This file provides guidance to Claude Code (claude.ai/code) when working with this MLX-based ModernBERT implementation for Kaggle competitions.

## Project Overview

This project implements ModernBERT using Apple's MLX framework for efficient training on Apple Silicon. The goal is to solve Kaggle tabular problems using text-based approaches with state-of-the-art language models.

### Key Technologies
- **MLX**: Apple's machine learning framework optimized for Apple Silicon
- **MLX-Data**: Efficient data loading library for MLX
- **MLX-LM**: Language modeling utilities for MLX
- **MLX-Embeddings**: Native MLX implementation of BERT models with 4-bit quantization
- **ModernBERT**: State-of-the-art BERT variant from Answer.AI
- **uv**: Fast Python package and project manager
- **MLflow**: Experiment tracking and model management
- **Loguru**: Structured logging

## Quick Start Commands

### Core CLI Commands

```bash
# Install dependencies
uv sync

# Quick training test (small batch size for testing)
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --epochs 1 \
    --batch-size 4 \
    --logging-steps 1

# Standard training (recommended batch size for performance)
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5

# Production training with config file
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/production.json

# Training with pre-tokenization (recommended for performance)
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --use-pretokenized \
    --batch-size 64

# Generate predictions
uv run python bert_cli.py predict \
    --test data/titanic/test.csv \
    --checkpoint output/run_*/checkpoints/final \
    --output submission.csv

# Benchmark performance
uv run python bert_cli.py benchmark \
    --batch-size 64 \
    --seq-length 256 \
    --steps 20

# System info
uv run python bert_cli.py info
```

### Advanced Commands (Future Implementation)

The following command groups are planned for future implementation:

- **Kaggle Integration**: Competition downloads, submissions, leaderboard tracking
- **MLflow Management**: Experiment tracking, model registry, metrics visualization
- **Model Serving**: REST API endpoints, ONNX export, model inspection
- **AutoML Features**: Hyperparameter tuning, architecture search

### Training Tips

```bash
# Background training with logging
nohup uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --batch-size 32 \
    --epochs 5 \
    > training.log 2>&1 &

# Monitor training progress
tail -f training.log

# Training with optimal settings for M1/M2 Mac
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --batch-size 32 \
    --lr 2e-5 \
    --epochs 5 \
    --num-workers 4 \
    --prefetch-size 4
```

## Project Structure

```
bert-playground/
├── cli/                     # Modern CLI interface using Typer
│   ├── app.py              # Main CLI application
│   ├── commands/           # Command implementations
│   │   ├── core/          # Core commands (train, predict, benchmark)
│   │   ├── kaggle/        # Kaggle integration commands
│   │   ├── mlflow/        # MLflow management commands
│   │   └── model/         # Model management commands
│   └── utils/             # CLI utilities and helpers
├── models/                  # Model implementations
│   ├── bert/              # BERT architectures
│   │   ├── core.py        # Classic BERT implementation
│   │   ├── modernbert_config.py  # ModernBERT configuration
│   │   └── layers/        # Attention, embeddings, feedforward
│   ├── heads/             # Task-specific heads
│   │   ├── classification.py  # Binary/multiclass/multilabel heads
│   │   └── regression.py      # Regression/ordinal heads
│   ├── lora/              # LoRA adapter implementation
│   └── factory.py         # Model creation factory
├── data/                    # Data handling and loading
│   ├── core/              # Base abstractions and protocols
│   ├── loaders/           # MLX, streaming, memory loaders
│   ├── templates/         # Tabular-to-text conversion
│   └── kaggle/            # Competition-specific datasets
├── training/               # Training infrastructure
│   ├── core/              # Base trainer and protocols
│   ├── callbacks/         # Training callbacks
│   ├── metrics/           # Evaluation metrics
│   ├── kaggle/            # Competition-specific training
│   └── integrations/      # MLflow integration
├── configs/                # Configuration files
│   ├── production.json    # Production settings
│   ├── quick.yaml         # Quick test settings
│   └── best_model_only.yaml  # Storage-optimized settings
└── scripts/               # Utility scripts
    └── download_titanic.py  # Data download utilities
```

## Key Components

### 1. Model Architecture (`models/`)
- **Classic BERT**: Full implementation with multi-head attention
- **ModernBERT**: Answer.AI's 2024 architecture with RoPE, GeGLU, and alternating attention
- **Task Heads**: 6 specialized heads for classification and regression tasks
- **LoRA Adapters**: Efficient fine-tuning with low-rank adaptation
- **Factory Pattern**: Unified model creation with presets

### 2. Data Pipeline (`data/`)
- **Protocol-Based Design**: Flexible interfaces for datasets and loaders
- **MLX-Optimized Loader**: Zero-copy operations with unified memory
- **Streaming Pipeline**: 1000+ samples/second for large datasets
- **Text Templates**: Convert tabular data to natural language
- **Dataset Registry**: Centralized management of competitions

### 3. Training Infrastructure (`training/`)
- **Declarative Configuration**: YAML/JSON-based training setup
- **Callback System**: Extensible hooks for training events
- **MLflow Integration**: Automatic experiment tracking
- **Kaggle Features**: Cross-validation, ensembling, auto-submission
- **State Management**: Comprehensive checkpointing and resume

### 4. CLI Interface (`cli/`)
- **Hierarchical Commands**: Organized command structure with Typer
- **Rich Output**: Beautiful console formatting with progress bars
- **Comprehensive Help**: Detailed documentation for all commands
- **Error Handling**: User-friendly error messages and suggestions
- **Validation**: Input validation with helpful callbacks

## MLX Optimization Tips

1. **Batch Size**: Use larger batch sizes (32-64) for better MLX performance
2. **Prefetching**: Enable data prefetching with 4-8 prefetch size
3. **Workers**: Use 4-8 workers for data loading
4. **Gradient Accumulation**: Use when memory is limited
5. **Mixed Precision**: MLX handles this automatically
6. **MLX Compilation**: Enable with `use_compilation: true` in config (default)
   - Best speedup with larger models and batch sizes
   - Test with `bert benchmark --test-compilation`
   - Compilation automatically handles dropout and random state
7. **LoRA Training**: Use LoRA adapters for efficient fine-tuning
   - Significantly reduces memory usage and training time
   - Compiled evaluation is automatically disabled for LoRA to handle train/eval mode switches
   - LoRA configs are available in `configs/` directory

## Checkpoint Management

### Best Model Only Mode
To save storage space, you can configure training to save only the best model instead of regular checkpoints:

```yaml
checkpoint:
  enable_checkpointing: true
  save_best_model: true
  save_best_only: true  # Skip regular checkpoints
  best_model_metric: "val_accuracy"
  best_model_mode: "max"
```

**Benefits:**
- Reduces storage usage significantly
- Focuses on keeping only the highest-performing model
- Still logs models to MLflow for experiment tracking

**Usage:**
```bash
# Use pre-configured best model only config
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/best_model_only.yaml
```

**Note:** Regular checkpoints are still saved during training when `save_best_only: false` (default).

## Common Issues and Solutions

### Issue: Training Appears to Hang
**Symptom**: Training shows "Starting training..." but no progress updates, or hangs at "Epoch 0 - Batch 0/X"
**Root Cause**: MLX lazy evaluation can build up large computation graphs without immediate evaluation
**Solution**: This issue has been fixed by adding `mx.eval()` calls after gradient computation. Additional debugging options:
```bash
# Use --logging-steps to see more frequent updates
uv run python bert_cli.py train --logging-steps 1 --train data/titanic/train.csv

# Monitor actual progress by checking output files
ls -la output/run_*/

# If compilation issues persist, disable compilation
uv run python bert_cli.py train --config configs/no_compile.yaml
```

**Note**: The training hang issue was resolved in recent updates by ensuring gradients are evaluated immediately after computation, preventing lazy evaluation buildup. See `docs/training_hang_fix.md` for detailed technical analysis.

### Issue: Slow Training with Small Batches
**Symptom**: Very slow training with batch sizes < 16
**Solution**: MLX performs better with larger batch sizes
```bash
# Use larger batch size (recommended: 32-64)
uv run python bert_cli.py train --batch-size 32

# If memory constrained, use gradient accumulation
uv run python bert_cli.py train \
    --batch-size 16 \
    --grad-accum 2
```

### Issue: Out of Memory
```bash
# Option 1: Reduce batch size
uv run python bert_cli.py train --batch-size 16

# Option 2: Use pre-tokenization to reduce memory
uv run python bert_cli.py train --use-pretokenized

# Option 3: Use gradient accumulation
uv run python bert_cli.py train \
    --batch-size 8 \
    --grad-accum 4
```

### Issue: Model Creation Errors
**Symptom**: "Unknown model type" or "head_type must be provided"
**Solution**: The CLI now uses modernbert_with_head model type
```bash
# Correct usage (model type is handled internally)
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv
```

### Issue: Predict Command Fails to Load Model
**Symptom**: "No config.json found" or "ValueError: No config.json found in checkpoint"
**Root Cause**: The predict command was looking for incorrect config file names and using wrong loading patterns
**Solution**: This issue has been fixed with auto-detection of model architecture and correct checkpoint loading
```bash
# Predict command now works correctly with any checkpoint
uv run python bert_cli.py predict \
    --test data/titanic/test.csv \
    --checkpoint output/run_*/checkpoints/final \
    --output predictions.csv
```

**Note**: The predict command now automatically detects classic BERT vs ModernBERT architecture from weight keys and uses the correct MLX loading patterns.

## Pre-tokenization Support

The project supports pre-tokenization for improved performance:

### Using Pre-tokenization

```bash
# Train with pre-tokenized data (automatic caching)
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --use-pretokenized

# Pre-tokenized data is cached in safetensors format at:
# data/.tokenizer_cache/{dataset_hash}_{model_name}_{split}.safetensors
```

### Benefits
- **Faster Loading**: Skip tokenization during training
- **Reduced Memory**: Efficient safetensors format
- **Automatic Caching**: Reuse tokenized data across runs
- **Zero-Copy Operations**: Direct MLX array loading

## MLflow Integration

### Enabling MLflow Tracking
```bash
# Train with MLflow tracking
uv run python bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --experiment titanic_experiment \
    --mlflow

# View experiments
mlflow ui --backend-store-uri ./output/mlruns
```

### Compare Runs
- Navigate to http://localhost:5000
- Select your experiment
- Compare metrics across runs

## Development Workflow

1. **Testing Changes**
   ```bash
   # Quick test with minimal data
   uv run python bert_cli.py train --epochs 1 --batch-size 8
   ```

2. **Benchmarking**
   ```bash
   # Test performance improvements
   uv run python bert_cli.py benchmark --steps 50
   ```

3. **Production Training**
   ```bash
   # Use production config
   uv run python bert_cli.py train --config configs/production.json
   ```

## Model Variants

### Available Architectures

1. **Classic BERT** (`bert`)
   - Standard BERT architecture
   - 12 layers, 768 hidden size, 12 attention heads
   - Supports all downstream tasks

2. **ModernBERT** (`modernbert`)
   - Answer.AI's 2024 improvements
   - RoPE embeddings, GeGLU activation
   - 8192 sequence length support
   - Alternating local/global attention

3. **neoBERT** (`neobert`)
   - 250M parameter efficient variant
   - 28 layers, SwiGLU activation
   - 4096 context length

### Task-Specific Heads

- **Binary Classification**: Sigmoid activation, focal loss support
- **Multiclass Classification**: Softmax, label smoothing
- **Multilabel Classification**: Per-label sigmoid, adaptive thresholds
- **Regression**: MSE/MAE/Huber loss options
- **Ordinal Regression**: Cumulative logits approach
- **Time Series Regression**: Multi-step predictions

### LoRA Configuration

LoRA (Low-Rank Adaptation) support is planned for efficient fine-tuning with reduced memory usage.

## Configuration Management

### Configuration Hierarchy
1. **Default values**: Built-in sensible defaults
2. **Config files**: YAML/JSON configuration
3. **Environment variables**: Override specific settings
4. **CLI arguments**: Highest priority

### Available Presets
- `quick`: Fast testing (1 epoch, small batch)
- `development`: Balanced for development
- `production`: Optimized production settings
- `kaggle`: Competition-optimized
- `titanic_competition`: Competition-specific settings for Titanic
- `titanic_lora`: LoRA-optimized training for Titanic
- `no_compile`: Disable compilation for debugging
- `competition_best_only`: Save only best model to reduce storage
- `test`: Minimal configuration for testing

### Custom Configuration
```yaml
# custom_config.yaml
training:
  epochs: 10
  batch_size: 32
  gradient_accumulation_steps: 2
  
optimizer:
  type: adamw
  learning_rate: 2e-5
  weight_decay: 0.01
  
scheduler:
  type: cosine
  warmup_steps: 500
  
data:
  max_length: 256
  num_workers: 8
  prefetch_size: 4
```

## Performance Metrics

Expected performance on M1/M2 Mac:
- Batch size 4: ~30-60 seconds/step (not recommended)
- Batch size 16: ~5-10 seconds/step
- Batch size 32: ~12-15 seconds/step (recommended)
- Batch size 64: ~20-30 seconds/step
- Optimal batch size: 32-64 for best performance/memory trade-off

## Future Improvements

1. **Model Compression**: Implement quantization for faster inference
2. **Multi-GPU**: Support distributed training across multiple devices
3. **ONNX Export**: Export models for deployment
4. **AutoML**: Automatic hyperparameter tuning
5. **More Datasets**: Support for more Kaggle competitions

## Debugging

Enable debug logging:
```bash
export LOGURU_LEVEL=DEBUG
uv run python bert_cli.py train --log-level DEBUG
```

Check MLX device:
```bash
uv run python bert_cli.py info
```

## Important Notes

- Always use `uv run` to execute Python scripts
- MLX works best with batch sizes that are powers of 2 (16, 32, 64)
- **Training hang issue resolved**: Recent fixes prevent MLX lazy evaluation buildup
- **Predict command fixed**: Now correctly loads checkpoints and auto-detects model architecture
- **LoRA improvements**: Compiled evaluation automatically disabled for proper train/eval mode handling
- Batch sizes < 16 will result in very slow training on MLX
- The model uses random initialization (pretrained weights not loaded yet)
- Pre-tokenization is recommended for better performance
- MLflow tracking is optional but recommended for experiments

## Development Best Practices

- **Debug Scripts**: Always place temporary debug scripts in `/tmp` to keep the repository clean
- **Clean Repository**: Strive to leave the repository in a clean, polished state after making changes
- **Commit Hygiene**: Make focused commits with clear messages that explain the changes
- **Testing is Mandatory**: ALWAYS run tests after making changes and ALWAYS add corresponding tests for new features or fixes
- **Test Structure**: Follow the existing test structure in the `tests/` directory:
  - `unit/`: Unit tests for individual components
  - `integration/`: Integration tests for component interactions
  - `e2e/`: End-to-end tests for complete workflows
  - `fixtures/`: Shared test fixtures and utilities
  - Add tests that mirror the structure of the code being tested

### Testing Commands

```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/data/unit/loaders/test_mlx_loader.py

# Run tests with coverage
uv run pytest --cov=bert_playground --cov-report=html

# Run tests in verbose mode
uv run pytest -v

# Run only unit tests
uv run pytest tests/*/unit/

# Run tests matching a pattern
uv run pytest -k "test_mlx"
```

### Example Test Structure

When adding a new feature in `data/loaders/mlx_loader.py`, create corresponding tests:
- `tests/data/unit/loaders/test_mlx_loader.py` - Unit tests for individual methods
- `tests/data/integration/test_data_pipeline.py` - Integration tests with other components
- Update existing tests if behavior changes

### Test Guidelines

1. **Write tests first**: Consider writing tests before implementing features (TDD)
2. **Test edge cases**: Include tests for error conditions and edge cases
3. **Use fixtures**: Leverage pytest fixtures in `conftest.py` files
4. **Mock external dependencies**: Use mocks for external services/APIs
5. **Keep tests fast**: Unit tests should run quickly
6. **Clear test names**: Use descriptive names that explain what is being tested
7. **Arrange-Act-Assert**: Follow the AAA pattern in test structure