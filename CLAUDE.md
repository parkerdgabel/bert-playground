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

### Using the Unified CLI

```bash
# Install dependencies
uv sync

# Quick training test
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --epochs 1 \
    --batch-size 32

# Standard production training
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/production.json

# Training with MLX embeddings (faster on Apple Silicon)
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --use-mlx-embeddings \
    --tokenizer-backend mlx \
    --model mlx-community/answerdotai-ModernBERT-base-4bit

# Generate predictions
uv run python mlx_bert_cli.py predict \
    --test data/titanic/test.csv \
    --checkpoint output/run_*/best_model_accuracy \
    --output submission.csv

# Benchmark performance
uv run python mlx_bert_cli.py benchmark \
    --batch-size 64 \
    --seq-length 256 \
    --steps 20

# System info
uv run python mlx_bert_cli.py info
```

### Production Commands

```bash
# Production training with all optimizations
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --batch-size 64 \
    --lr 2e-5 \
    --epochs 5 \
    --workers 8 \
    --augment \
    --experiment titanic_prod

# Training with best model only (saves storage)
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/best_model_only.yaml

# Quick test run
uv run python run_production.py --config quick

# Standard training (recommended)
uv run python run_production.py --config standard --enable-mlflow --predict

# Thorough training
uv run python run_production.py --config thorough --enable-mlflow --predict
```

## Project Structure

```
bert-playground/
├── mlx_bert_cli.py          # Unified CLI for all operations
├── models/
│   ├── modernbert_mlx.py    # Original MLX ModernBERT
│   ├── modernbert_optimized.py  # Optimized version with MLX best practices
│   └── classification_head.py   # Classification layers
├── embeddings/              # MLX embeddings integration
│   ├── mlx_adapter.py       # MLX embeddings adapter
│   ├── tokenizer_wrapper.py # Unified tokenizer interface
│   ├── model_wrapper.py     # MLX embedding model wrapper
│   └── migration.py         # Checkpoint migration utilities
├── data/
│   ├── titanic_loader.py    # Original data loader
│   ├── optimized_loader.py  # Optimized MLX-Data pipeline
│   └── text_templates.py    # Convert tabular data to text
├── training/
│   ├── trainer.py           # Basic trainer
│   └── trainer_v2.py        # Enhanced trainer with MLflow/logging
├── utils/
│   ├── logging_config.py    # Loguru configuration
│   ├── mlflow_utils.py      # MLflow tracking utilities
│   └── visualization.py     # Training visualization
├── configs/
│   └── production.json      # Production configurations
└── scripts/
    ├── train_production.sh  # Production training script
    └── download_titanic.py  # Download/create Titanic data
```

## Key Components

### 1. Optimized ModernBERT (`models/modernbert_optimized.py`)
- Fused QKV projections for efficiency
- Memory-efficient attention
- Optimized embeddings with position ID caching
- MLX-native save/load using safetensors

### 2. Optimized Data Pipeline (`data/optimized_loader.py`)
- Pre-tokenization for efficiency
- MLX-Data streaming with prefetching
- Multi-threaded data loading
- Efficient batching and caching

### 3. Unified CLI (`mlx_bert_cli.py`)
- Single entry point for all operations
- Rich console output with progress tracking
- Configuration file support
- Benchmarking utilities

### 4. Text-Based Approach (`data/text_templates.py`)
- Converts tabular data to natural language
- Multiple template variations for augmentation
- Handles missing values gracefully

## MLX Optimization Tips

1. **Batch Size**: Use larger batch sizes (32-64) for better MLX performance
2. **Prefetching**: Enable data prefetching with 4-8 prefetch size
3. **Workers**: Use 4-8 workers for data loading
4. **Gradient Accumulation**: Use when memory is limited
5. **Mixed Precision**: MLX handles this automatically

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
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/best_model_only.yaml
```

**Note:** Regular checkpoints are still saved during training when `save_best_only: false` (default).

## Common Issues and Solutions

### Issue: Slow Training
```bash
# Use optimized settings
uv run python mlx_bert_cli.py train \
    --batch-size 64 \
    --workers 8 \
    --config configs/production.json
```

### Issue: Out of Memory
```bash
# Reduce batch size and use gradient accumulation
uv run python mlx_bert_cli.py train \
    --batch-size 16 \
    --grad-accum 4
```

### Issue: Poor Accuracy
```bash
# Enable augmentation and train longer
uv run python mlx_bert_cli.py train \
    --augment \
    --epochs 10 \
    --lr 1e-5
```

## MLX Embeddings Integration

The project now supports native MLX embeddings for improved performance on Apple Silicon:

### Using MLX Embeddings

```bash
# Train with MLX embeddings backend (4-bit quantized models)
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --use-mlx-embeddings \
    --model mlx-community/answerdotai-ModernBERT-base-4bit
```

### Benefits
- **Native Performance**: Optimized for Apple Silicon
- **4-bit Quantization**: Reduced memory usage
- **Faster Tokenization**: MLX-native text processing
- **Backward Compatible**: Works with existing code

### Migration Guide
See `MLX_EMBEDDINGS_MIGRATION_GUIDE.md` for detailed migration instructions.

## MLflow Integration

### View Experiments
```bash
# Launch MLflow UI
mlflow ui --backend-store-uri ./output/mlruns

# Or use the training script
uv run python train_titanic_v2.py --launch_mlflow
```

### Compare Runs
- Navigate to http://localhost:5000
- Select experiment "titanic_modernbert"
- Compare metrics across runs

## Development Workflow

1. **Testing Changes**
   ```bash
   # Quick test with minimal data
   uv run python mlx_bert_cli.py train --epochs 1 --batch-size 8
   ```

2. **Benchmarking**
   ```bash
   # Test performance improvements
   uv run python mlx_bert_cli.py benchmark --steps 50
   ```

3. **Production Training**
   ```bash
   # Use production config
   uv run python mlx_bert_cli.py train --config configs/production.json
   ```

## Kaggle Integration

The project now includes comprehensive Kaggle competition integration with dedicated CLI commands:

### Competition Management

```bash
# List active competitions
uv run python mlx_bert_cli.py kaggle-competitions --limit 10

# Filter by category or search
uv run python mlx_bert_cli.py kaggle-competitions --category "tabular" --search "classification"

# Download competition data
uv run python mlx_bert_cli.py kaggle-download titanic --output data/titanic
```

### Submission Commands

```bash
# Submit predictions to Kaggle
uv run python mlx_bert_cli.py kaggle-submit titanic submission.csv \
    --message "MLX-BERT with attention heads" \
    --checkpoint output/run_001/best_model

# Auto-submit from checkpoint
uv run python mlx_bert_cli.py kaggle-auto-submit titanic \
    output/run_001/best_model \
    data/titanic/test.csv
```

### Leaderboard & History

```bash
# View competition leaderboard
uv run python mlx_bert_cli.py kaggle-leaderboard titanic --top 50

# View your submission history
uv run python mlx_bert_cli.py kaggle-history titanic --limit 20

# Generate detailed submission report
uv run python mlx_bert_cli.py kaggle-history titanic \
    --report reports/titanic_submissions.json
```

### Dataset Management

```bash
# Search Kaggle datasets
uv run python mlx_bert_cli.py kaggle-datasets --search "nlp classification"

# Download specific dataset
uv run python mlx_bert_cli.py kaggle-download-dataset username/dataset-name
```

### MLflow Integration
All Kaggle submissions are automatically tracked in MLflow when active:
- Submission scores logged as metrics
- Submission files saved as artifacts
- Competition metadata tracked as parameters

## Adding New Kaggle Datasets

1. Create data converter in `data/` following `text_templates.py` pattern
2. Update `optimized_loader.py` to handle new dataset
3. Add new config in `configs/` directory
4. Test with small batch first

## Performance Metrics

Expected performance on M1/M2 Mac:
- Batch size 32: ~1.5-2.0 seconds/step
- Batch size 64: ~2.5-3.5 seconds/step
- Throughput: 15-30 samples/second

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
uv run python mlx_bert_cli.py train --log-level DEBUG
```

Check MLX device:
```bash
uv run python mlx_bert_cli.py info
```

## Important Notes

- Always use `uv run` to execute Python scripts
- MLX works best with batch sizes that are powers of 2
- The model uses random initialization (pretrained weights not loaded)
- Data augmentation significantly improves performance
- MLflow tracking is optional but recommended for experiments