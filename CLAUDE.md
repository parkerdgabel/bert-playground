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

The CLI is now organized into logical command groups. You can access commands directly or through their groups:

#### Direct Commands (shortcuts for common operations)

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
uv run python -m cli info
```

#### Command Groups

##### Kaggle Commands (`bert kaggle`)
```bash
# List competitions
uv run python -m cli kaggle competitions --category tabular

# Download competition data
uv run python -m cli kaggle download titanic --output data/titanic

# Submit predictions
uv run python -m cli kaggle submit titanic submission.csv

# Auto-generate and submit
uv run python -m cli kaggle auto-submit titanic output/best_model data/test.csv

# View leaderboard
uv run python -m cli kaggle leaderboard titanic --top 20

# Check submission history
uv run python -m cli kaggle history titanic --limit 10
```

##### MLflow Commands (`bert mlflow`)
```bash
# Start MLflow server
uv run python -m cli mlflow server --port 5000

# View experiments
uv run python -m cli mlflow experiments list

# Compare runs
uv run python -m cli mlflow runs compare exp_001 exp_002

# Launch UI
uv run python -m cli mlflow ui

# Health check
uv run python -m cli mlflow health
```

##### Model Commands (`bert model`)
```bash
# Serve model via REST API
uv run python -m cli model serve output/best_model --port 8080

# Export to ONNX
uv run python -m cli model export output/best_model --format onnx

# Evaluate on test set
uv run python -m cli model evaluate output/best_model data/test.csv

# Inspect architecture
uv run python -m cli model inspect output/best_model

# List available models
uv run python -m cli model list --registry mlflow
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

```bash
# Train with LoRA adapter
uv run python -m cli train \
    --lora-preset balanced \
    --lora-target "query,value" \
    --lora-rank 8

# Available presets:
# - efficient (r=4): Minimal parameters
# - balanced (r=8): Good trade-off
# - expressive (r=16): Maximum flexibility
# - qlora_memory (r=4, 4-bit): Extreme memory savings
```

### MLflow Integration
All Kaggle submissions are automatically tracked in MLflow when active:
- Submission scores logged as metrics
- Submission files saved as artifacts
- Competition metadata tracked as parameters

## Advanced Training Features

### Cross-Validation
```bash
# K-fold cross-validation
uv run python -m cli train \
    --cv-folds 5 \
    --cv-strategy stratified \
    --save-oof-predictions
```

### Ensemble Training
```bash
# Train ensemble of models
uv run python -m cli train \
    --ensemble-size 3 \
    --ensemble-method voting \
    --ensemble-weights "0.5,0.3,0.2"
```

### Test-Time Augmentation
```bash
# Generate predictions with TTA
uv run python -m cli predict \
    --tta-rounds 5 \
    --tta-aggregate mean
```

### Pseudo-Labeling
```bash
# Semi-supervised learning
uv run python -m cli train \
    --pseudo-label-data data/unlabeled.csv \
    --pseudo-label-threshold 0.95
```

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
- `memory_efficient`: Minimal memory usage

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