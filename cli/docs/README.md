# MLX BERT CLI Documentation

The MLX BERT CLI provides a comprehensive command-line interface for training, evaluating, and deploying BERT models optimized for Apple Silicon using the MLX framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Structure](#command-structure)
4. [Core Commands](#core-commands)
5. [Kaggle Integration](#kaggle-integration)
6. [MLflow Integration](#mlflow-integration)
7. [Model Management](#model-management)
8. [Configuration](#configuration)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Enable shell completion (optional)
bert --install-completion
```

## Quick Start

```bash
# Initialize a new project
bert init my-kaggle-project
cd my-kaggle-project

# Train a model
bert train --train data/train.csv --val data/val.csv

# Generate predictions
bert predict --test data/test.csv --checkpoint output/best_model

# Submit to Kaggle
bert kaggle submit create --competition titanic --file predictions.csv
```

## Command Structure

The CLI is organized into logical command groups:

```
bert
├── train           # Train models
├── predict         # Generate predictions
├── benchmark       # Run performance tests
├── info           # Show system information
├── init           # Initialize projects
├── config         # Manage configuration
├── kaggle/        # Kaggle workflows
│   ├── competitions/
│   ├── submit/
│   └── datasets/
├── mlflow/        # Experiment tracking
│   ├── server
│   ├── experiments
│   └── runs
└── model/         # Model management
    ├── serve
    ├── registry
    └── export
```

## Core Commands

### Training

Train BERT models with various configurations:

```bash
# Basic training
bert train --train data/train.csv --val data/val.csv

# With configuration file
bert train --config configs/production.yaml

# With MLX embeddings
bert train --train data/train.csv --mlx-embeddings

# Resume from checkpoint
bert train --train data/train.csv --resume output/checkpoint_epoch_5
```

### Prediction

Generate predictions using trained models:

```bash
# Basic prediction
bert predict --test data/test.csv --checkpoint output/best_model

# With custom output format
bert predict --test data/test.csv --checkpoint output/model --format json
```

### Benchmarking

Test model performance:

```bash
# Basic benchmark
bert benchmark --batch-size 64 --steps 100

# With profiling
bert benchmark --model modernbert-cnn --profile
```

## Kaggle Integration

### Competition Management

```bash
# List competitions
bert kaggle competitions list --category tabular

# Get competition info
bert kaggle competitions info titanic

# Download competition data
bert kaggle download titanic --path data/
```

### Submissions

```bash
# Create submission
bert kaggle submit create --competition titanic --file predictions.csv

# Auto-submit from checkpoint
bert kaggle submit auto --competition titanic --checkpoint output/best_model

# View submission history
bert kaggle submit history --competition titanic
```

## MLflow Integration

### Server Management

```bash
# Start MLflow server
bert mlflow server start

# Check server status
bert mlflow server status

# Stop server
bert mlflow server stop
```

### Experiment Tracking

```bash
# List experiments
bert mlflow experiments list

# Create experiment
bert mlflow experiments create my-experiment

# Compare runs
bert mlflow runs compare --experiment my-experiment
```

## Model Management

### Model Serving

```bash
# Serve model via REST API
bert model serve --checkpoint output/best_model --port 8080

# With custom configuration
bert model serve --config serving.yaml
```

### Model Export

```bash
# Export to ONNX
bert model export onnx --checkpoint output/best_model

# Export to CoreML
bert model export coreml --checkpoint output/best_model
```

## Configuration

### Configuration Files

The CLI supports YAML and JSON configuration files:

```yaml
# bert.yaml
model:
  type: modernbert
  pretrained: answerdotai/ModernBERT-base

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  
data:
  train_path: data/train.csv
  val_path: data/val.csv
```

### Environment Variables

```bash
# Set default config path
export BERT_CONFIG_PATH=~/my-configs/bert.yaml

# Enable verbose output
export BERT_CLI_VERBOSE=1

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Examples

See the [examples directory](examples/) for detailed examples:

- [Basic Training Workflow](examples/basic/training.md)
- [Kaggle Competition Workflow](examples/basic/kaggle_competition.md)
- [Advanced Training Techniques](examples/advanced/advanced_training.md)
- [Production Deployment](examples/advanced/production.md)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   uv pip install -r requirements.txt
   ```

2. **MLX Compatibility**
   ```bash
   # Check MLX installation
   bert info --mlx
   ```

3. **Kaggle Authentication**
   ```bash
   # Set up Kaggle credentials
   kaggle config set -n username -v YOUR_USERNAME
   kaggle config set -n key -v YOUR_API_KEY
   ```

### Debug Mode

Run any command with `--debug` for detailed output:

```bash
bert train --train data/train.csv --debug
```

### Getting Help

```bash
# General help
bert --help

# Command-specific help
bert train --help

# Interactive help
bert interactive
```

## Advanced Features

### Shell Completion

Enable tab completion for your shell:

```bash
# Bash
bert --install-completion bash

# Zsh
bert --install-completion zsh

# Fish
bert --install-completion fish
```

### Output Formats

Most commands support multiple output formats:

```bash
# Human-readable (default)
bert kaggle competitions list

# JSON output for automation
bert kaggle competitions list --output json

# CSV output
bert mlflow runs list --output csv
```

### Workflow Commands

High-level commands for common workflows:

```bash
# Complete Kaggle competition workflow
bert workflow kaggle-compete --competition titanic

# Quick training workflow
bert workflow quick-train --data data/

# Production pipeline
bert workflow production --config configs/production.yaml
```