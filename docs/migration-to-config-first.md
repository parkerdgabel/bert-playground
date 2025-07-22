# Migration Guide: Config-First CLI

This guide explains the changes to k-bert's CLI interface and how to migrate from the old command-heavy approach to the new config-first approach.

## Overview of Changes

The k-bert CLI has been redesigned to prioritize configuration files over command-line arguments. This change improves:

- **Reproducibility**: All settings are in version-controlled files
- **Simplicity**: Shorter, cleaner commands
- **Organization**: Experiments and settings in one place
- **Shareability**: Easy to share configurations with teammates

## Quick Comparison

### Old Approach (Many CLI Arguments)
```bash
# Training with many arguments
k-bert train \
    --train data/train.csv \
    --val data/val.csv \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --max-length 256 \
    --warmup-ratio 0.1 \
    --early-stopping 3 \
    --output output/run1 \
    --experiment titanic \
    --model answerdotai/ModernBERT-base
```

### New Approach (Config-First)
```bash
# Training with configuration file
k-bert train

# Or with specific experiment
k-bert train --experiment baseline
```

## Creating Your Configuration

### 1. Interactive Setup (Recommended)
```bash
# Create project configuration interactively
k-bert config init --project

# Or use a preset for common competitions
k-bert config init --project --preset titanic
```

### 2. Manual Creation
Create a `k-bert.yaml` file in your project root:

```yaml
name: my-bert-project
description: My BERT project for Kaggle

# Model settings
models:
  default_model: answerdotai/ModernBERT-base
  head:
    type: binary_classification

# Data configuration  
data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
  batch_size: 32
  max_length: 256

# Training settings
training:
  default_epochs: 5
  default_learning_rate: 2e-5
  output_dir: ./outputs
```

## Command Migration

### Training

**Old:**
```bash
k-bert train --train data/train.csv --val data/val.csv --epochs 5 --batch-size 32
```

**New:**
```bash
# Ensure k-bert.yaml exists with your settings
k-bert train

# Override specific settings if needed
k-bert train --epochs 10

# Use different experiment
k-bert train --experiment long_training
```

### Prediction

**Old:**
```bash
k-bert predict --test data/test.csv --checkpoint output/model --batch-size 64
```

**New:**
```bash
# Uses config from checkpoint or project
k-bert predict output/model

# Override test data if needed
k-bert predict output/model --test data/test_final.csv
```

### Benchmarking

**Old:**
```bash
k-bert benchmark --batch-size 64 --seq-length 256 --steps 100
```

**New:**
```bash
# Uses project configuration
k-bert benchmark

# Compare different batch sizes
k-bert benchmark --compare
```

## Using --no-config Flag

If you need to run without a configuration file (not recommended), use `--no-config`:

```bash
# Training without config (must provide required arguments)
k-bert train --no-config --train data/train.csv --val data/val.csv

# Prediction without config
k-bert predict output/model --no-config --test data/test.csv

# Benchmark without config
k-bert benchmark --no-config --batch-size 32
```

## Experiments

The new system supports multiple experiments in one config file:

```yaml
experiments:
  - name: quick_test
    description: Quick test with 1 epoch
    config:
      training:
        default_epochs: 1
        
  - name: full_training
    description: Full training run
    config:
      training:
        default_epochs: 10
        default_learning_rate: 1e-5
```

Run experiments with:
```bash
k-bert train --experiment quick_test
k-bert train --experiment full_training
```

## Benefits of Config-First

### 1. Version Control
```bash
# Track configuration changes
git add k-bert.yaml
git commit -m "Update learning rate for better convergence"
```

### 2. Easy Experimentation
```bash
# Run different experiments without changing commands
k-bert train --experiment baseline
k-bert train --experiment large_batch
k-bert train --experiment lora_efficient
```

### 3. Reproducibility
```bash
# Share your exact configuration
cp k-bert.yaml k-bert-best-model.yaml
# Colleague can reproduce: k-bert train --config k-bert-best-model.yaml
```

### 4. Documentation
Your configuration file serves as documentation:
- Model architecture choices
- Data preprocessing settings
- Training hyperparameters
- All in one readable file

## Tips for Migration

1. **Start with presets**: Use `k-bert config init --preset titanic` for common competitions
2. **Use experiments**: Define multiple experiments instead of changing CLI args
3. **Override sparingly**: Only override config values when truly needed
4. **Keep configs in git**: Version control your configurations
5. **Use dry-run**: Test with `k-bert train --dry-run` to validate config

## Common Patterns

### Development Workflow
```yaml
experiments:
  - name: dev
    description: Quick development testing
    config:
      training:
        default_epochs: 1
      data:
        batch_size: 8
```

```bash
# Quick test during development
k-bert train --experiment dev

# Full training when ready
k-bert train --experiment full_training
```

### Hyperparameter Search
Instead of multiple command variations, define experiments:

```yaml
experiments:
  - name: lr_1e5
    config:
      training:
        default_learning_rate: 1e-5
        
  - name: lr_2e5
    config:
      training:
        default_learning_rate: 2e-5
        
  - name: lr_5e5
    config:
      training:
        default_learning_rate: 5e-5
```

### Competition Submission
```bash
# Train best model
k-bert train --experiment best_settings

# Generate predictions
k-bert predict output/best_model/final_model

# Submit
k-bert competition submit titanic submission.csv
```

## Troubleshooting

### "No configuration file found"
Create one with: `k-bert config init --project`

### "Configuration validation error"
Check your YAML syntax and required fields. Use: `k-bert config validate`

### Need to use old style
Add `--no-config` flag and provide all required arguments

## Summary

The config-first approach makes k-bert more powerful and easier to use. Start by creating a configuration file, then enjoy cleaner commands and better reproducibility. The old CLI-heavy approach is still available with `--no-config` but is not recommended for regular use.