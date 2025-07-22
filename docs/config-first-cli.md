# K-BERT Config-First CLI

## Overview

K-BERT has adopted a configuration-first approach to simplify usage and improve reproducibility. Instead of long command lines with many arguments, you define your settings in a `k-bert.yaml` file and use simple commands.

## Quick Start

### 1. Create Configuration

```bash
# Interactive setup (recommended)
k-bert config init --project

# Use a competition preset
k-bert config init --project --preset titanic
```

### 2. Train Your Model

```bash
# Train with default settings from k-bert.yaml
k-bert train

# Train with a specific experiment
k-bert train --experiment baseline

# Quick test run
k-bert train --experiment quick_test
```

### 3. Generate Predictions

```bash
# Predict using trained model
k-bert predict output/run_20240120/final_model

# Specify different test data
k-bert predict output/model --test data/test_final.csv
```

### 4. Benchmark Performance

```bash
# Benchmark with your configuration
k-bert benchmark

# Compare different batch sizes
k-bert benchmark --compare
```

## Configuration File Structure

Your `k-bert.yaml` file contains all settings:

```yaml
name: my-bert-project
description: BERT for Kaggle competitions

# Model configuration
models:
  default_model: answerdotai/ModernBERT-base
  head:
    type: binary_classification

# Data paths and settings
data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
  batch_size: 32
  max_length: 256

# Training parameters
training:
  default_epochs: 5
  default_learning_rate: 2e-5
  output_dir: ./outputs

# Experiments for different settings
experiments:
  - name: quick_test
    description: Quick test with 1 epoch
    config:
      training:
        default_epochs: 1
        
  - name: full_training
    description: Complete training
    config:
      training:
        default_epochs: 10
```

## Benefits

1. **Simplicity**: `k-bert train` instead of dozens of arguments
2. **Reproducibility**: Share your `k-bert.yaml` to reproduce results
3. **Organization**: All settings in one version-controlled file
4. **Experiments**: Switch between configurations easily

## Override When Needed

You can still override specific settings:

```bash
# Override epochs
k-bert train --epochs 10

# Override data paths
k-bert train --train custom_train.csv
```

## Using Without Config

If you need to run without a configuration file, use `--no-config`:

```bash
k-bert train --no-config --train data/train.csv --val data/val.csv
```

## Best Practices

1. **Keep configs in git**: Version control your configurations
2. **Use experiments**: Define different experiments instead of changing arguments
3. **Start with presets**: Use `--preset titanic` for common competitions
4. **Document in config**: Your config file serves as documentation

## Example Workflow

```bash
# 1. Initialize project with Titanic preset
k-bert config init --project --preset titanic

# 2. Quick test to ensure everything works
k-bert train --experiment quick_test

# 3. Full training
k-bert train --experiment full_training

# 4. Generate predictions
k-bert predict output/full_training_20240120/final_model

# 5. Submit to Kaggle
k-bert competition submit titanic submission.csv
```

## Migration from Old CLI

See [migration-to-config-first.md](migration-to-config-first.md) for detailed migration guide from the old command-heavy approach.