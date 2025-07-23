# K-BERT CLI Refactoring Summary

## Overview

The k-bert CLI has been refactored to make configuration files the primary way to run commands, improving reproducibility and ease of use.

## Key Changes Implemented

### 1. Config-First Command Design

All core commands (train, predict, benchmark) now:
- Look for `k-bert.yaml` by default
- Support `--config` to specify a different configuration file
- Provide `--no-config` flag for running without configuration
- Have minimal CLI options for common overrides only

### 2. Enhanced Config Command

The `config init` command now supports:
- `--project` flag to create project configurations (k-bert.yaml)
- Competition presets (e.g., `--preset titanic`)
- Interactive and non-interactive modes
- Custom output paths

### 3. Simplified Command Signatures

**Before (20+ options):**
```bash
k-bert train --train data/train.csv --val data/val.csv --epochs 5 --batch-size 32 --lr 2e-5 --model answerdotai/ModernBERT-base --output outputs --warmup-ratio 0.1 --early-stopping 3 --save-best-only --mixed-precision --grad-clip 1.0 --seed 42 --max-length 256 --workers 4 --prefetch 4
```

**After (config-first):**
```bash
k-bert train  # Uses k-bert.yaml
```

### 4. Files Created/Modified

#### New Files:
- `cli/commands/infrastructure/train_refactored.py` - Refactored train command
- `cli/commands/infrastructure/predict_refactored.py` - Refactored predict command  
- `cli/commands/infrastructure/benchmark_refactored.py` - Refactored benchmark command
- `cli/commands/infrastructure/train_v2.py` - Simplified train implementation
- `configs/templates/k-bert.template.yaml` - Comprehensive config template
- `k-bert.example.yaml` - Simple example configuration
- `docs/config-first-cli.md` - User documentation
- `docs/cli-refactoring-summary.md` - This summary

#### Modified Files:
- `cli/commands/config/init.py` - Enhanced with project config support

### 5. Configuration Structure

The new k-bert.yaml structure includes:
```yaml
name: project-name
competition: kaggle-competition  # Optional

models:
  default_model: answerdotai/ModernBERT-base
  use_lora: false
  head:
    type: binary_classification

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
  batch_size: 32
  max_length: 256

training:
  default_epochs: 5
  default_learning_rate: 2e-5
  output_dir: ./outputs

mlflow:
  auto_log: true
  default_experiment: my-experiments

experiments:  # Optional
  - name: quick_test
    config:
      training:
        default_epochs: 1
```

## Benefits Achieved

1. **Simplicity**: Commands are much simpler to use
2. **Reproducibility**: All settings in version-controlled files
3. **Flexibility**: Easy to switch between configurations
4. **Documentation**: Config files serve as documentation
5. **Experimentation**: Built-in support for multiple experiments

## Migration Path

For existing users:

1. **Create a config file:**
   ```bash
   k-bert config init --project
   ```

2. **Use the new commands:**
   ```bash
   k-bert train  # Instead of long command with many args
   ```

3. **For quick tests without config:**
   ```bash
   k-bert train --no-config --train data/train.csv --val data/val.csv
   ```

## Next Steps for Full Implementation

To complete the refactoring:

1. **Replace original files** with refactored versions
2. **Update tests** to work with new command signatures
3. **Add deprecation warnings** to old-style usage
4. **Update README** and main documentation
5. **Test all commands** with various configurations

## Breaking Changes

- Commands now expect a configuration file by default
- Many CLI options have been removed (use config instead)
- `--config` now points to project config, not training config

## Backwards Compatibility

- All commands support `--no-config` mode for compatibility
- Critical options (train/val data, epochs, batch size) can still be overridden via CLI
- Existing scripts can add `--no-config` to work as before