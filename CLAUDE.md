# CLAUDE.md - k-bert Development Guide

This file provides essential guidance to Claude Code when working with the k-bert MLX-based ModernBERT implementation for Kaggle competitions.

## Project Overview

k-bert is a state-of-the-art BERT implementation using Apple's MLX framework, designed to solve Kaggle tabular problems using text-based approaches. The project emphasizes configuration-driven development, plugin architecture, and efficient training on Apple Silicon.

### Core Technologies
- **MLX**: Apple's ML framework optimized for Apple Silicon
- **ModernBERT**: Answer.AI's 2024 BERT variant with RoPE, GeGLU, alternating attention
- **uv**: Fast Python package manager
- **Typer**: Modern CLI framework
- **Pydantic**: Configuration validation
- **MLflow**: Experiment tracking
- **Loguru**: Structured logging

## Project Structure

```
bert-playground/
├── cli/                     # CLI application (k-bert command)
│   ├── commands/           # Command implementations
│   │   ├── core/          # train, predict, benchmark, info
│   │   ├── config/        # Configuration management
│   │   ├── project/       # Project scaffolding
│   │   └── competition/   # Kaggle integration (future)
│   ├── config/            # Configuration system
│   ├── plugins/           # Plugin system for custom components
│   └── utils/             # CLI utilities
├── models/                 # Model implementations
│   ├── bert/              # BERT architectures
│   ├── heads/             # Task-specific heads
│   ├── lora/              # LoRA adapters
│   └── factory.py         # Model creation
├── data/                   # Data pipeline
│   ├── core/              # Protocols and base classes
│   ├── loaders/           # MLX-optimized data loading
│   ├── templates/         # Text conversion templates
│   └── factory.py         # Dataset creation
├── training/              # Training infrastructure
│   ├── core/              # Base trainer
│   ├── callbacks/         # Training callbacks
│   └── metrics/           # Evaluation metrics
├── configs/               # Example configurations
└── tests/                 # Comprehensive test suite
```

## Essential Commands

```bash
# Installation
uv sync

# Configuration-first workflow
k-bert config init            # Initialize user config
k-bert project init myproj    # Create new project
k-bert run                    # Run from project directory

# Core training commands
k-bert train --config configs/production.yaml
k-bert predict --checkpoint output/run_*/checkpoints/final --output predictions.csv
k-bert benchmark --batch-size 64

# System information
k-bert info --all
```

## Configuration System

The project uses a hierarchical configuration system:

1. **User Config**: `~/.k-bert/config.yaml` - Personal preferences
2. **Project Config**: `k-bert.yaml` - Project-specific settings
3. **Command Config**: `--config file.yaml` - Run-specific overrides
4. **CLI Arguments**: Direct command-line overrides

### Key Configuration Sections
- `models`: Model architecture and hyperparameters
- `training`: Training settings (epochs, batch size, learning rate)
- `data`: Data loading and preprocessing
- `checkpoint`: Checkpointing and model saving
- `logging`: Logging configuration
- `mlflow`: Experiment tracking settings

## Plugin System

k-bert supports custom components through a plugin system:

```python
# Project structure with plugins
myproject/
├── k-bert.yaml
├── src/
│   ├── heads/         # Custom task heads
│   ├── augmenters/    # Data augmentation
│   └── features/      # Feature engineering
└── data/
```

Plugins are automatically discovered and registered when running from a project directory.

## MLX Optimization

- **Batch Size**: Use 32-64 for optimal performance
- **Compilation**: Enabled by default, disable with `use_compilation: false`
- **Prefetching**: Set `prefetch_size: 4-8` for data loading
- **LoRA**: Use for memory-efficient fine-tuning
- **Pre-tokenization**: Enable with `--use-pretokenized` for faster loading

## Development Best Practices

### Repository Hygiene
- **Temporary Files**: ALWAYS use `/tmp` for debug scripts and temporary files
- **Documentation**: Only create docs when explicitly requested
- **Clean State**: Leave the repository in a polished state
- **Commits**: Make focused commits with clear messages

### Testing Requirements
- **Mandatory Testing**: ALWAYS run tests after changes
- **Test Structure**: Mirror code structure in tests/
  - `unit/`: Individual component tests
  - `integration/`: Component interaction tests
  - `e2e/`: Complete workflow tests
  - `contract/`: CLI stability tests
- **Test New Features**: Add tests for all new functionality

### Code Style
- Follow existing patterns and conventions
- Use type hints and docstrings
- Prefer configuration over hardcoding
- Use protocols for extensibility

## Key Implementation Details

### Model Factory
- Uses `modernbert_with_head` as default model type
- Automatically selects appropriate head based on labels
- Supports classic BERT, ModernBERT, and custom architectures

### Data Pipeline
- Protocol-based design for flexibility
- MLX-optimized loaders with zero-copy operations
- Automatic caching for pre-tokenized data
- Template system for tabular-to-text conversion

### Training Infrastructure
- Declarative configuration with YAML/JSON
- Callback system for extensibility
- Automatic MLflow integration
- Comprehensive checkpointing with resume support

### CLI Design
- Typer-based with rich console output
- Hierarchical command structure
- Comprehensive error handling
- Configuration-first approach

## Testing Commands

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest tests/cli/unit/
uv run pytest tests/cli/integration/
uv run pytest tests/cli/contract/

# Run with coverage
uv run pytest --cov=bert_playground --cov-report=html

# Run specific test
uv run pytest tests/cli/unit/test_config_manager.py -v
```

## Important Notes

- Always use `uv run` for Python execution
- Package name is `k-bert` on PyPI
- Entry point is `k-bert` command (replaces `bert_cli.py`)
- Configuration files use `k-bert.yaml` naming
- MLX performs best with batch sizes that are powers of 2
- Pre-tokenization cache stored in `data/.tokenizer_cache/`
- Training logs automatically saved to `{run_dir}/training.log`

## Quick Reference

### File Locations
- User config: `~/.k-bert/config.yaml`
- Project config: `./k-bert.yaml`
- Plugin source: `./src/`
- Output directory: `./output/`
- MLflow tracking: `./output/mlruns/`

### Environment Variables
- `K_BERT_CONFIG_PATH`: Override config location
- `LOGURU_LEVEL`: Set logging level (DEBUG, INFO, etc.)
- `K_BERT_CACHE_DIR`: Override cache directory

### Common Workflows
1. **New Project**: `k-bert config init` → `k-bert project init` → customize → `k-bert run`
2. **Training**: Edit config → `k-bert train --config myconfig.yaml`
3. **Experimentation**: Use different configs with MLflow tracking
4. **Plugin Development**: Add to src/ → test → use in config