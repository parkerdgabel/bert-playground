# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project using uv package manager to implement various ModernBERT implementations with MLX and MLX-Data for solving Kaggle problems. The project uses Python 3.13.

## Development Environment Setup

### Package Management
- **uv** is the primary package manager for this project
- Python version: 3.13 (specified in `.python-version`)

### Common Commands

```bash
# Install dependencies
uv pip install -e .

# Add a new dependency
uv add <package-name>

# Add MLX and MLX-Data dependencies
uv add mlx mlx-data

# Add development dependencies
uv add --dev pytest ruff mypy

# Run Python scripts
uv run python <script.py>

# Create virtual environment (if needed)
uv venv

# Sync dependencies
uv sync
```

## Project Architecture

### Expected Structure
When implementing ModernBERT models for Kaggle problems, organize code as follows:

```
bert-playground/
├── models/           # BERT model implementations
├── data/            # Data loading and preprocessing utilities
├── training/        # Training scripts and utilities
├── evaluation/      # Model evaluation code
├── kaggle/          # Kaggle-specific problem solutions
└── utils/           # Common utilities
```

### MLX-Specific Considerations
- Use MLX arrays instead of NumPy/PyTorch tensors
- Leverage MLX's lazy evaluation for efficient computation
- Use mlx.nn modules for neural network layers
- Implement data loading with mlx-data for efficient batching

### ModernBERT Implementation Notes
- Focus on efficient implementations suitable for Apple Silicon
- Consider using pre-trained weights when available
- Implement custom tokenizers compatible with MLX

## Testing & Quality

```bash
# Run tests (once pytest is added)
uv run pytest

# Run linter (once ruff is added)
uv run ruff check .

# Type checking (once mypy is added)
uv run mypy .
```

## MLflow & Logging

### Training with MLflow
```bash
# Full training pipeline with MLflow tracking
uv run python train_titanic_v2.py --do_train --do_predict --do_visualize --launch_mlflow

# Train without MLflow (faster for experiments)
uv run python train_titanic_v2.py --do_train --disable_mlflow

# View MLflow dashboard
mlflow ui --backend-store-uri ./output/mlruns --port 5000
```

### Logging Configuration
- Logs are saved to `./output/logs/`
- Use `--log_level DEBUG` for detailed debugging
- Structured logging with Loguru for better debugging
- Rich console output for better user experience

### Key Features
1. **MLflow Integration**: Automatic experiment tracking, model versioning, metrics visualization
2. **Enhanced Logging**: Structured logs, performance metrics, detailed error tracking
3. **Visualization Tools**: Training curves, confusion matrices, ROC curves, experiment comparison
4. **Model Registry**: Version control and deployment of best models

## Kaggle Integration
- Store competition-specific code in `kaggle/<competition-name>/`
- Keep data preprocessing pipelines reusable across competitions
- Document model performance and submission results