# MLX BERT CLI - Modular Command Line Interface

A comprehensive, modular CLI for the MLX BERT playground, designed for Kaggle competitions on Apple Silicon.

## Overview

The CLI has been refactored from a monolithic 1761-line file into a clean, modular structure with organized command groups, shared utilities, and comprehensive documentation.

## Command Structure

### Core Commands (`bert <command>`)
Direct access to frequently used commands:
- `train` - Train BERT models with MLX optimizations
- `predict` - Generate predictions from trained models
- `benchmark` - Run performance benchmarks
- `info` - Display system and project information

### Command Groups

#### 🏆 Kaggle (`bert kaggle`)
Competition integration and workflow automation:
- `competitions` - List and search Kaggle competitions
- `datasets` - Browse and search datasets
- `download` - Download competition data
- `download-dataset` - Download specific datasets
- `submit` - Submit predictions to competitions
- `auto-submit` - Automated submission workflow
- `leaderboard` - View competition rankings
- `history` - View submission history

#### 📊 MLflow (`bert mlflow`)
Experiment tracking and model management:
- `server` - Start MLflow tracking server
- `restart` - Restart MLflow server
- `health` - Check MLflow health and configuration
- `experiments` - List and manage experiments
- `runs` - View runs for experiments
- `clean` - Clean up old experiments and runs
- `dashboard` - Real-time monitoring dashboard
- `test` - Run MLflow test suite

#### 🤖 Model (`bert model`)
Model management and deployment:
- `list` - List available models and checkpoints
- `inspect` - Inspect model architecture and parameters
- `export` - Export models to different formats (MLX, ONNX, CoreML)
- `convert` - Convert between model formats
- `serve` - Serve models as REST API
- `evaluate` - Evaluate model performance

## Features

### 🎨 Rich Console Output
- Beautiful tables and formatted output using Rich
- Progress bars for long-running operations
- Color-coded status messages and errors
- Interactive confirmations for destructive operations

### 🛡️ Robust Error Handling
- Comprehensive error decorators
- Graceful failure with helpful error messages
- Automatic rollback for failed operations
- Debug mode with detailed stack traces

### 📝 Comprehensive Documentation
- Detailed help text for every command
- Examples included in command docstrings
- Type hints for all parameters
- Markdown-formatted output

### ⚡ Performance Optimizations
- Lazy imports for faster CLI startup
- Parallel execution where possible
- Efficient data streaming
- Smart caching for repeated operations

### 🔧 Developer Experience
- Modular architecture for easy extension
- Shared utilities and decorators
- Consistent command patterns
- Testing-friendly structure

## Architecture

```
cli/
├── __init__.py          # Package initialization
├── app.py               # Main CLI application
├── README.md            # This file
├── commands/            # Command implementations
│   ├── core/           # Core commands (train, predict, etc.)
│   ├── kaggle/         # Kaggle integration commands
│   ├── mlflow/         # MLflow management commands
│   └── model/          # Model management commands
├── utils/              # Shared utilities
│   ├── console.py      # Rich console utilities
│   ├── validators.py   # Input validation functions
│   ├── decorators.py   # Command decorators
│   └── config.py       # Configuration management
└── docs/               # Additional documentation
```

## Usage Examples

### Training a Model
```bash
# Basic training
bert train --train data/train.csv --val data/val.csv

# Training with config file
bert train --config configs/production.yaml

# Training with MLX embeddings
bert train --train data/train.csv --mlx-embeddings
```

### Kaggle Workflow
```bash
# Download competition data
bert kaggle download titanic

# Train model
bert train --train data/titanic/train.csv

# Generate and submit predictions
bert kaggle auto-submit titanic output/best_model data/titanic/test.csv

# Check submission status
bert kaggle history titanic
```

### Model Management
```bash
# List available models
bert model list --type checkpoint

# Inspect model architecture
bert model inspect output/best_model --detailed

# Export to ONNX
bert model export --checkpoint output/best_model --format onnx

# Serve model as API
bert model serve --checkpoint output/best_model --port 8000
```

### MLflow Integration
```bash
# Start MLflow server
bert mlflow server --port 5000

# Check MLflow health
bert mlflow health --detailed

# View experiments
bert mlflow experiments --active

# Launch dashboard
bert mlflow dashboard
```

## Installation

The CLI is automatically available when you install the bert-playground package:

```bash
uv sync
uv run bert --help
```

## Configuration

The CLI supports multiple configuration methods:
1. Command-line arguments (highest priority)
2. Configuration files (YAML/JSON)
3. Environment variables
4. Default values

## Extending the CLI

To add new commands:

1. Create a new module in the appropriate command group
2. Implement the command function with proper decorators
3. Register the command in the group's `__init__.py`
4. Add tests and documentation

Example:
```python
# cli/commands/kaggle/new_command.py
from ...utils import handle_errors, track_time

@handle_errors
@track_time("Running new command")
def new_command(
    param: str = typer.Option(..., help="Parameter description")
):
    """Command description with examples."""
    # Implementation
```

## Testing

Run the CLI test suite:
```bash
uv run pytest tests/cli/
```

## Contributing

When contributing to the CLI:
1. Follow the established patterns
2. Add comprehensive help text
3. Include examples in docstrings
4. Use type hints
5. Add appropriate error handling
6. Write tests for new commands

## Future Enhancements

- [ ] Interactive mode for guided workflows
- [ ] Shell completion support
- [ ] Plugin system for custom commands
- [ ] Web UI for CLI operations
- [ ] Workflow automation commands
- [ ] Advanced configuration management