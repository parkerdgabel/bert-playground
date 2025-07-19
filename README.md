# MLX ModernBERT for Kaggle

Optimized ModernBERT implementation using Apple's MLX framework for solving Kaggle competitions with a text-based approach.

## Features

- 🚀 **MLX Optimized**: Fully optimized for Apple Silicon with MLX
- 📊 **Text-Based Tabular**: Converts tabular data to natural language
- 🔄 **Data Augmentation**: Multiple text templates for better generalization
- 📈 **MLflow Integration**: Complete experiment tracking
- 🎯 **Production Ready**: Unified CLI with multiple configurations
- ⚡ **High Performance**: Efficient data pipeline with prefetching
- 🏗️ **Modular Architecture**: Clean separation of concerns with pluggable components
- 🔌 **Extensible Design**: Easy to add new datasets and competition types

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd bert-playground

# Install dependencies using uv
uv sync
```

## Quick Start

### Production Training

```bash
# Standard training (recommended)
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --batch-size 32 \
    --epochs 5
```

### Using Configuration Files

```bash
# Use pre-defined configurations
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/production.json
```

### Generate Predictions

```bash
# Create Kaggle submission
uv run python mlx_bert_cli.py predict \
    --test data/titanic/test.csv \
    --checkpoint output/run_*/best_model_accuracy \
    --output submission.csv
```

### Benchmark Performance

```bash
# Test MLX performance
uv run python mlx_bert_cli.py benchmark \
    --batch-size 64 \
    --seq-length 256 \
    --steps 20
```

## Available Configurations

- **quick**: Fast testing (1 epoch, batch size 64)
- **standard**: Balanced training (5 epochs, batch size 32)
- **thorough**: Extended training (10 epochs, batch size 16)
- **mlx_optimized**: Optimized for MLX (5 epochs, batch size 64)

## CLI Commands

### Training
```bash
uv run python mlx_bert_cli.py train --help
```

### Prediction
```bash
uv run python mlx_bert_cli.py predict --help
```

### Benchmarking
```bash
uv run python mlx_bert_cli.py benchmark --help
```

### System Info
```bash
uv run python mlx_bert_cli.py info
```

## Project Structure

```
bert-playground/
├── mlx_bert_cli.py          # Main CLI interface
├── models/                  # Model implementations
│   ├── modernbert_optimized.py
│   └── classification_head.py
├── data/                    # Data processing (modular architecture)
│   ├── core/                # Core data classes
│   │   ├── base.py          # Base classes (KaggleDataset, DatasetSpec)
│   │   ├── metadata.py      # Competition metadata and analysis
│   │   └── registry.py      # Dataset registry for managing competitions
│   ├── loaders/             # Data loading implementations
│   │   ├── mlx_loader.py    # MLX-optimized data loader
│   │   ├── streaming.py     # Streaming pipeline for large datasets
│   │   └── memory.py        # Unified memory management
│   ├── templates/           # Text conversion templates
│   │   ├── engine.py        # Template engine and management
│   │   ├── converters.py    # Tabular to text converters
│   │   └── base_template.py # Base template interface
│   └── datasets/            # Competition-specific implementations
│       ├── titanic.py       # Titanic competition dataset
│       └── __init__.py      # Auto-discovery of datasets
├── training/                # Training logic
│   └── trainer_v2.py
├── utils/                   # Utilities
│   ├── logging_config.py
│   └── mlflow_utils.py
└── configs/                 # Configuration files
    └── production.json
```

## New Data Architecture

The project now features a completely redesigned data module with:

### Core Components

1. **Base Classes** (`data/core/base.py`)
   - `CompetitionType`: Enum for different competition types
   - `DatasetSpec`: Specifications for dataset configuration
   - `KaggleDataset`: Abstract base class for all Kaggle datasets

2. **Dataset Registry** (`data/core/registry.py`)
   - Automatic dataset discovery and registration
   - Competition metadata management
   - Dataset caching and versioning

3. **MLX-Optimized Loader** (`data/loaders/mlx_loader.py`)
   - Zero-copy operations with unified memory
   - Intelligent prefetching and batching
   - Gradient accumulation support
   - Memory pool management

4. **Streaming Pipeline** (`data/loaders/streaming.py`)
   - Handles large datasets that don't fit in memory
   - Adaptive batching based on throughput
   - Multi-threaded data loading

5. **Template Engine** (`data/templates/engine.py`)
   - Flexible text template system
   - Competition-specific templates
   - Custom converters for different data types

### Key Features

- **Modular Design**: Easy to extend with new datasets
- **High Performance**: Target throughput of 1000+ samples/sec
- **Memory Efficient**: Unified memory architecture for Apple Silicon
- **Type Safety**: Full typing support throughout
- **Comprehensive Testing**: 100% test coverage for core components

### Adding New Datasets

1. Create a new file in `data/datasets/` (e.g., `house_prices.py`)
2. Implement the `KaggleDataset` interface
3. Register with the dataset registry
4. The dataset will be auto-discovered!

Example:
```python
from data.core.base import KaggleDataset, DatasetSpec, CompetitionType

class HousePricesDataset(KaggleDataset):
    """House Prices competition dataset."""
    
    def __init__(self, spec: DatasetSpec, split: str = "train"):
        super().__init__(spec, split)
        
    def _load_data(self):
        # Load your data here
        pass
        
    def __getitem__(self, index):
        # Return a sample
        pass
```

## Performance

Expected performance on Apple Silicon with MLX-Optimized Loader:
- **M1/M2**: 1000+ samples/second
- **Batch size 32**: ~0.05-0.1 seconds/step
- **Batch size 64**: ~0.1-0.2 seconds/step
- **Zero-copy operations**: Minimal memory overhead
- **Streaming support**: Handle datasets of any size

## Tips

1. Use larger batch sizes (32-64) for better MLX performance
2. Enable data augmentation for improved accuracy
3. Use MLflow to track experiments
4. Adjust learning rate based on batch size

## License

MIT License