# MLX Embeddings Migration Guide

This guide helps you migrate from HuggingFace transformers to the native MLX embeddings library for improved performance on Apple Silicon.

## Overview

The mlx-embeddings library provides:
- Native MLX implementation of BERT and ModernBERT models
- Optimized tokenization for Apple Silicon
- 4-bit quantized models for efficient inference
- Backward compatibility with existing code

## Quick Start

### 1. Installation

The mlx-embeddings package is already included in the project dependencies:

```bash
uv sync  # This will install mlx-embeddings>=0.0.3
```

### 2. Using MLX Embeddings in Training

#### Option A: Using CLI flags

```bash
# Train with MLX embeddings backend
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --use-mlx-embeddings \
    --tokenizer-backend mlx \
    --model mlx-community/answerdotai-ModernBERT-base-4bit

# Use the predefined configuration
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/production.json \
    --use-mlx-embeddings
```

#### Option B: Using configuration files

```json
{
  "model_name": "mlx-community/answerdotai-ModernBERT-base-4bit",
  "use_mlx_embeddings": true,
  "tokenizer_backend": "mlx",
  "batch_size": 32,
  "learning_rate": 2e-5,
  "num_epochs": 5
}
```

### 3. Available Models

The following models are available with MLX embeddings:

| HuggingFace Model | MLX Community Model |
|------------------|---------------------|
| answerdotai/ModernBERT-base | mlx-community/answerdotai-ModernBERT-base-4bit |
| answerdotai/ModernBERT-large | mlx-community/answerdotai-ModernBERT-large-4bit |
| bert-base-uncased | mlx-community/bert-base-uncased-4bit |
| sentence-transformers/all-MiniLM-L6-v2 | mlx-community/all-MiniLM-L6-v2-4bit |

## Migrating Existing Code

### 1. Updating Data Loaders

The data loaders now support both backends transparently:

```python
from data import create_kaggle_dataloader

# Automatic backend selection (will use MLX if available)
loader = create_kaggle_dataloader(
    dataset_name="titanic",
    csv_path="data/titanic/train.csv",
    tokenizer_backend="auto",  # or "mlx" to force MLX
)

# Explicit MLX backend
loader = create_kaggle_dataloader(
    dataset_name="titanic",
    csv_path="data/titanic/train.csv",
    tokenizer_backend="mlx",
    tokenizer_name="mlx-community/answerdotai-ModernBERT-base-4bit",
)
```

### 2. Using the Tokenizer Wrapper

For custom tokenization needs:

```python
from embeddings import TokenizerWrapper

# Create tokenizer with automatic backend selection
tokenizer = TokenizerWrapper(
    model_name="answerdotai/ModernBERT-base",
    backend="auto"  # Will use MLX if available
)

# Force MLX backend
tokenizer = TokenizerWrapper(
    model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
    backend="mlx"
)

# Use like a regular tokenizer
encoded = tokenizer.batch_encode_plus(
    ["Hello world", "MLX is great"],
    padding=True,
    truncation=True
)
```

### 3. Creating Models with MLX Embeddings

```python
from models.factory import create_model

# Create MLX embedding model
model = create_model(
    model_type="mlx_embedding",
    model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
    num_labels=2,
)

# Or use the registry
model = create_model("mlx-modernbert-base", num_labels=2)
```

### 4. Direct MLX Embeddings Usage

For embedding generation tasks:

```python
from embeddings import MLXEmbeddingModel

# Create embedding model
model = MLXEmbeddingModel(
    model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
    num_labels=None,  # No classification head
    pooling_strategy="mean",
)

# Get embeddings
embeddings = model.get_embeddings(
    ["Text 1", "Text 2", "Text 3"],
    normalize=True
)
```

## Migrating Existing Checkpoints

### Automatic Migration

Use the migration utility to convert existing checkpoints:

```python
from embeddings.migration import migrate_checkpoint

# Migrate a checkpoint
migrated_path = migrate_checkpoint(
    checkpoint_path="output/run_20241210_120000/checkpoints/best_model",
    output_path="output/migrated_checkpoint"
)
```

### Command Line Migration

```bash
# Migration script (to be implemented)
uv run python -m embeddings.migration \
    --checkpoint output/run_20241210_120000/checkpoints/best_model \
    --output output/migrated_checkpoint
```

### Manual Migration

1. Copy your checkpoint directory
2. Update the `config.json`:
   ```json
   {
     "model_name": "mlx-community/answerdotai-ModernBERT-base-4bit",
     "use_mlx_embeddings": true,
     "tokenizer_backend": "mlx",
     "original_model_name": "answerdotai/ModernBERT-base"
   }
   ```
3. The weights file (`model.safetensors`) remains unchanged

## Performance Considerations

### Benefits of MLX Embeddings

1. **Native MLX Operations**: No Python/C++ bridge overhead
2. **4-bit Quantization**: Reduced memory usage and faster inference
3. **Optimized Tokenization**: Faster text processing on Apple Silicon
4. **Unified Memory**: Efficient data transfer between CPU and GPU

### Recommended Settings

```python
# Optimal batch sizes for MLX
batch_size = 64  # Larger batch sizes work well with MLX

# Use multiple workers for data loading
num_workers = 8

# Enable prefetching
prefetch_size = 4
```

## Troubleshooting

### Issue: MLX embeddings not available

```python
# Check availability
from embeddings import MLXEmbeddingsAdapter
adapter = MLXEmbeddingsAdapter()
print(f"MLX embeddings available: {adapter.is_available}")
```

### Issue: Model not found

Ensure you're using the correct model name:
```python
# Check model compatibility
from embeddings.migration import check_mlx_embeddings_compatibility
info = check_mlx_embeddings_compatibility("your-model-name")
print(info)
```

### Issue: Tokenizer differences

The TokenizerWrapper ensures compatibility, but if you need exact parity:
```python
# Force HuggingFace backend for comparison
tokenizer_hf = TokenizerWrapper(backend="huggingface")
tokenizer_mlx = TokenizerWrapper(backend="mlx")
```

## Best Practices

1. **Start with Auto Backend**: Use `backend="auto"` to let the system choose
2. **Test Before Full Migration**: Run a quick training test with MLX embeddings
3. **Monitor Performance**: Compare training speed and memory usage
4. **Keep Backups**: Always backup checkpoints before migration

## Example: Complete Training Script

```python
from mlx_bert_cli import train
from pathlib import Path

# Train with MLX embeddings
train(
    train_path=Path("data/titanic/train.csv"),
    val_path=Path("data/titanic/val.csv"),
    model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
    use_mlx_embeddings=True,
    tokenizer_backend="mlx",
    batch_size=64,
    learning_rate=2e-5,
    num_epochs=5,
    output_dir=Path("output/mlx_embeddings_run"),
    experiment_name="mlx_embeddings_test",
)
```

## Future Enhancements

1. **More Models**: Additional model architectures will be added
2. **Fine-tuning Support**: Direct fine-tuning of MLX embedding models
3. **Performance Optimizations**: Continuous improvements to MLX backend
4. **Automatic Migration**: One-click migration for all checkpoints

## Resources

- [MLX Embeddings GitHub](https://github.com/Blaizzy/mlx-embeddings)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Community Models](https://huggingface.co/mlx-community)

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test suite: `tests/test_mlx_embeddings.py`
3. Open an issue with details about your setup and error messages