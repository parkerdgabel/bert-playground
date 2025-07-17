# Migration Guide: Unified Training System

This guide helps you migrate from the old training scripts to the new unified training system.

## Quick Start

The new unified training system consolidates all training functionality into a single, optimized implementation. 

### Old Way → New Way

#### Basic Training
```bash
# Old
uv run python train_titanic_v2.py --data data/titanic/train.csv --epochs 5

# New
uv run python mlx_bert_cli_v2.py train --train data/titanic/train.csv --epochs 5
```

#### Optimized Training
```bash
# Old
uv run python train_optimized.py --base_batch_size 32 --max_batch_size 64

# New
uv run python mlx_bert_cli_v2.py train --batch-size 32 --max-batch-size 64
```

#### CNN Hybrid Model
```bash
# Old
uv run python train_cnn_hybrid.py --kernel_sizes 2,3,4 --num_filters 128

# New
uv run python mlx_bert_cli_v2.py train --model-type cnn_hybrid --cnn-kernels 2,3,4 --cnn-filters 128
```

## Key Changes

### 1. Single Entry Point
- All training now goes through `mlx_bert_cli_v2.py`
- Use `train_unified.py` as a convenient wrapper

### 2. Unified Configuration
- Single `UnifiedTrainingConfig` dataclass
- Consistent parameter names across all model types
- Configuration files in JSON format

### 3. MLflow Integration
- Single MLflow database at `mlruns/mlflow.db`
- Use `mlx_bert_cli_v2.py list-experiments` to see all experiments
- Automatic experiment tracking with `--experiment` flag

### 4. Data Loading
- `UnifiedTitanicDataPipeline` is the standard loader
- `EnhancedUnifiedDataPipeline` adds persistent caching and dynamic batching
- All old loaders are deprecated but still work via compatibility wrappers

## Feature Mapping

| Old Feature | New Implementation |
|------------|-------------------|
| `TitanicTrainerV2` | `MLXTrainer` with MLflow enabled |
| `MLXOptimizedTrainer` | `MLXTrainer` with optimizations enabled |
| `gradient_accumulation_steps` | `--grad-accum` |
| `enable_mlflow` | `--no-mlflow` to disable (enabled by default) |
| `dynamic_batch_size` | `--dynamic-batch` |
| `early_stopping_patience` | `--early-stopping` |

## Advanced Features

### Dynamic Batching
```bash
uv run python mlx_bert_cli_v2.py train \
  --batch-size 16 \
  --max-batch-size 64 \
  --dynamic-batch
```

### Persistent Caching
The enhanced loader automatically uses persistent caching for optimized runs:
```bash
uv run python mlx_bert_cli_v2.py train \
  --train data/titanic/train.csv \
  --config configs/production.json
```

### Resume Training
```bash
uv run python mlx_bert_cli_v2.py train \
  --train data/titanic/train.csv \
  --resume output/run_*/checkpoints/best
```

## Configuration Files

Create a JSON config file for complex setups:

```json
{
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "batch_size": 32,
  "max_batch_size": 64,
  "enable_dynamic_batching": true,
  "gradient_accumulation": 2,
  "early_stopping_patience": 3,
  "label_smoothing": 0.1
}
```

Use it with:
```bash
uv run python mlx_bert_cli_v2.py train \
  --train data/titanic/train.csv \
  --config my_config.json
```

## Deprecation Timeline

- **Immediate**: All old training scripts are deprecated
- **Next Release**: Deprecation warnings will be added
- **Future**: Old scripts will be moved to `legacy/` folder

## Getting Help

- Run `uv run python mlx_bert_cli_v2.py --help` for all options
- Run `uv run python mlx_bert_cli_v2.py train --help` for training options
- Check `CONSOLIDATION_PLAN.md` for technical details

## Benefits of Migration

1. **Performance**: Up to 30% faster training with unified optimizations
2. **Simplicity**: One command for all training scenarios  
3. **Consistency**: Same interface for all model types
4. **Tracking**: Centralized MLflow tracking
5. **Maintenance**: Single codebase to maintain and improve