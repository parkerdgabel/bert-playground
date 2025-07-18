# bert train

Train BERT models optimized for Apple Silicon using MLX framework.

## Synopsis

```bash
bert train [OPTIONS]
```

## Description

The `train` command provides comprehensive training functionality for BERT models with MLX optimizations. It supports various model architectures, data formats, and training strategies optimized for Kaggle competitions.

## Options

### Data Options

- `--train, -t PATH` **(required)**: Path to training data file (CSV, JSON, or Parquet)
- `--val, -v PATH`: Path to validation data file
- `--test PATH`: Path to test data file (for final evaluation)

### Model Options

- `--model, -m TEXT`: Model type to use (default: "modernbert")
  - Available: `modernbert`, `modernbert-cnn`, `mlx-bert`, `bert-base-uncased`
- `--pretrained, -p TEXT`: Pretrained model name or path
- `--mlx-embeddings`: Use MLX embeddings for better performance

### Training Options

- `--epochs, -e INTEGER`: Number of training epochs (default: 3)
- `--batch-size, -b INTEGER`: Training batch size (default: 32)
- `--lr, -l FLOAT`: Learning rate (default: 2e-5)
- `--warmup, -w INTEGER`: Number of warmup steps (default: 500)
- `--grad-accum INTEGER`: Gradient accumulation steps (default: 1)

### Output Options

- `--output, -o PATH`: Output directory (default: "output")
- `--experiment TEXT`: MLflow experiment name
- `--run-name TEXT`: Name for this training run

### Performance Options

- `--mixed-precision/--no-mixed-precision`: Use mixed precision training (default: enabled)
- `--workers INTEGER`: Number of data loading workers (default: 4)
- `--profile`: Enable performance profiling

### Training Control

- `--seed, -s INTEGER`: Random seed for reproducibility (default: 42)
- `--resume, -r PATH`: Resume training from checkpoint
- `--early-stopping/--no-early-stopping`: Enable early stopping (default: enabled)
- `--save-best-only`: Save only the best model checkpoint

### Advanced Options

- `--config, -c PATH`: Load configuration from YAML/JSON file
- `--augment, -a`: Enable data augmentation
- `--debug, -d`: Enable debug mode with verbose output

## Examples

### Basic Training

Train a model with training and validation data:

```bash
bert train --train data/train.csv --val data/val.csv
```

### Training with Configuration

Use a configuration file for complex setups:

```bash
bert train --config configs/production.yaml
```

### Resume Training

Continue training from a checkpoint:

```bash
bert train --train data/train.csv --resume output/checkpoint_epoch_5
```

### MLX Embeddings

Use MLX embeddings for better performance:

```bash
bert train --train data/train.csv --mlx-embeddings --model mlx-bert
```

### Advanced Training

Full example with all options:

```bash
bert train \
    --train data/train.csv \
    --val data/val.csv \
    --test data/test.csv \
    --model modernbert-cnn \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-5 \
    --warmup 1000 \
    --grad-accum 2 \
    --output results/experiment_1 \
    --experiment "kaggle-titanic" \
    --run-name "cnn-model-v2" \
    --augment \
    --save-best-only
```

## Configuration File

You can specify all options in a configuration file:

```yaml
# configs/training.yaml
model:
  type: modernbert
  pretrained: answerdotai/ModernBERT-base
  use_mlx_embeddings: true

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  warmup_steps: 500
  gradient_accumulation: 2
  early_stopping: true
  save_best_only: true

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
  num_workers: 4
  augmentation: true

output:
  dir: output/
  experiment: my-experiment
  run_name: baseline-model
```

Then run:

```bash
bert train --config configs/training.yaml
```

## Output Structure

The training command creates the following output structure:

```
output/
├── checkpoints/
│   ├── checkpoint_epoch_1/
│   ├── checkpoint_epoch_2/
│   └── best_model/
├── logs/
│   ├── training.log
│   └── metrics.json
├── plots/
│   ├── loss_curve.png
│   └── metrics.png
└── final_model/
    ├── model.safetensors
    ├── config.json
    └── tokenizer/
```

## Integration with MLflow

When `--experiment` is specified, training metrics are automatically logged to MLflow:

- Parameters: All training configuration
- Metrics: Loss, accuracy, and custom metrics per epoch
- Artifacts: Model checkpoints, plots, and logs
- Tags: Model type, dataset info, git commit hash

View results with:

```bash
bert mlflow ui
```

## Performance Tips

1. **Batch Size**: Use the largest batch size that fits in memory
2. **Gradient Accumulation**: Use to simulate larger batch sizes
3. **Mixed Precision**: Keep enabled for faster training
4. **Workers**: Set to number of CPU cores for optimal data loading
5. **MLX Embeddings**: Use for 2-3x faster training on Apple Silicon

## Common Issues

### Out of Memory

Reduce batch size or use gradient accumulation:

```bash
bert train --train data/train.csv --batch-size 16 --grad-accum 4
```

### Slow Training

Enable MLX embeddings and increase workers:

```bash
bert train --train data/train.csv --mlx-embeddings --workers 8
```

### Poor Convergence

Adjust learning rate and warmup:

```bash
bert train --train data/train.csv --lr 5e-5 --warmup 1000
```

## See Also

- `bert predict` - Generate predictions with trained models
- `bert evaluate` - Evaluate model performance
- `bert kaggle submit` - Submit predictions to Kaggle
- `bert config` - Manage training configurations