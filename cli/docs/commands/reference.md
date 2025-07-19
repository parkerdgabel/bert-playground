# MLX BERT CLI Command Reference

This document provides a comprehensive reference for all commands available in the MLX BERT CLI.

## Table of Contents

- [Core Commands](#core-commands)
  - [train](#train)
  - [predict](#predict)
  - [benchmark](#benchmark)
  - [info](#info)
- [Kaggle Commands](#kaggle-commands)
  - [competitions](#kaggle-competitions)
  - [datasets](#kaggle-datasets)
  - [download](#kaggle-download)
  - [submit](#kaggle-submit)
  - [auto-submit](#kaggle-auto-submit)
  - [leaderboard](#kaggle-leaderboard)
  - [history](#kaggle-history)
- [MLflow Commands](#mlflow-commands)
  - [server](#mlflow-server)
  - [experiments](#mlflow-experiments)
  - [runs](#mlflow-runs)
  - [ui](#mlflow-ui)
  - [health](#mlflow-health)
- [Model Commands](#model-commands)
  - [serve](#model-serve)
  - [export](#model-export)
  - [evaluate](#model-evaluate)
  - [inspect](#model-inspect)
  - [list](#model-list)

## Core Commands

### train

Train a BERT model on your dataset.

```bash
bert train [OPTIONS]
```

**Required Options:**
- `--train PATH`: Path to training data (CSV/Parquet)
- `--val PATH`: Path to validation data (CSV/Parquet)

**Key Options:**
- `--config PATH`: Configuration file (YAML/JSON)
- `--model TEXT`: Model architecture (bert/modernbert/neobert)
- `--head TEXT`: Task head type (binary/multiclass/regression)
- `--epochs INT`: Number of training epochs
- `--batch-size INT`: Training batch size
- `--lr FLOAT`: Learning rate
- `--workers INT`: Number of data loading workers
- `--output PATH`: Output directory for checkpoints
- `--experiment TEXT`: MLflow experiment name
- `--augment`: Enable data augmentation
- `--resume PATH`: Resume from checkpoint

**LoRA Options:**
- `--use-lora`: Enable LoRA adapters
- `--lora-preset TEXT`: Preset configuration (efficient/balanced/expressive)
- `--lora-rank INT`: LoRA rank
- `--lora-alpha FLOAT`: LoRA alpha scaling
- `--lora-target TEXT`: Target modules (comma-separated)

**Cross-Validation Options:**
- `--cv-folds INT`: Number of CV folds
- `--cv-strategy TEXT`: CV strategy (kfold/stratified/group/timeseries)
- `--save-oof-predictions`: Save out-of-fold predictions

**Examples:**
```bash
# Basic training
bert train --train data/train.csv --val data/val.csv --epochs 5

# With configuration file
bert train --train data/train.csv --val data/val.csv --config configs/production.yaml

# With LoRA
bert train --train data/train.csv --val data/val.csv --use-lora --lora-preset balanced

# Cross-validation
bert train --train data/train.csv --val data/val.csv --cv-folds 5 --cv-strategy stratified
```

### predict

Generate predictions using a trained model.

```bash
bert predict [OPTIONS]
```

**Required Options:**
- `--test PATH`: Path to test data
- `--checkpoint PATH`: Path to model checkpoint

**Options:**
- `--output PATH`: Output file path (default: submission.csv)
- `--batch-size INT`: Prediction batch size
- `--tta-rounds INT`: Test-time augmentation rounds
- `--tta-aggregate TEXT`: TTA aggregation method (mean/median/vote)
- `--format TEXT`: Output format (csv/parquet/json)

**Examples:**
```bash
# Basic prediction
bert predict --test data/test.csv --checkpoint output/best_model

# With TTA
bert predict --test data/test.csv --checkpoint output/best_model --tta-rounds 5
```

### benchmark

Benchmark model performance.

```bash
bert benchmark [OPTIONS]
```

**Options:**
- `--model TEXT`: Model to benchmark
- `--batch-size INT`: Batch size for benchmarking
- `--seq-length INT`: Sequence length
- `--steps INT`: Number of benchmark steps
- `--warmup-steps INT`: Warmup steps before timing
- `--memory`: Track memory usage

**Examples:**
```bash
# Quick benchmark
bert benchmark --steps 20

# Detailed benchmark
bert benchmark --batch-size 64 --seq-length 512 --steps 100 --memory
```

### info

Display system and environment information.

```bash
bert info [OPTIONS]
```

**Options:**
- `--mlx`: Show MLX-specific information
- `--dependencies`: List all dependencies
- `--gpu`: Show GPU information

## Kaggle Commands

Commands for interacting with Kaggle competitions and datasets.

### kaggle competitions

List and search Kaggle competitions.

```bash
bert kaggle competitions [OPTIONS]
```

**Options:**
- `--limit INT`: Maximum number of competitions to show
- `--category TEXT`: Filter by category (tabular/nlp/vision)
- `--search TEXT`: Search competitions by keyword
- `--sort TEXT`: Sort by (deadline/prize/teams)
- `--active-only`: Show only active competitions

**Examples:**
```bash
# List active tabular competitions
bert kaggle competitions --category tabular --active-only

# Search for NLP competitions
bert kaggle competitions --search "nlp classification" --limit 20
```

### kaggle datasets

Search and explore Kaggle datasets.

```bash
bert kaggle datasets [OPTIONS]
```

**Options:**
- `--search TEXT`: Search query
- `--limit INT`: Maximum results
- `--sort TEXT`: Sort by (votes/updated/size)
- `--file-type TEXT`: Filter by file type

**Examples:**
```bash
# Search NLP datasets
bert kaggle datasets --search "sentiment analysis" --limit 10
```

### kaggle download

Download competition data.

```bash
bert kaggle download COMPETITION [OPTIONS]
```

**Arguments:**
- `COMPETITION`: Competition name (e.g., "titanic")

**Options:**
- `--output PATH`: Output directory
- `--unzip`: Automatically unzip files
- `--train-split FLOAT`: Create train/val split

**Examples:**
```bash
# Download Titanic competition
bert kaggle download titanic --output data/titanic --unzip

# Download with train/val split
bert kaggle download titanic --output data/titanic --train-split 0.2
```

### kaggle submit

Submit predictions to a competition.

```bash
bert kaggle submit COMPETITION SUBMISSION_FILE [OPTIONS]
```

**Arguments:**
- `COMPETITION`: Competition name
- `SUBMISSION_FILE`: Path to submission file

**Options:**
- `--message TEXT`: Submission message
- `--checkpoint PATH`: Associated checkpoint path
- `--track-mlflow`: Log to MLflow

**Examples:**
```bash
# Basic submission
bert kaggle submit titanic submission.csv --message "BERT with attention"

# With checkpoint tracking
bert kaggle submit titanic submission.csv --checkpoint output/best_model
```

### kaggle auto-submit

Generate predictions and submit automatically.

```bash
bert kaggle auto-submit COMPETITION CHECKPOINT TEST_DATA [OPTIONS]
```

**Arguments:**
- `COMPETITION`: Competition name
- `CHECKPOINT`: Model checkpoint path
- `TEST_DATA`: Test data path

**Options:**
- `--message TEXT`: Submission message
- `--tta-rounds INT`: TTA rounds
- `--batch-size INT`: Prediction batch size

**Examples:**
```bash
# Auto-generate and submit
bert kaggle auto-submit titanic output/best_model data/test.csv
```

### kaggle leaderboard

View competition leaderboard.

```bash
bert kaggle leaderboard COMPETITION [OPTIONS]
```

**Arguments:**
- `COMPETITION`: Competition name

**Options:**
- `--top INT`: Show top N entries
- `--team TEXT`: Search for specific team
- `--export PATH`: Export to file

**Examples:**
```bash
# View top 20
bert kaggle leaderboard titanic --top 20

# Export leaderboard
bert kaggle leaderboard titanic --export leaderboard.csv
```

### kaggle history

View submission history.

```bash
bert kaggle history COMPETITION [OPTIONS]
```

**Arguments:**
- `COMPETITION`: Competition name

**Options:**
- `--limit INT`: Number of submissions to show
- `--report PATH`: Generate detailed report
- `--plot`: Show score progression plot

**Examples:**
```bash
# View recent submissions
bert kaggle history titanic --limit 10

# Generate report
bert kaggle history titanic --report submissions.json --plot
```

## MLflow Commands

Commands for managing MLflow experiments and runs.

### mlflow server

Start or manage MLflow tracking server.

```bash
bert mlflow server [OPTIONS]
```

**Options:**
- `--port INT`: Server port (default: 5000)
- `--host TEXT`: Server host (default: localhost)
- `--backend-store PATH`: Backend store location
- `--artifact-store PATH`: Artifact store location
- `--workers INT`: Number of workers
- `--detach`: Run in background

**Examples:**
```bash
# Start server
bert mlflow server --port 5000

# Start with custom storage
bert mlflow server --backend-store ./mlruns --artifact-store ./artifacts
```

### mlflow experiments

Manage MLflow experiments.

```bash
bert mlflow experiments COMMAND [OPTIONS]
```

**Commands:**
- `list`: List all experiments
- `create NAME`: Create new experiment
- `delete ID`: Delete experiment
- `rename ID NEW_NAME`: Rename experiment

**Examples:**
```bash
# List experiments
bert mlflow experiments list

# Create experiment
bert mlflow experiments create "bert_titanic_v2"
```

### mlflow runs

View and compare MLflow runs.

```bash
bert mlflow runs COMMAND [OPTIONS]
```

**Commands:**
- `list`: List runs in experiment
- `compare RUN1 RUN2`: Compare two runs
- `best`: Show best run by metric

**Options:**
- `--experiment TEXT`: Experiment name/ID
- `--metric TEXT`: Metric to compare
- `--max`: Maximize metric (default: minimize)

**Examples:**
```bash
# List runs
bert mlflow runs list --experiment "bert_titanic"

# Find best run
bert mlflow runs best --experiment "bert_titanic" --metric "val_accuracy" --max
```

### mlflow ui

Launch MLflow UI.

```bash
bert mlflow ui [OPTIONS]
```

**Options:**
- `--port INT`: UI port
- `--backend-store PATH`: Backend store location

**Examples:**
```bash
# Launch UI
bert mlflow ui --port 5000
```

### mlflow health

Check MLflow server health.

```bash
bert mlflow health [OPTIONS]
```

**Options:**
- `--url TEXT`: MLflow server URL
- `--verbose`: Show detailed information

## Model Commands

Commands for model management and deployment.

### model serve

Serve model via REST API.

```bash
bert model serve CHECKPOINT [OPTIONS]
```

**Arguments:**
- `CHECKPOINT`: Model checkpoint path

**Options:**
- `--port INT`: Server port (default: 8080)
- `--host TEXT`: Server host
- `--workers INT`: Number of workers
- `--reload`: Auto-reload on changes
- `--cors`: Enable CORS

**Examples:**
```bash
# Basic serving
bert model serve output/best_model --port 8080

# Production serving
bert model serve output/best_model --port 8080 --workers 4
```

### model export

Export model to different formats.

```bash
bert model export CHECKPOINT [OPTIONS]
```

**Arguments:**
- `CHECKPOINT`: Model checkpoint path

**Options:**
- `--format TEXT`: Export format (onnx/coreml/tflite)
- `--output PATH`: Output path
- `--opset INT`: ONNX opset version
- `--quantize`: Apply quantization

**Examples:**
```bash
# Export to ONNX
bert model export output/best_model --format onnx --output model.onnx

# Export with quantization
bert model export output/best_model --format coreml --quantize
```

### model evaluate

Evaluate model performance.

```bash
bert model evaluate CHECKPOINT TEST_DATA [OPTIONS]
```

**Arguments:**
- `CHECKPOINT`: Model checkpoint path
- `TEST_DATA`: Test data path

**Options:**
- `--metrics TEXT`: Metrics to compute (comma-separated)
- `--batch-size INT`: Evaluation batch size
- `--report PATH`: Save evaluation report
- `--confusion-matrix`: Show confusion matrix

**Examples:**
```bash
# Basic evaluation
bert model evaluate output/best_model data/test.csv

# Detailed evaluation
bert model evaluate output/best_model data/test.csv --metrics "accuracy,f1,auc" --confusion-matrix
```

### model inspect

Inspect model architecture and parameters.

```bash
bert model inspect CHECKPOINT [OPTIONS]
```

**Arguments:**
- `CHECKPOINT`: Model checkpoint path

**Options:**
- `--layers`: Show layer details
- `--params`: Show parameter counts
- `--config`: Show configuration
- `--export PATH`: Export inspection report

**Examples:**
```bash
# Basic inspection
bert model inspect output/best_model

# Detailed inspection
bert model inspect output/best_model --layers --params --export report.json
```

### model list

List available models.

```bash
bert model list [OPTIONS]
```

**Options:**
- `--source TEXT`: Model source (local/mlflow/hub)
- `--pattern TEXT`: Filter by pattern
- `--sort TEXT`: Sort by (date/size/name)

**Examples:**
```bash
# List local models
bert model list --source local

# List MLflow models
bert model list --source mlflow --pattern "bert_titanic"
```

## Environment Variables

The CLI respects the following environment variables:

- `BERT_OUTPUT_DIR`: Default output directory
- `BERT_CONFIG_PATH`: Default config file path
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `KAGGLE_USERNAME`: Kaggle username
- `KAGGLE_KEY`: Kaggle API key
- `LOGURU_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `BERT_CACHE_DIR`: Cache directory for models and data

## Configuration Files

Configuration files can be in YAML or JSON format. Here's an example:

```yaml
# config.yaml
model:
  architecture: modernbert
  hidden_size: 768
  num_layers: 12
  num_heads: 12

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  gradient_accumulation_steps: 2
  
optimizer:
  type: adamw
  weight_decay: 0.01
  
scheduler:
  type: cosine
  warmup_steps: 500
  
data:
  max_length: 256
  num_workers: 8
  prefetch_size: 4

callbacks:
  - type: early_stopping
    patience: 3
    metric: val_loss
  - type: model_checkpoint
    save_best_only: true
    metric: val_accuracy
```

## Tips and Best Practices

1. **Start with Quick Tests**: Use `--epochs 1` and small batch sizes to test your setup
2. **Use Configuration Files**: Keep your experiments reproducible with config files
3. **Enable MLflow**: Track all experiments with `--experiment` flag
4. **Monitor Memory**: Use `bert info` to check available memory before training
5. **Leverage Presets**: Use `--config configs/production.yaml` for optimized settings
6. **Cross-Validation**: Use CV for robust model evaluation in competitions
7. **LoRA for Fine-tuning**: Use LoRA adapters to reduce memory usage
8. **Batch Size**: Use powers of 2 for optimal MLX performance