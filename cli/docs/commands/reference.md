# Command Reference

Complete reference for all MLX BERT CLI commands.

## Core Commands

### bert train
Train BERT models with MLX optimizations.

```bash
bert train --train data/train.csv --val data/val.csv [OPTIONS]
```

**Key Options:**
- `--train, -t`: Training data path (required)
- `--val, -v`: Validation data path
- `--model, -m`: Model type (default: modernbert)
- `--epochs, -e`: Number of epochs (default: 3)
- `--batch-size, -b`: Batch size (default: 32)
- `--config, -c`: Configuration file

### bert predict
Generate predictions using trained models.

```bash
bert predict --test data/test.csv --checkpoint path/to/model [OPTIONS]
```

**Key Options:**
- `--test, -t`: Test data path (required)
- `--checkpoint, -c`: Model checkpoint path (required)
- `--output, -o`: Output file path
- `--format, -f`: Output format (csv, json, parquet)

### bert benchmark
Run performance benchmarks.

```bash
bert benchmark [OPTIONS]
```

**Key Options:**
- `--batch-size, -b`: Batch size for benchmarking
- `--seq-length, -s`: Sequence length
- `--steps`: Number of steps to run
- `--model, -m`: Model type to benchmark

### bert info
Display system and project information.

```bash
bert info [OPTIONS]
```

**Key Options:**
- `--mlx`: Show MLX information
- `--models`: List available models
- `--datasets`: List configured datasets
- `--all, -a`: Show all information

### bert init
Initialize a new BERT project.

```bash
bert init PROJECT_NAME [OPTIONS]
```

**Key Options:**
- `--template`: Project template to use
- `--force, -f`: Overwrite existing files

### bert config
Manage project configuration.

```bash
bert config ACTION [OPTIONS]
```

**Actions:**
- `init`: Create new configuration
- `validate`: Validate configuration
- `show`: Display current configuration
- `edit`: Open configuration in editor

## Kaggle Commands

### bert kaggle competitions
Manage Kaggle competitions.

```bash
bert kaggle competitions SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `list`: List competitions
- `info`: Show competition details
- `search`: Search competitions

### bert kaggle download
Download competition data.

```bash
bert kaggle download COMPETITION [OPTIONS]
```

**Key Options:**
- `--path, -p`: Download directory
- `--unzip`: Automatically unzip files

### bert kaggle submit
Submit predictions to Kaggle.

```bash
bert kaggle submit SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `create`: Create new submission
- `auto`: Auto-submit from checkpoint
- `history`: View submission history

### bert kaggle leaderboard
View competition leaderboard.

```bash
bert kaggle leaderboard COMPETITION [OPTIONS]
```

**Key Options:**
- `--top, -t`: Number of entries to show
- `--csv`: Export as CSV

## MLflow Commands

### bert mlflow server
Manage MLflow tracking server.

```bash
bert mlflow server SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `start`: Start MLflow server
- `stop`: Stop MLflow server
- `status`: Check server status
- `restart`: Restart server

### bert mlflow experiments
Manage MLflow experiments.

```bash
bert mlflow experiments SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `list`: List all experiments
- `create`: Create new experiment
- `delete`: Delete experiment
- `rename`: Rename experiment

### bert mlflow runs
Manage MLflow runs.

```bash
bert mlflow runs SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `list`: List runs in experiment
- `compare`: Compare multiple runs
- `delete`: Delete runs
- `export`: Export run data

### bert mlflow ui
Launch MLflow UI.

```bash
bert mlflow ui [OPTIONS]
```

**Key Options:**
- `--port, -p`: Port number (default: 5000)
- `--host, -h`: Host address

## Model Commands

### bert model serve
Serve model via REST API.

```bash
bert model serve --checkpoint path/to/model [OPTIONS]
```

**Key Options:**
- `--checkpoint, -c`: Model checkpoint path (required)
- `--port, -p`: Port number (default: 8080)
- `--host, -h`: Host address
- `--workers, -w`: Number of workers

### bert model registry
Manage model registry.

```bash
bert model registry SUBCOMMAND [OPTIONS]
```

**Subcommands:**
- `list`: List registered models
- `register`: Register new model
- `promote`: Promote model version
- `delete`: Delete model

### bert model evaluate
Evaluate model performance.

```bash
bert model evaluate --checkpoint path/to/model --data path/to/data [OPTIONS]
```

**Key Options:**
- `--checkpoint, -c`: Model checkpoint path
- `--data, -d`: Evaluation data path
- `--metrics, -m`: Metrics to compute

### bert model export
Export model to different formats.

```bash
bert model export FORMAT --checkpoint path/to/model [OPTIONS]
```

**Formats:**
- `onnx`: Export to ONNX format
- `coreml`: Export to CoreML
- `tflite`: Export to TensorFlow Lite

## Global Options

These options are available for all commands:

- `--help`: Show help message
- `--version, -v`: Show version
- `--verbose, -V`: Enable verbose output
- `--quiet, -q`: Minimal output
- `--debug`: Enable debug mode

## Environment Variables

- `BERT_CONFIG_PATH`: Default configuration file path
- `BERT_CLI_VERBOSE`: Enable verbose output (0/1)
- `BERT_CLI_QUIET`: Enable quiet mode (0/1)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `KAGGLE_USERNAME`: Kaggle username
- `KAGGLE_KEY`: Kaggle API key

## Output Formats

Many commands support multiple output formats:

- `human`: Human-readable format (default)
- `json`: JSON format for automation
- `csv`: CSV format for data analysis
- `yaml`: YAML format for configuration

Example:
```bash
bert kaggle competitions list --output json
```

## Configuration Files

Commands can load options from configuration files:

```yaml
# bert.yaml
model:
  type: modernbert
  pretrained: answerdotai/ModernBERT-base

training:
  epochs: 10
  batch_size: 32
```

Use with:
```bash
bert train --config bert.yaml
```

## Shell Completion

Enable tab completion:

```bash
# Bash
bert --install-completion bash

# Zsh
bert --install-completion zsh

# Fish
bert --install-completion fish
```