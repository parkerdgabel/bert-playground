# MLflow Integration Guide

This guide covers the comprehensive MLflow integration in the MLX BERT project, including experiment tracking, model registry, and best practices for managing ML experiments.

## Table of Contents

- [Overview](#overview)
- [Setup and Configuration](#setup-and-configuration)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [CLI Integration](#cli-integration)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

MLflow integration provides:
- **Experiment Tracking**: Log metrics, parameters, and artifacts
- **Model Registry**: Version and deploy models
- **Visualization**: Compare runs and experiments
- **Reproducibility**: Track code versions and environments

## Setup and Configuration

### 1. Installation

```bash
# MLflow is included in project dependencies
uv sync

# Or install separately
pip install mlflow
```

### 2. Server Setup

```bash
# Start MLflow tracking server
bert mlflow server --port 5000

# With custom backend store
bert mlflow server \
    --backend-store sqlite:///mlflow.db \
    --artifact-store ./mlruns

# Run in background
bert mlflow server --detach
```

### 3. Environment Configuration

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or configure in code
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

### 4. Project Configuration

```yaml
# configs/mlflow.yaml
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: bert-kaggle
  run_name: null  # Auto-generated
  tags:
    project: mlx-bert
    environment: development
  log_models: true
  log_artifacts: true
  registry_uri: null  # Same as tracking_uri
```

## Experiment Tracking

### Basic Usage

```python
from training import create_trainer
from training.callbacks import MLflowCallback

# Create trainer with MLflow tracking
trainer = create_trainer(
    model,
    callbacks=[MLflowCallback()],
    experiment_name="titanic-competition"
)

# Training automatically logs to MLflow
result = trainer.train(train_loader, val_loader)
```

### Manual Logging

```python
import mlflow
import mlflow.pytorch

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "modernbert")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 2e-5)
    
    # Log metrics
    mlflow.log_metric("train_loss", 0.45)
    mlflow.log_metric("val_accuracy", 0.89)
    
    # Log artifacts
    mlflow.log_artifact("configs/production.yaml")
    mlflow.log_artifacts("outputs/", artifact_path="model_outputs")
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Automatic Logging

```python
from training.integrations import MLflowConfig

# Configure automatic logging
mlflow_config = MLflowConfig(
    experiment_name="bert-experiments",
    run_name="modernbert-v2",
    log_models=True,
    log_artifacts=True,
    log_model_signatures=True,
    log_input_examples=True,
    tags={
        "competition": "titanic",
        "cv_folds": "5",
        "ensemble": "yes"
    }
)

# Use with trainer
trainer = create_trainer(
    model,
    mlflow_config=mlflow_config
)
```

### What Gets Logged

#### Parameters
- Model architecture and configuration
- Training hyperparameters
- Data configuration
- Optimizer and scheduler settings

#### Metrics
- Training loss (per step and epoch)
- Validation metrics
- Learning rate
- Best model scores
- Custom competition metrics

#### Artifacts
- Model checkpoints
- Configuration files
- Predictions (OOF and test)
- Plots and visualizations
- Training logs

## Model Registry

### Registering Models

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from run
run_id = "your-run-id"
model_uri = f"runs:/{run_id}/model"
model_name = "bert-titanic-classifier"

# Create registered model
client.create_registered_model(model_name)

# Create model version
model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)
```

### Model Stages

```python
# Transition model to staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

### Loading Models

```python
# Load specific version
model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

# Load production model
model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/Production"
)

# Load from run
model = mlflow.pytorch.load_model(
    model_uri=f"runs:/{run_id}/model"
)
```

## CLI Integration

### MLflow Commands

```bash
# Start server
bert mlflow server --port 5000

# Launch UI
bert mlflow ui

# List experiments
bert mlflow experiments list

# Create experiment
bert mlflow experiments create "new-competition"

# List runs
bert mlflow runs list --experiment "bert-experiments"

# Compare runs
bert mlflow runs compare run1 run2 --metric val_accuracy

# Find best run
bert mlflow runs best \
    --experiment "titanic" \
    --metric "val_accuracy" \
    --max
```

### Training with MLflow

```bash
# Train with MLflow tracking
bert train \
    --train data/train.csv \
    --val data/val.csv \
    --experiment titanic-v2 \
    --run-name "modernbert-focal-loss" \
    --tags "loss:focal,cv:5fold"

# Train with model registration
bert train \
    --train data/train.csv \
    --val data/val.csv \
    --register-model \
    --model-name "titanic-classifier"
```

## Advanced Features

### 1. Experiment Comparison

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get all runs for experiment
experiment = client.get_experiment_by_name("titanic")
runs = client.search_runs(experiment.experiment_id)

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {
        "run_id": run.info.run_id,
        "model_type": run.data.params.get("model_type"),
        "val_accuracy": run.data.metrics.get("best_val_accuracy"),
        "cv_score": run.data.metrics.get("cv_mean_score"),
        "training_time": run.data.metrics.get("training_time_seconds")
    }
    for run in runs
])

# Find best configuration
best_run = comparison_df.loc[comparison_df["val_accuracy"].idxmax()]
```

### 2. Hyperparameter Search

```python
from training.integrations import MLflowHyperparamSearch

# Define search space
search_space = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "batch_size": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
    "lora_rank": [4, 8, 16]
}

# Run hyperparameter search
searcher = MLflowHyperparamSearch(
    experiment_name="hyperparam-search",
    search_space=search_space,
    metric="val_accuracy",
    direction="maximize"
)

best_params = searcher.search(
    create_model_fn=create_model,
    train_fn=train_model,
    n_trials=20
)
```

### 3. Custom Metrics and Artifacts

```python
class CustomMLflowCallback(MLflowCallback):
    """Extended MLflow callback with custom logging."""
    
    def on_epoch_end(self, trainer, state):
        super().on_epoch_end(trainer, state)
        
        # Log custom metrics
        mlflow.log_metric(
            "learning_rate",
            trainer.optimizer.learning_rate,
            step=state.epoch
        )
        
        # Log confusion matrix
        if hasattr(state, "confusion_matrix"):
            fig = plot_confusion_matrix(state.confusion_matrix)
            mlflow.log_figure(fig, f"confusion_matrix_epoch_{state.epoch}.png")
        
        # Log model complexity
        mlflow.log_metric("model_parameters", count_parameters(trainer.model))
```

### 4. Model Signatures

```python
from mlflow.models.signature import infer_signature

# Infer signature from data
signature = infer_signature(
    train_loader.dataset[0]["input_ids"].numpy(),
    predictions.numpy()
)

# Log model with signature
mlflow.pytorch.log_model(
    model,
    "model",
    signature=signature,
    input_example=train_loader.dataset[0]
)
```

### 5. Automated Experiment Analysis

```python
from training.integrations import MLflowExperimentAnalyzer

analyzer = MLflowExperimentAnalyzer("titanic-competition")

# Generate report
report = analyzer.generate_report()
print(report)

# Output:
# Experiment Summary: titanic-competition
# Total Runs: 45
# Best Accuracy: 0.8934 (run_abc123)
# Average CV Score: 0.8756 ± 0.0123
# Most Successful Config: modernbert + lora_rank=8
```

## Best Practices

### 1. Experiment Organization

```python
# Use hierarchical naming
experiment_names = {
    "development": "dev/titanic/exploratory",
    "hyperparameter_tuning": "tuning/titanic/lr_search",
    "cross_validation": "cv/titanic/5fold",
    "final_models": "production/titanic/ensemble"
}

# Tag consistently
tags = {
    "stage": "development",
    "dataset_version": "v2",
    "feature_set": "text_only",
    "cv_folds": "5",
    "ensemble": "voting"
}
```

### 2. Run Naming Convention

```python
def generate_run_name(config):
    """Generate descriptive run name."""
    components = [
        config.model_type,
        f"lr{config.learning_rate}",
        f"bs{config.batch_size}",
        f"fold{config.fold_id}" if hasattr(config, "fold_id") else "",
        datetime.now().strftime("%Y%m%d_%H%M")
    ]
    return "_".join(filter(None, components))

# Example: modernbert_lr2e-5_bs32_fold1_20240115_1430
```

### 3. Artifact Management

```python
# Organize artifacts by type
artifact_structure = {
    "models/": "Model checkpoints",
    "predictions/": "OOF and test predictions",
    "configs/": "Configuration files",
    "plots/": "Visualizations",
    "logs/": "Training logs",
    "data/": "Preprocessed data samples"
}

# Log with structure
for artifact_type, files in artifacts.items():
    mlflow.log_artifacts(files, artifact_path=artifact_type)
```

### 4. Metric Tracking Strategy

```python
# Log metrics at different granularities
metrics_strategy = {
    "step": ["train_loss", "learning_rate"],
    "epoch": ["train_loss_avg", "val_loss", "val_accuracy"],
    "run": ["best_val_score", "training_time", "total_parameters"]
}

# Custom metric aggregation
class MetricAggregator:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metric_name, value):
        self.metrics[metric_name].append(value)
    
    def log_summary(self):
        for name, values in self.metrics.items():
            mlflow.log_metric(f"{name}_mean", np.mean(values))
            mlflow.log_metric(f"{name}_std", np.std(values))
            mlflow.log_metric(f"{name}_max", np.max(values))
```

### 5. Reproducibility

```python
# Log environment information
mlflow.log_param("python_version", sys.version)
mlflow.log_param("mlx_version", mlx.__version__)
mlflow.log_param("git_commit", get_git_commit_hash())
mlflow.log_param("random_seed", config.seed)

# Log requirements
mlflow.log_artifact("requirements.txt")

# Log system info
mlflow.log_dict(get_system_info(), "system_info.json")
```

## Integration Examples

### Complete Training Pipeline

```python
from training import create_trainer
from training.integrations import setup_mlflow

def train_with_mlflow(config):
    # Setup MLflow
    mlflow_run = setup_mlflow(
        experiment_name=config.competition,
        run_name=f"{config.model_type}_cv_{config.timestamp}",
        tags={
            "competition": config.competition,
            "phase": "cross_validation",
            "user": getpass.getuser()
        }
    )
    
    with mlflow_run:
        # Log configuration
        mlflow.log_params(config.to_dict())
        
        # Create and log model architecture
        model = create_model(**config.model_params)
        mlflow.log_text(str(model), "model_architecture.txt")
        
        # Train with automatic logging
        trainer = create_trainer(
            model,
            config=config,
            callbacks=[
                MLflowCallback(
                    log_models=True,
                    log_best_model_only=True
                )
            ]
        )
        
        # Cross-validation
        cv_results = trainer.train_with_cv(
            train_loader,
            n_folds=5
        )
        
        # Log CV results
        mlflow.log_metric("cv_mean_score", cv_results.mean_score)
        mlflow.log_metric("cv_std_score", cv_results.std_score)
        mlflow.log_artifact(cv_results.oof_predictions_path)
        
        # Register best model
        if cv_results.mean_score > config.score_threshold:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                config.model_registry_name
            )
```

### Experiment Dashboard

```python
from training.integrations import MLflowDashboard

# Create dashboard
dashboard = MLflowDashboard(experiment_name="titanic")

# Generate visualizations
dashboard.plot_metric_history("val_accuracy")
dashboard.plot_parameter_importance()
dashboard.plot_run_comparison(["run1", "run2", "run3"])

# Export report
dashboard.export_html("experiment_report.html")
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if server is running
   bert mlflow health
   
   # Restart server
   bert mlflow server restart
   ```

2. **Storage Issues**
   ```bash
   # Clean old runs
   bert mlflow cleanup --older-than 30d
   
   # Compact database
   bert mlflow compact-db
   ```

3. **Slow UI**
   ```bash
   # Use pagination
   bert mlflow ui --page-size 50
   
   # Filter experiments
   bert mlflow ui --experiment-filter "active"
   ```

4. **Model Registry Errors**
   ```python
   # Check model exists
   try:
       client.get_registered_model(model_name)
   except:
       client.create_registered_model(model_name)
   ```

### Performance Tips

1. **Batch Metric Logging**
   ```python
   # Instead of logging one by one
   metrics = {"loss": 0.5, "acc": 0.9, "f1": 0.85}
   mlflow.log_metrics(metrics, step=epoch)
   ```

2. **Async Logging**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   executor = ThreadPoolExecutor(max_workers=2)
   executor.submit(mlflow.log_artifacts, "outputs/")
   ```

3. **Selective Artifact Logging**
   ```python
   # Log only best model
   if is_best_model:
       mlflow.pytorch.log_model(model, "best_model")
   ```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Project MLflow Integration](../training/integrations/mlflow.py)