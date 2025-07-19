# Declarative Training Module for MLX-BERT

A modern, configuration-driven training module optimized for Kaggle competitions using Apple's MLX framework.

## Features

- **Declarative Configuration**: Define entire training runs via YAML/JSON configs
- **Protocol-Based Design**: Flexible, extensible architecture using Python protocols
- **MLX Optimizations**: Native support for Apple Silicon performance
- **Kaggle Integration**: Built-in support for competitions, CV, ensembles, and submissions
- **MLflow Tracking**: First-class experiment tracking and model registry
- **Comprehensive Callbacks**: Extensible hook system for custom training behavior
- **Advanced Features**: Gradient accumulation, mixed precision, distributed training

## Quick Start

### Basic Training

```python
from training import create_trainer
from models import create_model

# Create model
model = create_model("bert_with_head", head_type="binary_classification")

# Create trainer with preset config
trainer = create_trainer(model, config="production")

# Train
result = trainer.train(train_loader, val_loader)
```

### Kaggle Competition

```python
from training import create_kaggle_trainer

# Create Kaggle trainer for Titanic
trainer = create_kaggle_trainer(
    model=model,
    competition="titanic",
    test_dataloader=test_loader
)

# Train with cross-validation
cv_results = trainer.train_with_cv(train_loader)

# Create submission
submission_path = trainer.create_submission()
```

## Architecture

### Core Components

1. **Protocols** (`core/protocols.py`): Define interfaces for all components
2. **Base Trainer** (`core/base.py`): Core training loop with MLX optimizations
3. **Configuration** (`core/config.py`): Hierarchical config system with validation
4. **State Management** (`core/state.py`): Training state and checkpoint management
5. **Optimization** (`core/optimization.py`): Optimizers, schedulers, gradient handling

### Callback System

Built-in callbacks:
- `EarlyStopping`: Stop training when metric stops improving
- `ModelCheckpoint`: Save best and periodic checkpoints
- `LearningRateScheduler`: Dynamic learning rate adjustment
- `ProgressBar`: Visual training progress
- `MLflowCallback`: Automatic experiment tracking
- `MetricsLogger`: Comprehensive metrics collection

### Kaggle Features

- **Cross-Validation**: K-fold CV with OOF predictions
- **Ensemble Support**: Voting, blending, stacking
- **Pseudo-Labeling**: Semi-supervised learning
- **Test-Time Augmentation**: Improved predictions
- **Auto-Submission**: Direct submission to Kaggle

## Configuration

### Hierarchical Configuration

```yaml
# trainer_config.yaml
optimizer:
  type: adamw
  learning_rate: 2e-5
  weight_decay: 0.01

scheduler:
  type: cosine
  warmup_ratio: 0.1

data:
  batch_size: 32
  num_workers: 8

training:
  num_epochs: 10
  eval_strategy: epoch
  early_stopping: true

kaggle:
  competition_name: titanic
  cv_folds: 5
  enable_ensemble: true
```

### Preset Configurations

- `quick_test`: Fast testing with minimal epochs
- `development`: Balanced settings for development
- `production`: Optimized for production training
- `kaggle`: Competition-optimized settings
- `titanic`, `house_prices`, etc.: Competition-specific presets

## Advanced Usage

### Custom Callbacks

```python
from training.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, state):
        print(f"Epoch {state.epoch} completed!")

trainer = create_trainer(
    model,
    callbacks=[CustomCallback()]
)
```

### Configuration Override

```python
trainer = create_trainer(
    model,
    config="production",
    # Override specific settings
    learning_rate=1e-5,
    num_epochs=20,
    batch_size=64
)
```

### MLflow Integration

```python
from training.integrations import MLflowConfig

mlflow_config = MLflowConfig(
    experiment_name="bert-experiments",
    run_name="titanic-v2",
    log_models=True,
    register_model=True,
    model_name="titanic-bert"
)

trainer = create_trainer(
    model,
    callbacks=[MLflowCallback(mlflow_config)]
)
```

## API Reference

### Factory Functions

- `create_trainer()`: Create trainer with configuration
- `create_kaggle_trainer()`: Create Kaggle-specific trainer
- `get_trainer_config()`: Get configuration object
- `register_trainer()`: Register custom trainer class

### Trainer Methods

- `train()`: Standard training loop
- `train_with_cv()`: Cross-validation training (Kaggle)
- `evaluate()`: Evaluate on dataset
- `predict()`: Generate predictions
- `save_checkpoint()`: Save training state
- `load_checkpoint()`: Resume from checkpoint

### Configuration Classes

- `BaseTrainerConfig`: Standard trainer configuration
- `KaggleTrainerConfig`: Kaggle-specific configuration
- `OptimizerConfig`: Optimizer settings
- `SchedulerConfig`: Learning rate scheduler settings

## Migration from Old Module

The new module maintains compatibility while offering a cleaner API:

### Old Way
```python
from training.mlx_trainer import MLXTrainer
from training.config import TrainingConfig

config = TrainingConfig(...)
trainer = MLXTrainer(model, config, ...)
trainer.train(...)
```

### New Way
```python
from training import create_trainer

trainer = create_trainer(model, config="production")
result = trainer.train(train_loader, val_loader)
```

## Best Practices

1. **Use Configuration Files**: Define configs in YAML for reproducibility
2. **Leverage Presets**: Start with preset configs and override as needed
3. **Enable MLflow**: Track all experiments for comparison
4. **Use Callbacks**: Extend functionality without modifying core code
5. **Cross-Validation**: Always use CV for Kaggle competitions

## Performance Tips

1. **Batch Size**: Use largest batch size that fits in memory
2. **Gradient Accumulation**: Simulate larger batches
3. **Mixed Precision**: Automatic in MLX
4. **Data Loading**: Use multiple workers and prefetching
5. **Checkpoint Strategy**: Save best model only to reduce I/O

## Future Enhancements

- [ ] Distributed training across multiple devices
- [ ] Advanced ensemble methods
- [ ] AutoML capabilities
- [ ] More competition presets
- [ ] Enhanced visualization