# MLX BERT Training Module

A comprehensive, protocol-driven training infrastructure optimized for BERT models on Apple Silicon using MLX. This module provides a declarative configuration system, extensive callback support, and specialized features for Kaggle competitions.

## Features

- **Protocol-Based Architecture**: Clean interfaces using Python protocols for maximum flexibility
- **Declarative Configuration**: Complete training runs defined via YAML/JSON with validation
- **MLX Optimization**: Native Apple Silicon performance with unified memory and lazy evaluation
- **Callback System**: Extensible hooks for training customization without code modification
- **Kaggle Integration**: Cross-validation, ensembling, pseudo-labeling, and auto-submission
- **MLflow Tracking**: Automatic experiment tracking, model registry, and artifact management
- **State Management**: Comprehensive checkpointing with resume capability
- **Advanced Training**: Gradient accumulation, clipping, multiple optimizers and schedulers

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

## Architecture Overview

### Core Components

#### 1. Protocols (`core/protocols.py`)
Defines clean interfaces for all training components:
- `Model`: Forward pass, parameters, save/load
- `DataLoader`: Iteration and batching interface
- `Optimizer`: Parameter updates
- `LRScheduler`: Learning rate scheduling
- `Trainer`: Main training interface
- `TrainingHook`: Callback system interface

#### 2. Base Trainer (`core/base.py`)
Implements the core training loop with MLX optimizations:
- Efficient gradient computation using `value_and_grad`
- Automatic mixed precision (native in MLX)
- Gradient accumulation and clipping
- Comprehensive metric tracking
- Hook integration for extensibility

#### 3. Configuration System (`core/config.py`)
Hierarchical configuration with validation:
- `OptimizerConfig`: AdamW, Adam, SGD, Lion, Adafactor
- `SchedulerConfig`: Linear, cosine, exponential, reduce on plateau
- `DataConfig`: Batch size, workers, prefetching
- `TrainingConfig`: Epochs, evaluation, checkpointing
- `EnvironmentConfig`: Output, MLflow, debugging

#### 4. State Management (`core/state.py`)
- `TrainingState`: Tracks all training state (epoch, step, metrics, history)
- `TrainingStateManager`: Persistence and recovery
- `CheckpointManager`: Model and optimizer checkpointing with safetensors

#### 5. Optimization (`core/optimization.py`)
- Optimizer creation with MLX-native implementations
- Learning rate schedulers with warmup support
- Gradient accumulation for large effective batch sizes
- Global gradient norm clipping

### Callback System

#### Built-in Callbacks

1. **EarlyStopping** (`callbacks/early_stopping.py`)
   - Monitor any metric (loss, accuracy, etc.)
   - Configurable patience and min_delta
   - Option to restore best weights

2. **ModelCheckpoint** (`callbacks/checkpoint.py`)
   - Save by steps, epochs, or best metric
   - Automatic old checkpoint cleanup
   - Best model tracking with symlinks

3. **LearningRateScheduler** (`callbacks/lr_scheduler.py`)
   - Step-wise learning rate updates
   - Support for all scheduler types
   - Learning rate logging

4. **MLflowCallback** (`callbacks/mlflow_callback.py`)
   - Automatic run management
   - Metric and artifact logging
   - Model registry integration
   - Hyperparameter tracking

5. **ProgressBar** (`callbacks/progress.py`)
   - Rich console output with live metrics
   - Training speed tracking
   - ETA calculation

6. **MetricsLogger** (`callbacks/metrics.py`)
   - Comprehensive metric collection
   - History tracking and export
   - Multiple output formats

### Kaggle-Specific Features

#### KaggleTrainer (`kaggle/trainer.py`)
Extends base trainer with competition features:

1. **Cross-Validation**
   - K-fold, stratified, group, time series splits
   - Out-of-fold (OOF) prediction tracking
   - Per-fold model saving
   - CV metric aggregation

2. **Ensemble Methods**
   - Voting: Hard/soft voting for classification
   - Blending: Weighted average of predictions
   - Stacking: Meta-model on base predictions
   - Multi-seed training for diversity

3. **Advanced Techniques**
   - Pseudo-labeling: Use high-confidence predictions
   - Test-time augmentation: Multiple predictions per sample
   - Adversarial validation: Detect distribution shift
   - Feature engineering hooks

4. **Competition Integration**
   - Direct submission file generation
   - Leaderboard score tracking in MLflow
   - Competition metadata logging
   - Submission message templates

### Metrics System

#### Classification Metrics (`metrics/classification.py`)
- Accuracy, precision, recall, F1
- AUC-ROC, AUC-PR
- Confusion matrix
- Per-class metrics

#### Regression Metrics (`metrics/regression.py`)
- MSE, MAE, RMSE
- R², explained variance
- Mean absolute percentage error
- Quantile loss

#### Loss Tracking (`metrics/loss.py`)
- Averaged loss computation
- Batch and epoch aggregation
- Multiple loss component tracking

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

## Comprehensive API Reference

### Factory Functions

#### `create_trainer(model, config=None, **kwargs)`
Create a trainer with flexible configuration options.

**Parameters:**
- `model`: MLX model instance
- `config`: String preset name, path to config file, or config object
- `**kwargs`: Override any configuration setting

**Returns:** Configured trainer instance

**Example:**
```python
trainer = create_trainer(
    model,
    config="production",
    learning_rate=1e-5,
    num_epochs=20
)
```

#### `create_kaggle_trainer(model, competition, config=None, **kwargs)`
Create a Kaggle-specific trainer with competition optimizations.

**Parameters:**
- `model`: MLX model instance
- `competition`: Competition name or identifier
- `config`: Configuration (defaults to competition preset)
- `**kwargs`: Additional overrides

**Returns:** KaggleTrainer instance

#### `get_trainer_config(name_or_path)`
Load a trainer configuration.

**Parameters:**
- `name_or_path`: Preset name or file path

**Returns:** TrainerConfig object

### Trainer Classes

#### BaseTrainer

**Methods:**

##### `train(train_dataloader, val_dataloader=None, num_epochs=None)`
Main training loop.

**Parameters:**
- `train_dataloader`: Training data loader
- `val_dataloader`: Optional validation data loader
- `num_epochs`: Override configured epochs

**Returns:** `TrainingResult` with metrics and model path

##### `evaluate(dataloader, prefix="eval")`
Evaluate model on a dataset.

**Parameters:**
- `dataloader`: Evaluation data loader
- `prefix`: Metric name prefix

**Returns:** Dictionary of metrics

##### `predict(dataloader, return_logits=False)`
Generate predictions.

**Parameters:**
- `dataloader`: Prediction data loader
- `return_logits`: Return raw logits instead of probabilities

**Returns:** Array of predictions

##### `save_checkpoint(path=None, save_optimizer=True)`
Save training checkpoint.

**Parameters:**
- `path`: Save path (auto-generated if None)
- `save_optimizer`: Include optimizer state

##### `load_checkpoint(path, load_optimizer=True)`
Load training checkpoint.

**Parameters:**
- `path`: Checkpoint path
- `load_optimizer`: Load optimizer state

#### KaggleTrainer

Extends BaseTrainer with competition features.

**Additional Methods:**

##### `train_with_cv(train_dataloader, cv_strategy="stratified", n_folds=5)`
Train with cross-validation.

**Parameters:**
- `train_dataloader`: Full training data
- `cv_strategy`: "kfold", "stratified", "group", "timeseries"
- `n_folds`: Number of CV folds

**Returns:** `CVResult` with OOF predictions and fold metrics

##### `create_submission(test_dataloader, use_tta=False, tta_rounds=5)`
Generate competition submission.

**Parameters:**
- `test_dataloader`: Test data loader
- `use_tta`: Enable test-time augmentation
- `tta_rounds`: Number of TTA rounds

**Returns:** Path to submission file

##### `train_ensemble(train_dataloader, n_models=5, strategy="voting")`
Train ensemble of models.

**Parameters:**
- `train_dataloader`: Training data
- `n_models`: Number of models in ensemble
- `strategy`: "voting", "blending", "stacking"

**Returns:** `EnsembleResult` with ensemble model

### Configuration Classes

#### OptimizerConfig
```python
OptimizerConfig(
    type="adamw",  # adamw, adam, sgd, lion, adafactor
    learning_rate=2e-5,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    momentum=0.9,  # For SGD
    nesterov=False  # For SGD
)
```

#### SchedulerConfig
```python
SchedulerConfig(
    type="cosine",  # linear, cosine, exponential, reduce_on_plateau
    warmup_steps=500,
    warmup_ratio=0.1,  # Alternative to warmup_steps
    min_lr=1e-6,
    gamma=0.95,  # For exponential
    patience=3,  # For reduce_on_plateau
    factor=0.5  # For reduce_on_plateau
)
```

#### DataConfig
```python
DataConfig(
    train_batch_size=32,
    eval_batch_size=64,
    num_workers=8,
    prefetch_size=4,
    drop_last=False,
    pin_memory=False
)
```

#### TrainingConfig
```python
TrainingConfig(
    num_epochs=10,
    gradient_accumulation_steps=1,
    gradient_clip_value=1.0,
    eval_strategy="epoch",  # epoch, steps, no
    eval_steps=500,
    save_strategy="best",  # best, epoch, steps
    save_steps=1000,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)
```

#### KaggleConfig
```python
KaggleConfig(
    competition_name="titanic",
    cv_strategy="stratified",
    n_folds=5,
    ensemble_size=5,
    ensemble_strategy="voting",
    enable_pseudo_labeling=False,
    pseudo_label_threshold=0.95,
    enable_tta=True,
    tta_rounds=5,
    submit_best_fold=False  # Submit best CV fold instead of ensemble
)
```

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

## Usage Examples

### Custom Training Loop
```python
from training import BaseTrainer
from training.callbacks import Callback

class CustomTrainer(BaseTrainer):
    def training_step(self, batch, state):
        # Custom training logic
        loss, grads = self.compute_loss_and_gradients(batch)
        
        # Custom gradient processing
        grads = self.process_gradients(grads)
        
        # Update
        self.optimizer.update(self.model, grads)
        return loss
```

### Advanced Kaggle Pipeline
```python
from training import create_kaggle_trainer
from training.kaggle import PseudoLabelingConfig, EnsembleConfig

# Configure advanced features
pseudo_config = PseudoLabelingConfig(
    unlabeled_data=unlabeled_loader,
    threshold=0.95,
    update_frequency=3  # Update every 3 epochs
)

ensemble_config = EnsembleConfig(
    n_models=5,
    strategy="stacking",
    meta_model="ridge",
    diversity_method="different_seeds"
)

# Create trainer
trainer = create_kaggle_trainer(
    model,
    competition="titanic",
    pseudo_labeling=pseudo_config,
    ensemble=ensemble_config
)

# Train with advanced features
result = trainer.train_advanced(train_loader)
```

### Multi-Stage Training
```python
# Stage 1: Pre-train on large dataset
trainer = create_trainer(model, config="pretrain")
trainer.train(large_dataset_loader)

# Stage 2: Fine-tune on competition data
trainer.update_config(
    learning_rate=1e-6,
    num_epochs=5
)
trainer.train(competition_loader)

# Stage 3: Final optimization
trainer.update_config(
    learning_rate=1e-7,
    early_stopping_patience=10
)
final_result = trainer.train(competition_loader)
```

## Best Practices

### Configuration Management
1. **Version Control Configs**: Keep configs in git for reproducibility
2. **Use Hierarchical Configs**: Base configs with competition-specific overrides
3. **Document Custom Settings**: Add comments explaining non-standard choices

### Training Strategy
1. **Start Simple**: Begin with standard settings, add complexity gradually
2. **Use Validation Set**: Always validate hyperparameters before CV
3. **Monitor Metrics**: Watch for overfitting, especially with small datasets
4. **Ensemble Wisely**: Ensemble diverse models, not just random seeds

### Kaggle Competitions
1. **Always Use CV**: Even for initial experiments
2. **Save OOF Predictions**: Useful for stacking and analysis
3. **Track Everything**: Use MLflow to compare all experiments
4. **Blend Submissions**: Combine different approaches for robustness

### Performance Optimization
1. **Profile First**: Identify bottlenecks before optimizing
2. **Batch Size**: Use largest that fits, then gradient accumulation
3. **Data Loading**: Ensure data loading doesn't bottleneck training
4. **Checkpointing**: Balance frequency with I/O overhead

## Troubleshooting

### Common Issues

**Training Loss Not Decreasing**
- Check learning rate (try 10x smaller/larger)
- Verify data loading and preprocessing
- Ensure model outputs match expected format
- Check for gradient clipping being too aggressive

**Out of Memory**
- Reduce batch size
- Enable gradient accumulation
- Use gradient checkpointing
- Consider LoRA or quantization

**Slow Training**
- Increase number of data workers
- Enable prefetching
- Profile data loading vs model forward/backward
- Check for unnecessary metric computations

**Poor Validation Performance**
- Check for data leakage
- Verify train/val split is appropriate
- Try different regularization (dropout, weight decay)
- Consider different model architecture

## Advanced Topics

### Custom Metrics
```python
from training.metrics import Metric

class CustomMetric(Metric):
    def update(self, predictions, targets):
        # Update internal state
        pass
    
    def compute(self):
        # Return final metric value
        pass
    
    def reset(self):
        # Reset for new epoch
        pass
```

### Dynamic Training
```python
from training.callbacks import Callback

class DynamicLRCallback(Callback):
    def on_epoch_end(self, trainer, state):
        # Adjust learning rate based on metrics
        if state.metrics["val_loss"] > self.threshold:
            trainer.optimizer.learning_rate *= 0.5
```

### Experiment Tracking
```python
from training.integrations import create_mlflow_callback

mlflow_cb = create_mlflow_callback(
    experiment_name="bert-experiments",
    tags={"version": "v2", "dataset": "titanic"},
    log_models=True,
    log_artifacts=["configs/", "predictions/"]
)
```