# Monitoring Adapters

This package provides monitoring adapters that implement the `MonitoringPort` interface from the domain layer. These adapters enable tracking of training progress, metrics, hyperparameters, and artifacts across different monitoring backends.

## Available Adapters

### 1. Console Adapter (`ConsoleMonitoringAdapter`)
Beautiful terminal output using Rich (with fallback to plain text).

**Features:**
- Progress bars with real-time metrics
- Formatted tables for hyperparameters
- Color-coded log levels
- Configurable verbosity (quiet, normal, verbose)

**Usage:**
```python
from adapters.secondary.monitoring import ConsoleMonitoringAdapter

# Create adapter
adapter = ConsoleMonitoringAdapter(
    verbosity="normal",  # quiet, normal, verbose
    use_rich=True       # Use Rich for beautiful output
)

# Start a run
run_id = adapter.start_run("my_training", tags={"model": "bert"})

# Log metrics
adapter.log_metrics({"loss": 1.5, "accuracy": 0.85}, step=100)

# Create progress bar
progress = adapter.create_progress_bar(1000, "Training", "steps")
for i in range(1000):
    progress.update(1)
    progress.set_postfix(loss=1.5 - i*0.001)
progress.close()

# End run
adapter.end_run("SUCCESS")
```

### 2. MLflow Adapter (`MLflowMonitoringAdapter`)
Integration with MLflow for experiment tracking and model registry.

**Features:**
- Experiment tracking with run management
- Hyperparameter and metric logging
- Artifact storage
- Model registry support
- Nested runs for complex experiments
- Run comparison and metric history

**Usage:**
```python
from adapters.secondary.monitoring import MLflowMonitoringAdapter, MLflowConfig

# Configure MLflow
config = MLflowConfig(
    tracking_uri="http://localhost:5000",  # Optional
    experiment_name="k-bert-experiments",
    log_models=True,
    register_models=False
)

# Create adapter
adapter = MLflowMonitoringAdapter(config)

# Use same interface as other adapters
run_id = adapter.start_run("experiment_1")
adapter.log_hyperparameters({"learning_rate": 0.001})
adapter.log_metrics({"loss": 1.2}, step=0)
adapter.end_run()
```

### 3. TensorBoard Adapter (`TensorBoardMonitoringAdapter`)
TensorBoard logging for visualization.

**Features:**
- Scalar metrics logging
- Histogram support for distributions
- Text logging for messages
- Hyperparameter tracking
- Fallback to JSON logging if TensorBoard not installed

**Usage:**
```python
from adapters.secondary.monitoring import TensorBoardMonitoringAdapter

# Create adapter
adapter = TensorBoardMonitoringAdapter(
    log_dir="./runs",
    tag_prefix="experiment"  # Optional prefix for all tags
)

# Log training session
adapter.start_run("training_run")
adapter.log_metrics({"train/loss": 1.5, "train/acc": 0.8}, step=0)
adapter.log_training_session(session)  # Log complete session info
adapter.end_run()
```

### 4. Composite Adapter (`MultiMonitorAdapter`)
Use multiple monitoring backends simultaneously.

**Features:**
- Dispatch operations to multiple adapters
- Graceful failure handling
- Composite progress bars
- Flexible adapter combination

**Usage:**
```python
from adapters.secondary.monitoring import (
    MultiMonitorAdapter,
    ConsoleMonitoringAdapter,
    MLflowMonitoringAdapter,
    TensorBoardMonitoringAdapter
)

# Create individual adapters
console = ConsoleMonitoringAdapter(verbosity="normal")
mlflow = MLflowMonitoringAdapter()
tensorboard = TensorBoardMonitoringAdapter(log_dir="./runs")

# Combine them
adapter = MultiMonitorAdapter(
    adapters=[console, mlflow, tensorboard],
    fail_fast=False  # Continue even if one adapter fails
)

# Use as normal - all adapters will receive the calls
adapter.start_run("multi_backend_run")
adapter.log_metrics({"loss": 1.2})
adapter.end_run()
```

## Factory Function

Use the factory function for easy adapter creation:

```python
from adapters.secondary.monitoring import create_monitoring_adapter

# Create adapters by type
console = create_monitoring_adapter("console", verbosity="verbose")
mlflow = create_monitoring_adapter("mlflow", config=MLflowConfig())
tensorboard = create_monitoring_adapter("tensorboard", log_dir="./tb_logs")
multi = create_monitoring_adapter("multi", adapters=[console, mlflow])
```

## Domain Integration

All adapters implement the `MonitoringPort` interface:

```python
from domain.ports.monitoring import MonitoringPort
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.entities.training import TrainingSession

def train_model(monitor: MonitoringPort):
    """Example training function using monitoring port."""
    # Start monitoring
    run_id = monitor.start_run("training")
    
    # Log hyperparameters
    monitor.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 32
    })
    
    # Training loop
    for epoch in range(num_epochs):
        # Log training metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=global_step,
            loss=loss_value,
            learning_rate=current_lr
        )
        monitor.log_training_metrics(metrics)
        
        # Log evaluation metrics
        eval_metrics = EvaluationMetrics(
            dataset_name="validation",
            split="val",
            loss=val_loss,
            accuracy=val_acc
        )
        monitor.log_evaluation_metrics(eval_metrics)
    
    # End monitoring
    monitor.end_run("SUCCESS")
```

## Progress Bars

All adapters support progress bars through the `ProgressBarPort` interface:

```python
# Create progress bar
progress = adapter.create_progress_bar(
    total=1000,
    description="Processing",
    unit="items"
)

# Update progress
for i in range(1000):
    progress.update(1)
    progress.set_postfix(processed=i+1, rate=f"{i/10:.1f}/s")

# Close when done
progress.close()
```

## Best Practices

1. **Choose the right adapter**: 
   - Console for development and debugging
   - MLflow for experiment tracking
   - TensorBoard for visualization
   - Multi for production with multiple backends

2. **Handle missing dependencies gracefully**:
   ```python
   try:
       adapter = MLflowMonitoringAdapter()
   except ImportError:
       # Fallback to console
       adapter = ConsoleMonitoringAdapter()
   ```

3. **Use structured metrics**:
   - Use consistent metric names with prefixes (e.g., "train/loss", "val/accuracy")
   - Log metrics at regular intervals
   - Include both raw and derived metrics

4. **Leverage domain entities**:
   - Use `TrainingMetrics` and `EvaluationMetrics` for structured logging
   - Log complete `TrainingSession` objects for comprehensive tracking

5. **Configure for your environment**:
   - Development: Console adapter with verbose output
   - Experiments: MLflow with experiment tracking
   - Production: Multi adapter with console + persistent backend