# Advanced Logging with Loguru

This guide covers the advanced Loguru features available in the MLX BERT training pipeline.

## Table of Contents

1. [Structured Logging with Context](#structured-logging-with-context)
2. [Performance Timing](#performance-timing)
3. [Lazy Evaluation](#lazy-evaluation)
4. [Metrics Logging](#metrics-logging)
5. [Error Handling](#error-handling)
6. [Progress Tracking](#progress-tracking)
7. [Frequency-Based Logging](#frequency-based-logging)
8. [MLX Array Debugging](#mlx-array-debugging)
9. [Best Practices](#best-practices)

## Structured Logging with Context

Bind context values to logger instances for consistent structured logging:

```python
from utils.loguru_advanced import bind_training_context

# Create logger with training context
log = bind_training_context(epoch=1, step=100, fold=0, phase="train")
log.info("Training step completed", loss=0.5, accuracy=0.92)
```

Output includes all context values automatically:
```
2024-01-21 10:15:32 | INFO | epoch:1 step:100 fold:0 phase:train - Training step completed | loss: 0.5 | accuracy: 0.92
```

## Performance Timing

Automatically time operations with context managers:

```python
from utils.loguru_advanced import log_timing

# Basic timing
with log_timing("model_forward_pass"):
    outputs = model(inputs)

# With memory tracking
with log_timing("training_step", include_memory=True, batch_size=32):
    loss = train_step(batch)
```

## Lazy Evaluation

Avoid expensive computations in debug logs when not needed:

```python
from utils.loguru_advanced import lazy_debug

# This function only runs if DEBUG level is active
lazy_debug(
    "Gradient statistics",
    lambda: compute_gradient_stats(model),  # Expensive computation
    step=current_step
)
```

## Metrics Logging

Log metrics in both human-readable and structured formats:

```python
from utils.loguru_advanced import MetricsLogger

# Create metrics logger with JSON output
metrics_logger = MetricsLogger(sink_path="output/metrics.jsonl")

# Log metrics
metrics_logger.log_metrics(
    {"loss": 0.45, "accuracy": 0.89, "f1": 0.86},
    step=1000,
    epoch=5
)
```

## Error Handling

Decorator for consistent error logging with context:

```python
from utils.loguru_advanced import catch_and_log

@catch_and_log(
    ValueError,
    "Model loading failed",
    reraise=True,
    model_path=checkpoint_path
)
def load_model(path):
    # Model loading logic
    pass
```

## Progress Tracking

Enhanced progress tracking with automatic rate calculation:

```python
from utils.loguru_advanced import ProgressTracker

with ProgressTracker(
    total=len(dataloader),
    desc="Training epoch",
    log_frequency=10  # Log every 10%
) as tracker:
    for batch in dataloader:
        loss = train_step(batch)
        tracker.update(1, loss=loss, lr=get_lr())
```

## Frequency-Based Logging

Log repetitive events only at intervals:

```python
from utils.loguru_advanced import FrequencyLogger

freq_logger = FrequencyLogger(frequency=100)

for step in range(1000):
    # This only logs every 100 steps
    freq_logger.log(
        "gradient_check",
        "Gradient norm",
        norm=compute_norm(),
        step=step
    )
```

## MLX Array Debugging

Debug MLX arrays with detailed information:

```python
from utils.loguru_advanced import log_mlx_info

# Log array information (only in DEBUG mode)
log_mlx_info(model_output, "model_predictions")
log_mlx_info(gradients, "parameter_gradients")
```

## Best Practices

### 1. Use Context Binding for Consistency

Always bind relevant context at the beginning of operations:

```python
# Good
log = bind_training_context(epoch=epoch, phase="train")
log.info("Starting training")

# Avoid
logger.info(f"Starting training for epoch {epoch}")
```

### 2. Leverage Lazy Evaluation

Use lazy evaluation for expensive debug computations:

```python
# Good - only computes if DEBUG is enabled
lazy_debug("Stats", lambda: expensive_computation())

# Avoid - always computes
logger.debug(f"Stats: {expensive_computation()}")
```

### 3. Structure Your Metrics

Use MetricsLogger for consistent metric formatting:

```python
# Good
metrics_logger.log_metrics(
    {"loss": loss, "acc": acc},
    step=step,
    epoch=epoch
)

# Avoid
logger.info(f"Step {step}: loss={loss}, acc={acc}")
```

### 4. Time Critical Operations

Use log_timing for performance monitoring:

```python
# Good
with log_timing("data_loading", batch_size=32):
    batch = next(dataloader)

# More detailed
with log_timing("forward_pass", include_memory=True) as timer:
    output = model(input)
    timer.debug(f"Output shape: {output.shape}")
```

### 5. Handle Errors Gracefully

Use catch_and_log for consistent error handling:

```python
@catch_and_log(
    Exception,
    "Operation failed",
    reraise=False,
    default=None
)
def risky_operation():
    # Implementation
    pass
```

### 6. Configure Sinks Appropriately

Set up different sinks for different purposes:

```python
# Console output - human readable
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)

# File output - detailed with rotation
logger.add(
    "logs/training.log",
    rotation="500 MB",
    retention="30 days",
    level="DEBUG"
)

# Metrics sink - structured JSON
logger.add(
    "logs/metrics.jsonl",
    serialize=True,
    filter=lambda record: record["extra"].get("metrics", False)
)
```

## Integration Example

Here's how to integrate multiple features in a training loop:

```python
from utils.loguru_advanced import *

# Setup
metrics_logger = MetricsLogger("output/metrics.jsonl")
freq_logger = FrequencyLogger(frequency=100)

# Training loop
for epoch in range(num_epochs):
    epoch_log = bind_training_context(epoch=epoch, phase="train")
    
    with log_timing(f"epoch_{epoch}", epoch=epoch):
        with ProgressTracker(len(train_loader), "Training") as tracker:
            for step, batch in enumerate(train_loader):
                step_log = bind_training_context(
                    epoch=epoch,
                    step=step,
                    phase="train"
                )
                
                # Forward pass with timing
                with log_timing("forward", level="DEBUG"):
                    loss = model.train_step(batch)
                
                # Update progress
                tracker.update(1, loss=loss)
                
                # Log metrics periodically
                if step % 100 == 0:
                    metrics_logger.log_metrics(
                        {"loss": loss},
                        step=epoch * len(train_loader) + step,
                        epoch=epoch
                    )
                
                # Frequency-based gradient logging
                freq_logger.log(
                    "gradients",
                    "Gradient check",
                    max_norm=get_grad_norm()
                )
```

## Environment Variables

Control logging behavior with environment variables:

```bash
# Set minimum log level
export LOGURU_LEVEL=DEBUG

# Disable color output
export LOGURU_COLORIZE=False

# Custom format
export LOGURU_FORMAT="{time} | {level} | {message}"
```

## Performance Considerations

1. **Lazy Evaluation**: Always use `lazy_debug()` for expensive computations
2. **Frequency Logging**: Use `FrequencyLogger` for high-frequency events
3. **Async Logging**: Use `enqueue=True` for file handlers
4. **Conditional Binding**: Only bind context when needed

```python
# Good - conditional binding
if logger._core.min_level <= 10:  # DEBUG
    debug_log = bind_context(detailed=True)
    debug_log.debug("Detailed information")

# Avoid - always binds
log = bind_context(detailed=True)
log.debug("Maybe not shown")
```

## Troubleshooting

### Issue: Too Much Log Output

Solution: Use frequency logging and appropriate levels:

```python
# Reduce noise
freq_logger = FrequencyLogger(frequency=1000)
freq_logger.log("training", "Still training...")
```

### Issue: Performance Impact

Solution: Use lazy evaluation and async handlers:

```python
# File handler with async
logger.add("file.log", enqueue=True)

# Lazy computation
lazy_debug("Stats", expensive_function)
```

### Issue: Missing Context

Solution: Ensure context is bound at appropriate scope:

```python
# Bind at the right level
with bind_context(request_id=123):
    # All logs in this block have request_id
    process_request()
```

---

For more examples, see `examples/advanced_logging_demo.py`.