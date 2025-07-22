# k-bert Error Handling Framework

A comprehensive error handling framework for k-bert that provides rich error context, recovery strategies, and seamless integration with the CLI and logging systems.

## Features

- **Rich Error Context**: Capture detailed debugging information, stack traces, and structured error data
- **Error Hierarchy**: Well-organized error types for different components (Config, Model, Data, Training, etc.)
- **Recovery Strategies**: Automatic error recovery with retry, resource reduction, and checkpoint recovery
- **CLI Integration**: Beautiful error formatting for the command-line interface
- **Handler Registry**: Extensible system for custom error handling
- **Error Groups**: Handle multiple errors in batch operations
- **Fluent Interface**: Chain error context, suggestions, and recovery actions

## Quick Start

### Basic Usage

```python
from core.errors import ConfigurationError, DataError, with_error_handling

# Raise errors with context
raise ConfigurationError(
    "Invalid configuration value",
    config_path=Path("config.yaml"),
    field_path="training.batch_size",
    invalid_value=-1,
).with_suggestion(
    "Batch size must be a positive integer"
).with_recovery(
    "Use default batch size of 32"
)

# Automatic error wrapping with decorators
@with_error_handling(
    error_type=DataError,
    error_message="Failed to load dataset"
)
def load_data(path: Path):
    # Exceptions are automatically wrapped in DataError
    return pd.read_csv(path)
```

### Error Recovery

```python
from core.errors import with_recovery, RetryStrategy, attempt_recovery

# Automatic retry with decorator
@with_recovery(strategies=[RetryStrategy(max_attempts=3)])
def fetch_data(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Manual recovery
try:
    train_model(config)
except TrainingError as e:
    result = attempt_recovery(e, context={"config": config})
    if result.success:
        # Apply modifications and retry
        config.update(result.modifications)
        train_model(config)
```

## Error Types

### ConfigurationError
For configuration-related issues:
```python
# Missing required field
error = ConfigurationError.missing_required_field(
    "models.bert.hidden_size",
    config_path=Path("config.yaml")
)

# Invalid value
error = ConfigurationError.invalid_value(
    "training.batch_size",
    value="invalid",
    expected="int",
    config_path=Path("config.yaml")
)
```

### ModelError
For model creation and loading issues:
```python
# Unsupported model type
error = ModelError.unsupported_model_type("unknown_bert")

# Checkpoint not found
error = ModelError.checkpoint_not_found(Path("checkpoint.safetensors"))

# Incompatible checkpoint
error = ModelError.incompatible_checkpoint(
    checkpoint_path=Path("checkpoint.safetensors"),
    expected_arch="modernbert",
    found_arch="bert"
)
```

### DataError
For data loading and processing issues:
```python
# File not found
error = DataError.file_not_found(Path("data.csv"))

# Missing column
error = DataError.missing_column(
    "target",
    available_columns=["feature1", "feature2"],
    data_path=Path("data.csv")
)

# Invalid format
error = DataError.invalid_format(
    Path("data.txt"),
    expected_format="csv",
    detected_format="txt"
)
```

### TrainingError
For training-related issues:
```python
# NaN loss detected
error = TrainingError.nan_loss(epoch=5, step=100)

# Out of memory
error = TrainingError.out_of_memory(batch_size=64, model_size="large")
```

### ValidationError
For input validation:
```python
# Value out of range
error = ValidationError.invalid_range(
    "learning_rate",
    value=2.0,
    min_value=0.0,
    max_value=1.0
)
```

### PluginError
For plugin-related issues:
```python
# Plugin not found
error = PluginError.not_found("custom_head", "heads")

# Plugin load failure
error = PluginError.load_failed(
    "custom_head",
    Path("plugin.py"),
    reason="Import error"
)
```

### CLIError
For CLI-specific issues:
```python
# Invalid command
error = CLIError.invalid_command(
    "trian",
    similar=["train", "trial"]
)

# Missing argument
error = CLIError.missing_argument("--config", "train")
```

## Recovery Strategies

### RetryStrategy
Retry with exponential backoff:
```python
strategy = RetryStrategy(
    max_attempts=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=60.0,
    jitter=True
)
```

### ResourceReductionStrategy
Reduce resource usage on OOM:
```python
strategy = ResourceReductionStrategy(
    reduction_factor=0.5,
    min_batch_size=1,
    enable_gradient_accumulation=True
)
```

### CheckpointRecoveryStrategy
Recover from checkpoint after failure:
```python
strategy = CheckpointRecoveryStrategy(
    checkpoint_dir=Path("checkpoints"),
    load_best=True,
    modify_config=True
)
```

### CompositeStrategy
Combine multiple strategies:
```python
strategy = CompositeStrategy([
    RetryStrategy(max_attempts=2),
    ResourceReductionStrategy(),
    CheckpointRecoveryStrategy()
])
```

## Error Handlers

### Register Custom Handlers

```python
from core.errors import register_handler, register_type_handler

# Register handler class
class CustomHandler(ErrorHandler):
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, CustomError)
    
    def handle(self, error: Exception) -> Optional[Any]:
        # Custom handling logic
        return None

register_handler(CustomHandler(), priority=50)

# Register handler function
@register_type_handler(ModelError)
def handle_model_error(error: ModelError):
    # Log to monitoring system
    monitoring.send_event("model_error", error.to_dict())
    return None  # Continue with default handling
```

### Error Context Manager

```python
from core.errors import ErrorHandlingContext

# Temporarily modify error handling
with ErrorHandlingContext(
    suppress=[ValueError],  # Suppress these errors
    transform={TypeError: ValidationError}  # Transform these errors
):
    # Code with modified error handling
    process_data()
```

## Integration Examples

### CLI Integration

```python
from core.errors import CLIError, setup_default_handlers
from rich.console import Console

console = Console()
setup_default_handlers(console)

@app.command()
def train(config: Path):
    try:
        # Training logic
        pass
    except Exception as e:
        raise CLIError.from_exception(
            e,
            "Training failed"
        ).with_suggestion(
            "Check the logs for details"
        )
```

### Configuration Validation

```python
@with_error_handling(
    error_type=ConfigurationError,
    error_message="Configuration validation failed"
)
def validate_config(config: Dict[str, Any]):
    required_fields = [
        ("models.type", str),
        ("training.epochs", int),
        ("data.path", str)
    ]
    
    for field_path, expected_type in required_fields:
        value = get_nested_value(config, field_path)
        
        if value is None:
            raise ConfigurationError.missing_required_field(field_path)
        
        if not isinstance(value, expected_type):
            raise ConfigurationError.invalid_value(
                field_path,
                value,
                expected_type.__name__
            )
```

### Error Groups for Batch Operations

```python
def process_files(paths: List[Path]):
    results = []
    errors = []
    
    for path in paths:
        try:
            result = process_file(path)
            results.append(result)
        except Exception as e:
            errors.append(
                DataError.from_exception(e, f"Failed to process {path}")
            )
    
    if errors:
        raise ErrorGroup(
            f"Failed to process {len(errors)} files",
            errors
        )
    
    return results
```

## Best Practices

1. **Use Specific Error Types**: Choose the most specific error type for your use case
2. **Add Context**: Always add relevant context information for debugging
3. **Provide Suggestions**: Help users understand how to fix the error
4. **Define Recovery Actions**: When possible, provide automatic recovery options
5. **Chain Fluently**: Use the fluent interface to build rich error objects
6. **Handle at Boundaries**: Handle errors at system boundaries (CLI, API, etc.)
7. **Log Appropriately**: Errors are automatically logged with appropriate severity

## Testing

Test error scenarios in your code:

```python
import pytest
from core.errors import ConfigurationError

def test_config_validation():
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config({"invalid": "config"})
    
    error = exc_info.value
    assert error.error_code == "CONFIG_MISSING_FIELD"
    assert len(error.context.suggestions) > 0
```

## Advanced Usage

### Custom Error Context

```python
error = KBertError("Complex error")
error.context.add_technical_detail("gpu_memory", "16GB")
error.context.add_technical_detail("model_params", "8.5B")
error.context.add_related_error(original_exception)
```

### Conditional Recovery

```python
class ConditionalRecovery(RecoveryStrategy):
    def can_recover(self, error: Exception) -> bool:
        # Only recover during business hours
        hour = datetime.now().hour
        return 9 <= hour <= 17
    
    def recover(self, error: Exception, context: Dict[str, Any]):
        # Recovery logic
        pass
```

### Error Serialization

```python
# Convert to dict for API responses
error_data = error.to_dict()

# Format for CLI display
cli_output = error.format_for_cli(verbose=True)

# JSON serialization
import json
json_data = json.dumps(error_data)
```

## Contributing

When adding new error types:

1. Extend the appropriate base class
2. Define specific factory methods for common cases
3. Add appropriate error codes
4. Include suggestions and recovery actions
5. Add tests for the new error type
6. Update this documentation