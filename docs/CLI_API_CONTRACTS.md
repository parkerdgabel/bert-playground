# CLI API Contracts Documentation

This document defines the API contracts between CLI commands and utility functions. These contracts must be maintained to ensure CLI stability.

## Core Principles

1. **Backward Compatibility**: Changes to utility functions must maintain backward compatibility
2. **Explicit Contracts**: All interfaces should be explicitly documented
3. **Version Management**: Breaking changes require version bumps and migration paths
4. **Type Safety**: Use type hints and runtime validation where possible

## Contract Definitions

### Core Commands

#### Train Command

**Depends on:**
- `TrainerV2` class
- `create_model` factory function
- `TitanicDataModule` class

**Contract:**
```python
# TrainerV2 initialization
trainer = TrainerV2(
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    eval_interval: int,
    checkpoint_dir: Path,
    log_interval: int,
    grad_accum_steps: int,
    mixed_precision: bool,
    compile_model: bool,
    experiment_name: str,
    enable_mlflow: bool
)

# TrainerV2.train method
trainer.train(
    model: nn.Module,  # Must have forward method returning dict with 'loss'
    data_module: DataModule  # Must have train_dataloader and val_dataloader
)

# create_model function
model = create_model(
    model_type: str,  # One of: "modernbert", "bert_core", "bert_with_head"
    **kwargs
) -> nn.Module
```

#### Predict Command

**Depends on:**
- `load_checkpoint` function
- Model inference interface

**Contract:**
```python
# load_checkpoint function
model = load_checkpoint(
    checkpoint_path: Union[str, Path]
) -> nn.Module

# Model inference
outputs = model(
    input_ids: mx.array,      # Shape: [batch_size, seq_length]
    attention_mask: mx.array  # Shape: [batch_size, seq_length]
) -> Dict[str, mx.array]
# Must contain 'predictions' key
```

#### Benchmark Command

**Depends on:**
- `create_bert_with_head` function
- MLX `nn.value_and_grad` API

**Contract:**
```python
# create_bert_with_head function
model = create_bert_with_head(
    config: BertConfig,
    head_type: HeadType,
    num_labels: int
) -> nn.Module

# MLX gradient computation
def loss_fn():
    outputs = model(input_ids=..., attention_mask=..., labels=...)
    return outputs['loss']  # Must return scalar

value_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss_value, grads = value_and_grad_fn()  # No arguments!
```

### Kaggle Commands

#### Competitions Command

**Depends on:**
- `KaggleIntegration.list_competitions` method

**Contract:**
```python
# list_competitions method
competitions_df = kaggle.list_competitions(
    category: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "latestDeadline",
    page: int = 1  # NOT page_size!
) -> pd.DataFrame

# DataFrame must contain:
# - id: str
# - title: str
# - deadline: Union[str, pd.Timestamp]
# - numTeams: int
# - reward: Optional[str]
# - isCompleted: bool
```

#### Download Command

**Depends on:**
- `KaggleIntegration.download_competition` method

**Contract:**
```python
# download_competition method
files = kaggle.download_competition(
    competition_id: str,
    path: Path
) -> List[str]  # List of downloaded file names
```

#### Submit Command

**Depends on:**
- `KaggleIntegration.submit_predictions` method

**Contract:**
```python
# submit_predictions method
result = kaggle.submit_predictions(
    competition_id: str,
    submission_file: Path,
    message: str
) -> Dict[str, Any]

# Result must contain:
# - status: str
# - publicScore: Optional[float]
# - submissionId: str
```

### MLflow Commands

#### Health Command

**Depends on:**
- `MLflowHealthChecker.run_full_check` method

**Contract:**
```python
# run_full_check method - NO PARAMETERS!
results = health_checker.run_full_check() -> Dict[str, Dict[str, Any]]

# Each check result must have:
# - status: Literal["PASS", "FAIL"]
# - message: str
# - suggestions: Optional[List[str]]  # Only if status == "FAIL"
```

#### Experiments Command

**Depends on:**
- `MLflowCentral.get_all_experiments` method
- `MLflowCentral.get_experiment_runs` method

**Contract:**
```python
# get_all_experiments method
experiments = mlflow_central.get_all_experiments() -> List[Dict[str, Any]]

# Each experiment must have:
# - experiment_id: str
# - name: str
# - artifact_location: str
# - lifecycle_stage: str

# get_experiment_runs method
runs_df = mlflow_central.get_experiment_runs(
    experiment_id: str
) -> pd.DataFrame
```

### Model Commands

#### Inspect Command

**Depends on:**
- `load_checkpoint` function with metadata
- Model introspection interface

**Contract:**
```python
# load_checkpoint with metadata
model, metadata = load_checkpoint(
    checkpoint_path: Union[str, Path]
) -> Tuple[nn.Module, Dict[str, Any]]

# Model introspection
parameters = dict(model.parameters())  # name -> mx.array mapping
modules = model.leaf_modules()  # name -> module mapping
```

#### Convert Command

**Depends on:**
- `load_checkpoint` function
- `save_checkpoint` function

**Contract:**
```python
# save_checkpoint function
save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    metadata: Dict[str, Any]
) -> None
```

## Validation Requirements

### Input Validation

All commands should validate inputs using these utilities:

```python
from cli.utils.validation import (
    validate_file_exists,    # Raises typer.BadParameter if not exists
    validate_directory,      # Raises typer.BadParameter if not exists
    validate_model_type,     # Validates against allowed model types
    validate_positive_int,   # Ensures value > 0
    validate_learning_rate   # Ensures 0 < lr < 1
)
```

### Error Handling

All commands must use the `@handle_errors` decorator which:
1. Catches all exceptions
2. Formats error messages consistently
3. Exits with code 1 on error
4. Logs errors if logging is configured

```python
from cli.utils import handle_errors

@handle_errors
def command_function(...):
    # Command implementation
```

## Breaking Change Protocol

When a breaking change is necessary:

1. **Deprecation Warning**: Add warning in current version
2. **Migration Guide**: Document how to update
3. **Grace Period**: Maintain both APIs for 2 minor versions
4. **Version Bump**: Breaking changes require major version bump

Example:
```python
# v1.0 - Original API
def old_function(param1, param2):
    warnings.warn(
        "old_function is deprecated and will be removed in v2.0. "
        "Use new_function instead.",
        DeprecationWarning
    )
    return new_function(param1, param2, default_param3=None)

# v1.0 - New API available
def new_function(param1, param2, param3):
    # New implementation
```

## Testing Requirements

All API contracts must have:

1. **Contract Tests**: Verify interface compatibility (see `test_api_contracts.py`)
2. **Integration Tests**: Test actual functionality
3. **Regression Tests**: Ensure backward compatibility
4. **Mock Tests**: Test error conditions and edge cases

## Monitoring

Track API usage and compatibility:

1. **Version Headers**: Include API version in logs
2. **Deprecation Metrics**: Track deprecated API usage
3. **Error Patterns**: Monitor contract violations
4. **Performance**: Track API response times

## Current API Versions

| Component | Version | Last Changed | Notes |
|-----------|---------|--------------|-------|
| TrainerV2 | 1.0.0 | 2024-01-15 | Stable |
| KaggleIntegration | 1.1.0 | 2024-01-16 | Fixed page parameter |
| MLflowHealthChecker | 1.0.1 | 2024-01-16 | Removed detailed param |
| Model Factory | 1.2.0 | 2024-01-14 | Added modular BERT |

## Future Considerations

1. **API Versioning**: Consider explicit API versioning
2. **Contract Generation**: Auto-generate from type hints
3. **Runtime Validation**: Add pydantic models for contracts
4. **API Documentation**: Generate from contracts