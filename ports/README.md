# Ports - Hexagonal Architecture Boundaries

This directory contains the port definitions for the k-bert project's hexagonal architecture. Ports define the boundaries between the application core and external systems.

## Directory Structure

```
ports/
├── primary/          # Driving ports - APIs exposed to external actors
│   ├── training.py   # Training API (Trainer, TrainingStrategy)
│   ├── commands.py   # Command execution API
│   └── model_management.py  # Model management API
│
└── secondary/        # Driven ports - Interfaces we depend on
    ├── compute.py    # Compute backend (MLX, PyTorch adapters)
    ├── storage.py    # Storage services (filesystem, cloud)
    ├── monitoring.py # Monitoring & logging (loguru, MLflow)
    ├── configuration.py  # Configuration providers
    ├── neural.py     # Neural network operations
    ├── tokenizer.py  # Text tokenization
    ├── optimization.py   # Optimizers & schedulers
    ├── metrics.py    # Metric computation
    └── checkpointing.py  # Checkpoint management
```

## Primary Ports (Driving)

Primary ports are the APIs that external actors (CLI, web UI, etc.) use to interact with the application core.

### Training Port (`primary/training.py`)
- `Trainer`: Main training interface
- `TrainingStrategy`: Different training approaches
- `TrainingState` & `TrainingResult`: Data structures for training info
- High-level functions: `train_model()`, `evaluate_model()`, `predict_with_model()`

### Command Port (`primary/commands.py`)
- `Command`: Individual training operations
- `Pipeline`: Sequences of commands
- `CommandContext` & `CommandResult`: Execution context and results
- Functions: `execute_command()`, `create_pipeline()`

### Model Management Port (`primary/model_management.py`)
- `ModelManager`: Complete model lifecycle management
- `ModelInfo`: Model metadata
- Functions: `save_model()`, `load_model()`, `list_models()`, `delete_model()`

## Secondary Ports (Driven)

Secondary ports are interfaces that the application core depends on. These are implemented by adapters.

### Compute Port (`secondary/compute.py`)
- `ComputeBackend`: Framework-agnostic ML operations
- `NeuralOps`: Higher-level neural network operations
- Type definitions: `Array`, `Shape`, `DataType`

### Storage Port (`secondary/storage.py`)
- `StorageService`: General storage operations
- `ModelStorageService`: Model-specific storage
- Handles persistence to filesystem, cloud, databases

### Monitoring Port (`secondary/monitoring.py`)
- `MonitoringService`: Logging and metrics
- `ExperimentTracker`: ML experiment tracking
- `Timer` & `Span`: Performance monitoring

### Configuration Port (`secondary/configuration.py`)
- `ConfigurationProvider`: Load/save configurations
- `ConfigRegistry`: Multiple configuration sources
- Supports YAML, JSON, environment variables

### Neural Port (`secondary/neural.py`)
- `NeuralBackend`: Neural network building blocks
- Configuration classes: `AttentionConfig`, `FeedForwardConfig`, etc.
- Enums: `ActivationType`, `NormalizationType`, `LossType`

### Tokenizer Port (`secondary/tokenizer.py`)
- `TokenizerPort`: Text tokenization interface
- `TokenizerConfig` & `TokenizerOutput`: Configuration and results
- Supports different tokenizer backends

### Optimization Port (`secondary/optimization.py`)
- `Optimizer`: Parameter optimization
- `LRScheduler`: Learning rate scheduling
- Configuration classes for different optimizers

### Metrics Port (`secondary/metrics.py`)
- `Metric`: Individual metric computation
- `MetricsCollector`: Manage multiple metrics
- `MetricType`: Supported metric types

### Checkpointing Port (`secondary/checkpointing.py`)
- `CheckpointManager`: Save/load training checkpoints
- `CheckpointInfo`: Checkpoint metadata
- Handles checkpoint lifecycle and cleanup

## Usage Example

```python
# External actor (e.g., CLI) uses primary port
from ports.primary import train_model, TrainerConfig
from adapters.mlx import MLXComputeAdapter
from adapters.filesystem import FileSystemStorageAdapter

# Configure dependencies (secondary ports)
compute = MLXComputeAdapter()
storage = FileSystemStorageAdapter()

# Use primary port API
result = train_model(
    model=my_model,
    train_data=train_loader,
    config=TrainerConfig(num_epochs=10),
)
```

## Design Principles

1. **Protocol-based**: All ports use Python Protocol for maximum flexibility
2. **Framework-agnostic**: No direct dependencies on specific frameworks
3. **Clear boundaries**: Primary ports expose APIs, secondary ports define dependencies
4. **Type safety**: Full type annotations with runtime checking
5. **Extensibility**: Easy to add new implementations via adapters

## Adding New Ports

1. Identify if it's primary (API) or secondary (dependency)
2. Create the protocol interface with clear documentation
3. Use `@runtime_checkable` for runtime validation
4. Add to appropriate `__init__.py` for easy imports
5. Update this README with the new port

## Testing Ports

Ports themselves are just interfaces, but you should:
1. Create contract tests that any adapter must pass
2. Use mock implementations for testing the core
3. Ensure all methods are properly documented
4. Validate type annotations with mypy