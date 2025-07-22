# Core Protocols

This module provides a unified location for all protocol definitions used throughout the k-bert codebase. Protocols define the contracts that implementations must follow, enabling clean dependency inversion and testability.

## Structure

The protocols are organized into logical groups:

### base.py
Common protocols used across multiple modules:
- `Configurable`: Objects that can be configured from dictionaries
- `Serializable`: Objects that can be saved/loaded from disk
- `Comparable`: Objects that can be compared for performance

### data.py
Data pipeline protocols:
- `Dataset`: Dataset implementations
- `DataLoader`: Data loaders for batching
- `DataFormat`: File format handlers
- `BatchProcessor`: Batch processing logic
- `Augmenter`: Data augmentation
- `FeatureAugmenter`: Feature-specific augmentation

### models.py
Model-related protocols:
- `Model`: Base model interface
- `Head`: Task-specific head interface
- `ModelConfig`: Model configuration interface

### training.py
Training infrastructure protocols:
- `Trainer`: Training loop implementations
- `TrainerConfig`: Trainer configuration
- `TrainingState`: Training state tracking
- `TrainingResult`: Training results
- `Optimizer`: Optimization algorithms
- `LRScheduler`: Learning rate scheduling
- `TrainingHook`: Training event hooks
- `Callback`: Alternative callback interface
- `Metric`: Metric computation
- `MetricsCollector`: Metric collection and history
- `CheckpointManager`: Checkpoint management

### plugins.py
Plugin system protocols:
- `Plugin`: Base plugin interface
- `PluginMetadata`: Plugin metadata
- `HeadPlugin`: Custom head implementations
- `AugmenterPlugin`: Custom augmentation strategies
- `FeatureExtractorPlugin`: Custom feature extraction
- `DataLoaderPlugin`: Custom data loading
- `ModelPlugin`: Custom model architectures
- `MetricPlugin`: Custom evaluation metrics

## Usage

Import protocols from the centralized location:

```python
from core.protocols import Model, DataLoader, Trainer
```

Or import from specific modules:

```python
from core.protocols.training import TrainingState, TrainingResult
from core.protocols.data import Dataset, Augmenter
```

## Backward Compatibility

For backward compatibility, the old import locations are maintained as re-exports:
- `training.core.protocols` → re-exports from `core.protocols.training`
- `data.core.interfaces` → re-exports from `core.protocols.data`
- `cli.plugins.base` → re-exports from `core.protocols.plugins`

## Benefits

1. **Single Source of Truth**: All protocols in one location
2. **No Circular Dependencies**: Clean dependency graph
3. **Better Organization**: Logical grouping of related protocols
4. **Easy Discovery**: All contracts in one place
5. **Type Safety**: Full typing support with Protocol

## Implementation Guidelines

When implementing a protocol:

1. Import the protocol from `core.protocols`
2. Implement all required methods and properties
3. Use type hints matching the protocol signatures
4. Document any implementation-specific behavior

Example:

```python
from core.protocols import Model

class MyModel:
    """Custom model implementation."""
    
    def __call__(self, inputs: dict[str, mx.array]) -> dict[str, mx.array]:
        # Implementation
        ...
    
    def parameters(self) -> dict[str, mx.array]:
        # Implementation
        ...
    
    # ... implement remaining protocol methods
```

## Adding New Protocols

To add a new protocol:

1. Determine the appropriate module (base, data, models, training, plugins)
2. Define the protocol with clear method signatures
3. Add comprehensive docstrings
4. Update the module's `__all__` export
5. Update this README with the new protocol

## Protocol vs ABC

This module uses `typing.Protocol` instead of `abc.ABC` because:

- Protocols support structural subtyping (duck typing)
- No need for explicit inheritance
- Better for defining interfaces
- More flexible for existing code
- Cleaner separation of interface and implementation

ABCs are still used for base class implementations that provide shared functionality.