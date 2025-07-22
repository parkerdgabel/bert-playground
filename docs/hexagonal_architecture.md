# Hexagonal Architecture in k-bert

This document describes the hexagonal architecture (ports and adapters) implementation in k-bert, providing clean separation between business logic and infrastructure concerns.

## Overview

The hexagonal architecture pattern allows k-bert to be independent of external frameworks and infrastructure. All external dependencies are accessed through well-defined port interfaces, with concrete adapter implementations providing the actual functionality.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Domain Logic                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Models    │  │  Training   │  │    Data     │        │
│  │   Factory   │  │   Logic     │  │  Pipeline   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │
│         ▼                ▼                ▼                │
├─────────────────────────────────────────────────────────────┤
│                      Port Interfaces                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Compute    │  │   Storage   │  │ Monitoring  │        │
│  │    Port     │  │    Port     │  │    Port     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │   Config    │  │ Neural Ops  │                         │
│  │    Port     │  │    Port     │                         │
│  └─────────────┘  └─────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│                    Adapter Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ MLX Compute │  │ File Storage│  │   Loguru    │        │
│  │   Adapter   │  │   Adapter   │  │ Monitoring  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │ YAML Config │  │ MLX Neural  │                         │
│  │   Adapter   │  │ Ops Adapter │                         │
│  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Ports (Interfaces)

Ports define the contracts that the core domain uses to interact with external systems:

- **ComputeBackend**: Framework-agnostic tensor operations
- **StorageService**: Generic storage operations
- **ModelStorageService**: Specialized model persistence
- **ConfigurationProvider**: Configuration management
- **MonitoringService**: Logging and metrics
- **ExperimentTracker**: ML experiment tracking

### Adapters (Implementations)

Adapters provide concrete implementations of port interfaces:

- **MLXComputeAdapter**: MLX framework implementation
- **FileStorageAdapter**: File system storage
- **ModelFileStorageAdapter**: Model-specific file storage
- **YAMLConfigAdapter**: YAML configuration handling
- **LoguruMonitoringAdapter**: Loguru-based logging
- **MLflowExperimentTracker**: MLflow experiment tracking

## Usage Examples

### Basic Usage

```python
from core.factory import get_context

# Get adapter context
context = get_context()

# Use framework-agnostic compute operations
arr = context.compute.array([1, 2, 3, 4])
result = context.compute.zeros((5, 5))

# Use storage operations
context.storage.save("data.json", {"key": "value"})
data = context.storage.load("data.json")

# Use monitoring
context.monitoring.info("Operation completed")
context.monitoring.metric("accuracy", 0.95)
```

### Model Factory Integration

```python
from models.port_based_factory import PortBasedModelFactory

# Create factory with ports
factory = PortBasedModelFactory()

# Create model (framework-agnostic)
model = factory.create_model(
    model_type="bert_with_head",
    config={"hidden_size": 768}
)

# Save model using storage port
factory.save_model(model, Path("./model_checkpoint"))

# Load model using storage port
loaded_model, metadata = factory.load_model(Path("./model_checkpoint"))
```

### Custom Adapter Registration

```python
from core.factory import get_factory, register_adapter
from core.ports import ComputeBackend

# Create custom compute adapter
class PyTorchComputeAdapter:
    def __init__(self):
        import torch
        self.torch = torch
    
    @property
    def name(self) -> str:
        return "pytorch"
    
    def array(self, data, dtype=None, device=None):
        return self.torch.tensor(data, dtype=dtype, device=device)
    
    # ... implement other methods

# Register custom adapter
pytorch_adapter = PyTorchComputeAdapter()
register_adapter(ComputeBackend, pytorch_adapter)

# Now all code using ComputeBackend will use PyTorch
context = get_context()
arr = context.compute.array([1, 2, 3])  # Creates PyTorch tensor
```

## Port Interface Details

### ComputeBackend Port

Provides framework-agnostic tensor operations:

```python
@runtime_checkable
class ComputeBackend(Protocol):
    @property
    def name(self) -> str: ...
    
    def array(self, data, dtype=None, device=None): ...
    def zeros(self, shape, dtype=None, device=None): ...
    def compile(self, function): ...
    def gradient(self, function): ...
    # ... more methods
```

Key features:
- Framework-agnostic array creation and manipulation
- JIT compilation support
- Automatic differentiation
- Data type conversion utilities

### StorageService Port

Provides generic storage operations:

```python
@runtime_checkable
class StorageService(Protocol):
    def save(self, key, value, metadata=None): ...
    def load(self, key, expected_type=None): ...
    def exists(self, key) -> bool: ...
    def delete(self, key): ...
    def list_keys(self, prefix=None, pattern=None): ...
    def get_metadata(self, key): ...
```

Key features:
- Key-value storage abstraction
- Metadata support
- Type validation on load
- Pattern-based key listing

### MonitoringService Port

Provides observability and metrics:

```python
@runtime_checkable
class MonitoringService(Protocol):
    def log(self, level, message, context=None): ...
    def metric(self, name, value, tags=None): ...
    def timer(self, name, tags=None): ...
    def span(self, name, context=None): ...
    # ... more methods
```

Key features:
- Structured logging with context
- Metrics collection (gauge, counter, histogram)
- Timing measurements
- Distributed tracing support

## Adapter Implementation Guidelines

### Creating a New Adapter

1. **Define the interface** (if not exists):
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MyServicePort(Protocol):
    def operation(self, param: str) -> str: ...
```

2. **Implement the adapter**:
```python
class MyServiceAdapter:
    def __init__(self, config: dict):
        self.config = config
    
    def operation(self, param: str) -> str:
        # Concrete implementation
        return f"processed: {param}"
```

3. **Register the adapter**:
```python
from core.factory import register_adapter

register_adapter(MyServicePort, MyServiceAdapter(config))
```

### Adapter Best Practices

1. **Error Handling**: Convert framework-specific errors to generic exceptions
2. **Resource Management**: Properly handle resource lifecycle
3. **Configuration**: Accept configuration in constructor
4. **Logging**: Use structured logging for debugging
5. **Testing**: Implement comprehensive unit tests

## Testing Strategy

### Unit Tests

Test each adapter in isolation:

```python
def test_mlx_compute_adapter():
    adapter = MLXComputeAdapter()
    
    # Test basic operations
    arr = adapter.array([1, 2, 3])
    assert adapter.shape(arr) == (3,)
    
    # Test framework-specific behavior
    assert adapter.name == "mlx"
    assert adapter.supports_compilation is True
```

### Integration Tests

Test adapter interactions:

```python
def test_compute_storage_integration():
    compute = MLXComputeAdapter()
    storage = FileStorageAdapter()
    
    # Create data with compute port
    arr = compute.array([1, 2, 3])
    
    # Store via storage port
    storage.save("test.npy", compute.to_numpy(arr))
    
    # Load and convert back
    loaded = storage.load("test.npy")
    arr_loaded = compute.from_numpy(loaded)
    
    assert compute.array_equal(arr, arr_loaded)
```

### Contract Tests

Ensure adapters conform to port protocols:

```python
def test_adapter_protocol_compliance():
    adapter = MLXComputeAdapter()
    assert isinstance(adapter, ComputeBackend)
    
    # Test all required methods exist and are callable
    assert hasattr(adapter, 'array')
    assert callable(adapter.array)
```

## Configuration

### Adapter Configuration

Configure adapters through the factory:

```python
from core.factory import configure_adapters

config = {
    "storage": {
        "base_path": "/data/models",
        "format": "safetensors"
    },
    "compute": {
        "device": "gpu",
        "precision": "float32"
    },
    "monitoring": {
        "level": "INFO",
        "format": "json"
    }
}

configure_adapters(config)
```

### Environment-Based Configuration

Use environment variables for deployment-specific settings:

```python
import os
from core.factory import configure_adapters

config = {
    "storage": {
        "base_path": os.getenv("K_BERT_MODEL_PATH", "./models")
    },
    "monitoring": {
        "level": os.getenv("LOG_LEVEL", "INFO")
    }
}

configure_adapters(config)
```

## Benefits

### Framework Independence

The core domain logic is independent of specific ML frameworks:

```python
# This code works regardless of backend (MLX, PyTorch, TensorFlow)
def create_model_weights(compute: ComputeBackend):
    weights = compute.randn((768, 768))
    bias = compute.zeros((768,))
    return {"weight": weights, "bias": bias}
```

### Testability

Easy to test with mock adapters:

```python
def test_model_creation():
    mock_compute = Mock()
    mock_compute.randn.return_value = "mock_weights"
    
    weights = create_model_weights(mock_compute)
    assert weights["weight"] == "mock_weights"
```

### Flexibility

Easy to swap implementations:

```python
# Development: Use file storage
register_adapter(StorageService, FileStorageAdapter("./dev_data"))

# Production: Use cloud storage
register_adapter(StorageService, S3StorageAdapter(bucket="prod-models"))
```

### Maintainability

Changes to external dependencies are isolated to adapters:

```python
# Upgrade from MLX v1 to v2
class MLXv2ComputeAdapter:
    # Updated implementation for new MLX version
    pass

register_adapter(ComputeBackend, MLXv2ComputeAdapter())
# Core logic unchanged
```

## Migration Path

### Existing Code Migration

1. **Identify Dependencies**: Find direct framework usage
2. **Create Ports**: Define interface for the dependency
3. **Implement Adapters**: Create concrete implementations
4. **Update Code**: Use ports instead of direct dependencies
5. **Add Tests**: Ensure behavior is preserved

### Example Migration

Before (direct MLX usage):
```python
import mlx.core as mx

def create_weights():
    return mx.random.normal((768, 768))
```

After (using ports):
```python
from core.factory import get_context

def create_weights():
    compute = get_context().compute
    return compute.randn((768, 768))
```

## Future Extensions

### Planned Adapters

- **PyTorch Compute Adapter**: For PyTorch backend support
- **Cloud Storage Adapters**: S3, GCS, Azure Blob storage
- **Database Adapters**: For metadata and experiment storage
- **Distributed Computing**: Ray, Dask adapters
- **Advanced Monitoring**: Weights & Wandb, TensorBoard integration

### Plugin System Integration

The hexagonal architecture enables a plugin system where adapters can be loaded dynamically:

```python
# Plugin configuration
plugins = {
    "compute": "plugins.pytorch_adapter:PyTorchAdapter",
    "storage": "plugins.s3_adapter:S3Adapter",
    "monitoring": "plugins.wandb_adapter:WandbAdapter"
}

load_plugins(plugins)
```

This provides maximum flexibility for different deployment scenarios and user preferences.