# Secondary Adapters (Infrastructure)

This directory contains the concrete implementations of secondary (driven) ports - the infrastructure that the domain drives.

## Directory Structure

```
secondary/
├── compute/          # Neural network compute backends
│   ├── mlx.py      # Apple MLX framework implementation
│   └── pytorch.py  # PyTorch implementation (future)
├── storage/         # Persistence backends
│   ├── filesystem.py # Local file system storage
│   ├── s3.py       # Amazon S3 storage (future)
│   └── gcs.py      # Google Cloud Storage (future)
├── monitoring/      # Observability backends
│   ├── loguru.py   # Loguru logging and MLflow tracking
│   ├── tensorboard.py # TensorBoard integration (future)
│   └── wandb.py    # Weights & Biases integration (future)
├── data/           # Data loading backends
│   ├── mlx_loader.py # MLX-optimized data loader
│   └── torch_loader.py # PyTorch DataLoader (future)
└── tokenizer/      # Text tokenization backends
    ├── huggingface.py # HuggingFace tokenizers
    └── mlx_tokenizer.py # MLX-native tokenizer (future)
```

## Implementation Status

### ✅ Completed
- **MLX Compute Adapter**: Full MLX backend implementation with neural operations
- **File System Storage**: Local storage for models and checkpoints
- **Loguru Monitoring**: Structured logging with loguru
- **MLflow Integration**: Experiment tracking with MLflow
- **MLX Data Loader**: High-performance data loading for Apple Silicon
- **HuggingFace Tokenizer**: Integration with HuggingFace tokenizer ecosystem

### 🚧 Future Implementations
- **PyTorch Compute**: PyTorch backend for cross-platform support
- **Cloud Storage**: S3, GCS, Azure Blob storage adapters
- **Advanced Monitoring**: TensorBoard, W&B, Neptune integrations
- **Alternative Data Loaders**: PyTorch, TensorFlow data pipelines
- **Custom Tokenizers**: MLX-native and custom tokenizer implementations

## Usage Examples

### Using MLX Compute Backend

```python
from adapters.secondary.compute.mlx import MLXComputeAdapter, MLXNeuralBackend

# Initialize compute backend
compute = MLXComputeAdapter()

# Create arrays
x = compute.array([1, 2, 3], dtype="float32")
y = compute.zeros((3, 3))

# Use neural backend
neural = MLXNeuralBackend(compute)
linear = neural.linear(10, 5)
```

### Using File Storage

```python
from adapters.secondary.storage.filesystem import FileStorageAdapter, ModelFileStorageAdapter

# General storage
storage = FileStorageAdapter("./data")
storage.save("config.json", {"model": "bert", "epochs": 10})
config = storage.load("config.json")

# Model storage
model_storage = ModelFileStorageAdapter("./models")
model_storage.save_model(model, "bert-finetuned", metadata)
loaded_model, metadata = model_storage.load_model("bert-finetuned")
```

### Using Monitoring

```python
from adapters.secondary.monitoring.loguru import LoguruMonitoringAdapter, MLflowExperimentTracker

# Logging
monitor = LoguruMonitoringAdapter()
monitor.info("Training started", epoch=1, batch_size=32)
monitor.log_metric("loss", 0.5, step=100)

# Experiment tracking
tracker = MLflowExperimentTracker()
tracker.create_experiment("bert-tuning")
run_id = tracker.start_run("baseline")
tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
tracker.log_metrics({"train_loss": 0.5, "val_loss": 0.4}, step=100)
tracker.end_run()
```

### Using Data Loader

```python
from adapters.secondary.data.mlx_loader import MLXDataLoader, MLXLoaderConfig

config = MLXLoaderConfig(
    batch_size=32,
    shuffle=True,
    max_length=512,
    prefetch_size=2
)

loader = MLXDataLoader(
    dataset=dataset,
    config=config,
    tokenizer=tokenizer,
    compute=compute
)

for batch in loader:
    # Process batch
    pass
```

## Design Principles

1. **Port Implementation**: Each adapter implements a specific port interface
2. **Framework Isolation**: Framework-specific code is isolated in adapters
3. **Dependency Injection**: Adapters can be injected into domain services
4. **Configuration**: Adapters are configurable through their constructors
5. **Error Handling**: Adapters handle framework-specific errors and convert them

## Testing

Each adapter should have comprehensive tests:
- Unit tests for individual methods
- Integration tests with actual frameworks
- Performance tests for critical paths
- Error handling tests

## Contributing

When adding new adapters:
1. Implement the corresponding port interface
2. Keep framework-specific imports isolated
3. Provide comprehensive documentation
4. Add unit and integration tests
5. Update this README with usage examples