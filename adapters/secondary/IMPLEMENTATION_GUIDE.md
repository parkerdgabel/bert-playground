# Secondary Adapters Implementation Guide

This directory contains the secondary (driven) adapters that implement the domain ports using specific technologies.

## Structure

```
secondary/
├── compute/               # Neural network compute operations
│   ├── base.py           # Base compute adapter
│   └── mlx/              # MLX-specific implementations
│       ├── compute_adapter.py    # Main compute port implementation
│       ├── model_adapter.py      # Model wrapping and conversion
│       ├── optimization.py       # Optimizer implementations
│       └── utils.py              # MLX utilities
├── data/                  # Data loading and processing
│   ├── base.py           # Base data adapter
│   └── mlx/              # MLX data implementations
│       ├── data_loader.py        # DataLoader implementation
│       ├── dataset.py            # Dataset wrapper
│       └── transforms.py         # Data transformations
└── metrics/               # Metrics calculation
    └── mlx/
        └── metrics_calculator.py # Metrics port implementation
```

## Key Components

### Compute Adapters

The compute adapters handle all neural network operations:

- **MLXComputeAdapter**: Implements the `ComputePort` for MLX framework
  - Forward/backward passes
  - Model compilation
  - Gradient computation
  - Device management

- **MLXModelAdapter**: Wraps domain `BertModel` entities as MLX modules
  - Converts architecture specifications to MLX models
  - Manages weights and parameters
  - Handles training/eval modes

- **MLXOptimizer**: Optimization algorithms for MLX
  - AdamW, Adam, SGD, RMSprop, Lion
  - Gradient clipping
  - Learning rate management

### Data Adapters

The data adapters provide efficient data loading:

- **MLXDataLoader**: Implements `DataLoaderPort` 
  - Efficient batching with MLX arrays
  - Shuffling and prefetching
  - Pre-tokenization support
  - Streaming for large datasets

- **MLXDatasetWrapper**: Wraps domain `Dataset` entities
  - Caching for small datasets
  - Pre-tokenized data support
  - Efficient sample access

- **Transform Classes**: Data preprocessing
  - Tokenization
  - Padding/truncation
  - Normalization
  - Augmentation

### Metrics Adapters

- **MLXMetricsCalculator**: Implements `MetricsCalculatorPort`
  - Classification metrics (accuracy, precision, recall, F1)
  - Regression metrics (MSE, MAE, R²)
  - Advanced metrics (AUC-ROC, AUC-PR)
  - Per-class metrics
  - Loss calculations

## Usage Example

```python
from domain.entities.model import BertModel, ModelArchitecture
from domain.entities.dataset import Dataset, DatasetSplit
from adapters.secondary.compute.mlx import MLXComputeAdapter
from adapters.secondary.data.mlx import MLXDataLoader
from adapters.secondary.metrics.mlx import MLXMetricsCalculator

# Create adapters
compute = MLXComputeAdapter()
data_loader = MLXDataLoader()
metrics = MLXMetricsCalculator()

# Create domain entities
architecture = ModelArchitecture(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
)
model = BertModel(architecture=architecture)

# Create dataset
dataset = Dataset(
    name="my_dataset",
    split=DatasetSplit.TRAIN,
    size=1000,
)

# Create dataloader
loader = data_loader.create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
)

# Training loop
for batch in loader:
    # Forward pass
    outputs = compute.forward(model, batch, training=True)
    
    # Calculate metrics
    if "logits" in outputs and batch.labels is not None:
        accuracy = metrics.calculate_accuracy(
            predictions=outputs["logits"].argmax(axis=-1),
            labels=batch.labels
        )
```

## Implementation Notes

### MLX Optimizations

1. **Unified Memory**: MLX uses unified memory on Apple Silicon, eliminating CPU-GPU transfers
2. **Lazy Evaluation**: Operations are lazily evaluated for optimization
3. **Compilation**: Models can be compiled with `mx.compile` for faster execution
4. **Mixed Precision**: Support for bfloat16 on supported hardware

### Data Loading

1. **Zero-Copy**: MLX arrays support zero-copy operations where possible
2. **Pre-tokenization**: Datasets can be pre-tokenized and cached for faster loading
3. **Streaming**: Large datasets can be streamed with buffering
4. **Prefetching**: Multiple batches can be prefetched for better throughput

### Error Handling

All adapters include proper error handling:
- Invalid input validation
- Graceful degradation for unsupported features
- Clear error messages for debugging

### Testing

Each adapter should be tested with:
- Unit tests for individual methods
- Integration tests with domain entities
- Performance benchmarks
- Memory usage profiling

## Adding New Adapters

To add support for a new framework (e.g., PyTorch):

1. Create a new subdirectory (e.g., `compute/pytorch/`)
2. Implement the relevant domain ports
3. Follow the same structure as MLX adapters
4. Add conversion utilities between domain entities and framework types
5. Include comprehensive tests
6. Update the factory functions to support the new adapter

## Performance Considerations

- **Batch Size**: Use powers of 2 for optimal MLX performance
- **Compilation**: Enable model compilation for production
- **Caching**: Use pre-tokenized data for large datasets
- **Memory**: Monitor memory usage with unified memory architecture
- **Profiling**: Use MLX profiling tools to identify bottlenecks