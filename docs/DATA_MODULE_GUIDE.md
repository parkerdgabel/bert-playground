# Data Module Architecture Guide

## Overview

The data module in MLX-BERT has been completely redesigned for maximum performance and extensibility. This guide explains the architecture, components, and how to use the new system.

## Architecture Overview

```
data/
├── core/                    # Core abstractions and base classes
│   ├── base.py             # Fundamental types and interfaces
│   ├── metadata.py         # Competition metadata handling
│   └── registry.py         # Dataset discovery and management
├── loaders/                # Data loading implementations
│   ├── mlx_loader.py       # MLX-optimized loader
│   ├── streaming.py        # Streaming for large datasets
│   └── memory.py           # Memory management
├── templates/              # Text conversion system
│   ├── engine.py           # Template management
│   ├── converters.py       # Data converters
│   └── base_template.py    # Template interface
└── datasets/               # Competition implementations
    ├── titanic.py          # Example: Titanic dataset
    └── __init__.py         # Auto-discovery

```

## Core Components

### 1. Base Classes (`core/base.py`)

#### CompetitionType
Defines the type of machine learning competition:
```python
class CompetitionType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    # ... more types
```

#### DatasetSpec
Specifications for dataset configuration:
```python
@dataclass
class DatasetSpec:
    competition_name: str
    dataset_path: Path
    competition_type: CompetitionType
    num_samples: int
    num_features: int
    # ... additional fields
```

#### KaggleDataset
Abstract base class for all datasets:
```python
class KaggleDataset(ABC):
    def __init__(self, spec: DatasetSpec, split: str = "train"):
        self.spec = spec
        self.split = split
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pass
```

### 2. Dataset Registry (`core/registry.py`)

The registry manages all available datasets and their metadata:

```python
registry = DatasetRegistry()

# Register a new competition
registry.register_competition(
    competition_name="titanic",
    metadata=metadata,
    spec=spec
)

# List available competitions
competitions = registry.list_competitions()

# Search competitions
results = registry.search_competitions(
    competition_type=CompetitionType.BINARY_CLASSIFICATION
)
```

### 3. MLX Data Loader (`loaders/mlx_loader.py`)

High-performance loader optimized for Apple Silicon:

```python
from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig

# Configure the loader
config = MLXLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_size=4,
    use_unified_memory=True,
    pin_memory=True
)

# Create loader
loader = MLXDataLoader(dataset, config)

# Iterate through batches
for batch in loader:
    # batch contains MLX arrays
    input_ids = batch['input_ids']      # mx.array
    attention_mask = batch['attention_mask']  # mx.array
    labels = batch['labels']            # mx.array
```

Key features:
- **Zero-copy operations**: Minimal memory overhead
- **Unified memory**: Optimized for Apple Silicon
- **Intelligent prefetching**: Keeps GPU fed with data
- **Gradient accumulation**: Support for effective larger batches

### 4. Streaming Pipeline (`loaders/streaming.py`)

For datasets too large to fit in memory:

```python
from data.loaders.streaming import StreamingPipeline, StreamingConfig

config = StreamingConfig(
    buffer_size=1024,
    chunk_size=256,
    num_workers=4,
    target_throughput=1000  # samples/sec
)

pipeline = StreamingPipeline(dataset, config)
pipeline.start()

for sample in pipeline:
    # Process streaming data
    pass
    
pipeline.stop()
```

### 5. Template Engine (`templates/engine.py`)

Flexible system for converting tabular data to text:

```python
from data.templates.engine import TextTemplateEngine
from data.templates.converters import TabularTextConverter

engine = TextTemplateEngine()

# Register templates
engine.register_template(competition_template)

# Convert data
texts = engine.convert_data(
    data=df,
    template_name="titanic_v1"
)
```

## Creating a New Dataset

### Step 1: Create Dataset Class

Create a new file in `data/datasets/`:

```python
# data/datasets/house_prices.py
from pathlib import Path
import pandas as pd
import mlx.core as mx

from data.core.base import KaggleDataset, DatasetSpec, CompetitionType
from data.templates.converters import TabularTextConverter

class HousePricesDataset(KaggleDataset):
    """House Prices competition dataset."""
    
    def __init__(self, spec: DatasetSpec, split: str = "train", 
                 transform=None, cache_dir=None):
        super().__init__(spec, split, transform, cache_dir)
        self.converter = TabularTextConverter()
        
    def _load_data(self):
        """Load the dataset."""
        file_path = self.spec.dataset_path / f"{self.split}.csv"
        self._data = pd.read_csv(file_path)
        
    def _validate_data(self):
        """Validate the loaded data."""
        required_columns = ['Id', 'SalePrice'] if self.split == 'train' else ['Id']
        for col in required_columns:
            if col not in self._data.columns:
                raise ValueError(f"Missing required column: {col}")
                
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample."""
        row = self._data.iloc[index]
        
        # Convert to text
        text = self.converter.convert_row(row)
        
        # Tokenize (mock for example)
        tokens = self._tokenize(text)
        
        sample = {
            'text': text,
            'input_ids': mx.array(tokens['input_ids']),
            'attention_mask': mx.array(tokens['attention_mask']),
            'metadata': {'id': row['Id']}
        }
        
        if self.split == 'train':
            sample['labels'] = mx.array([row['SalePrice']])
            
        return sample
```

### Step 2: Register with Auto-Discovery

The dataset will be automatically discovered if it follows the naming convention:
- File ends with `_dataset.py` or is in `datasets/` folder
- Class name ends with `Dataset`
- Inherits from `KaggleDataset`

### Step 3: Create Competition Spec

```python
from data.core.metadata import CompetitionMetadata

# Create metadata
metadata = CompetitionMetadata(
    competition_name="house-prices",
    competition_type=CompetitionType.REGRESSION,
    description="Predict house prices",
    evaluation_metric="rmse",
    train_file="train.csv",
    test_file="test.csv"
)

# Create spec
spec = DatasetSpec(
    competition_name="house-prices",
    dataset_path=Path("data/house-prices"),
    competition_type=CompetitionType.REGRESSION,
    num_samples=1460,
    num_features=79,
    target_column="SalePrice"
)

# Register
registry.register_competition("house-prices", metadata, spec)
```

## Performance Optimization

### 1. Memory Management

The unified memory manager optimizes memory usage:

```python
from data.loaders.memory import UnifiedMemoryManager, MemoryConfig

config = MemoryConfig(
    pool_size_mb=512,
    enable_pooling=True,
    enable_unified_memory=True
)

manager = UnifiedMemoryManager(config)

# Allocate tensors from pool
tensor = manager.allocate_tensor(shape=(32, 512), dtype=mx.float32)

# Cache frequently used tensors
manager.cache_tensor("embeddings", tensor)
```

### 2. Profiling and Monitoring

```python
# Enable profiling
loader.enable_profiling()

# Train for a few batches
for batch in loader:
    # ... training code ...
    pass

# Get profiling results
profile = loader.get_profiling_results()
print(f"Throughput: {profile['throughput_samples_per_sec']} samples/sec")
print(f"Avg batch time: {profile['avg_batch_time_ms']} ms")
```

### 3. Best Practices

1. **Batch Size**: Use powers of 2 (32, 64, 128)
2. **Prefetching**: Set prefetch_size to 2-8x batch size
3. **Workers**: Use 4-8 workers for data loading
4. **Memory**: Enable unified memory for Apple Silicon
5. **Streaming**: Use for datasets > 1GB

## Usage Example

Here's how to use the data module:

```python
from data.datasets.titanic import TitanicDataset
from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig

# Create dataset
dataset = TitanicDataset(spec, split="train")

# Configure loader
config = MLXLoaderConfig(batch_size=32)

# Create loader
loader = MLXDataLoader(dataset, config)

# Train your model
for batch in loader:
    # batch contains MLX arrays ready for training
    outputs = model(batch['input_ids'], batch['attention_mask'])
    loss = compute_loss(outputs, batch['labels'])
```

## Testing

The data module includes comprehensive tests:

```bash
# Run all data module tests
pytest tests/unit/data/

# Run specific test file
pytest tests/unit/data/test_loaders.py

# Run with coverage
pytest tests/unit/data/ --cov=data --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use streaming pipeline

2. **Slow Loading**
   - Increase num_workers
   - Enable prefetching
   - Check disk I/O

3. **Type Errors**
   - Ensure all tensors are MLX arrays
   - Check data types match model expectations

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger("data").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Distributed Loading**: Multi-device support
2. **Smart Caching**: Adaptive caching based on access patterns
3. **Data Versioning**: Track dataset versions
4. **Auto-optimization**: Automatic parameter tuning
5. **Cloud Integration**: Direct loading from cloud storage