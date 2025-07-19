# MLX BERT Data Module

This module provides a sophisticated, protocol-driven data handling system optimized for BERT models on Apple Silicon using MLX. It features high-performance data loaders, text conversion templates, and comprehensive Kaggle competition support.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Data Loading Strategies](#data-loading-strategies)
- [Text Conversion System](#text-conversion-system)
- [Dataset Registry](#dataset-registry)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Extending the Module](#extending-the-module)

## Architecture Overview

The data module follows a layered, protocol-based architecture:

```
data/
├── core/                   # Base abstractions and protocols
│   ├── base.py            # KaggleDataset base class
│   ├── interfaces.py      # Protocol definitions
│   ├── metadata.py        # Competition metadata
│   └── registry.py        # Dataset registry
├── loaders/               # Data loading implementations
│   ├── mlx_loader.py      # MLX-optimized loader
│   ├── streaming.py       # Streaming pipeline
│   └── memory.py          # Memory management
├── templates/             # Text conversion
│   ├── engine.py          # Template engine
│   ├── converters.py      # High-level converters
│   └── templates.py       # Template definitions
└── kaggle/               # Competition-specific datasets
```

## Core Components

### Protocols and Interfaces

The module uses Python protocols for flexibility:

```python
from data.core.interfaces import Dataset, DataLoader, TextConverter

# All implementations follow these protocols
class MyDataset(Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...

class MyLoader(DataLoader):
    def __iter__(self) -> Iterator[Batch]: ...
    def __len__(self) -> int: ...

class MyConverter(TextConverter):
    def convert(self, row: Dict[str, Any]) -> str: ...
```

### Base Dataset Class

`KaggleDataset` provides a unified interface for all competitions:

```python
from data.core.base import KaggleDataset, DatasetSpec

class TitanicDataset(KaggleDataset):
    def __init__(self, data_path: str, spec: DatasetSpec):
        super().__init__(data_path, spec)
        # Custom initialization
    
    def _load_data(self) -> pd.DataFrame:
        # Load and preprocess data
        return pd.read_csv(self.data_path)
    
    def _create_sample(self, idx: int) -> Dict[str, mx.array]:
        # Convert row to MLX arrays
        return {
            "input_ids": mx.array(tokens),
            "attention_mask": mx.array(mask),
            "labels": mx.array(label)
        }
```

### Dataset Specification

Configure datasets with comprehensive specifications:

```python
from data.core.base import DatasetSpec, CompetitionType

spec = DatasetSpec(
    competition_name="titanic",
    competition_type=CompetitionType.BINARY_CLASSIFICATION,
    num_samples=891,
    num_features=11,
    text_columns=["Name", "Ticket"],
    categorical_columns=["Sex", "Embarked", "Pclass"],
    numerical_columns=["Age", "Fare", "SibSp", "Parch"],
    target_column="Survived",
    # MLX optimization settings
    mlx_config={
        "batch_size": 32,
        "prefetch_size": 4,
        "num_workers": 8,
        "cache_size": 1000
    }
)
```

## Data Loading Strategies

### 1. MLX-Optimized Loader

High-performance loader leveraging Apple Silicon:

```python
from data.loaders.mlx_loader import MLXDataLoader

loader = MLXDataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    prefetch_size=4,
    drop_last=False
)

# Automatic prefetching and caching
for batch in loader:
    # Zero-copy operations with unified memory
    input_ids = batch["input_ids"]  # MLX array
    labels = batch["labels"]        # MLX array
```

**Features:**
- Zero-copy operations using unified memory
- Multi-threaded prefetching
- Intelligent batch caching
- Automatic tokenization integration
- Performance tracking (samples/sec)

### 2. Streaming Pipeline

For large datasets that don't fit in memory:

```python
from data.loaders.streaming import StreamingPipeline

pipeline = StreamingPipeline(
    data_source="path/to/large_dataset.csv",
    batch_size=64,
    buffer_size=10000,
    num_producers=4
)

# Asynchronous iteration
async for batch in pipeline:
    # Process batch
    pass

# Or synchronous
for batch in pipeline.sync_iter():
    # Process batch
    pass
```

**Features:**
- Producer-consumer architecture
- Target: 1000+ samples/second
- Adaptive batching
- Memory monitoring
- Real-time statistics

### 3. Memory Manager

Global memory optimization:

```python
from data.loaders.memory import UnifiedMemoryManager

# Singleton instance
memory_manager = UnifiedMemoryManager.get_instance()

# Create tensor pool for common shapes
with memory_manager.tensor_pool_context():
    # Tensors are pooled and reused
    tensor = memory_manager.allocate((32, 512))
    
# Monitor memory pressure
if memory_manager.should_cleanup():
    memory_manager.cleanup_tensors()
```

## Text Conversion System

### Template Engine

Convert tabular data to natural language:

```python
from data.templates.engine import TextTemplateEngine

engine = TextTemplateEngine(
    tokenizer=tokenizer,
    max_length=256,
    competition_type="classification"
)

# Convert row to text
text = engine.convert_row({
    "Age": 25,
    "Sex": "male",
    "Pclass": 1,
    "Fare": 100.0
})
# Output: "The passenger is a 25 year old male traveling in first class..."
```

### High-Level Converters

Ready-to-use converters for common scenarios:

```python
from data.templates.converters import TabularTextConverter, BERTTextConverter

# General tabular-to-text
converter = TabularTextConverter(
    text_columns=["Name", "Description"],
    categorical_columns=["Category", "Type"],
    numerical_columns=["Price", "Rating"]
)

# BERT-specific with tokenization
bert_converter = BERTTextConverter(
    tokenizer=tokenizer,
    max_length=512,
    add_special_tokens=True
)

# Convert and tokenize
tokens = bert_converter.fit_transform(df)
```

### Template Types

Different templates for different competition types:

```python
# Binary Classification
template = "Predict if {target}: {features}"

# Multiclass Classification  
template = "Classify into {num_classes} categories based on: {features}"

# Regression
template = "Estimate the {target} value given: {features}"

# Time Series
template = "At time {timestamp}, the features are: {features}"
```

## Dataset Registry

Centralized management of multiple competitions:

```python
from data.core.registry import DatasetRegistry

# Initialize registry
registry = DatasetRegistry()

# Auto-discover datasets
registry.scan_directory("data/kaggle/")

# Register new dataset
registry.register(
    name="titanic",
    spec=titanic_spec,
    path="data/kaggle/titanic"
)

# Create dataset by name
dataset = registry.create_dataset("titanic")

# Find competitions by type
binary_competitions = registry.find_by_type(CompetitionType.BINARY_CLASSIFICATION)

# Export/import registry
registry.save("registry.json")
registry = DatasetRegistry.load("registry.json")
```

## Usage Examples

### Complete Training Pipeline

```python
from data import create_data_pipeline
from models import create_model
from training import create_trainer

# Create complete data pipeline
train_loader, val_loader = create_data_pipeline(
    competition="titanic",
    batch_size=32,
    max_length=256,
    augmentation=True
)

# Train model
model = create_model("modernbert", num_labels=2)
trainer = create_trainer(model)
trainer.fit(train_loader, val_loader)
```

### Custom Dataset Creation

```python
from data.core.base import KaggleDataset, DatasetSpec
from data.templates.engine import TextTemplateEngine

class CustomDataset(KaggleDataset):
    def __init__(self, data_path: str):
        spec = DatasetSpec(
            competition_name="custom",
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            num_samples=10000,
            num_features=20,
            target_column="category"
        )
        super().__init__(data_path, spec)
        
        # Custom text conversion
        self.text_engine = TextTemplateEngine(
            tokenizer=self.tokenizer,
            template="Custom template: {features}"
        )
    
    def _create_sample(self, idx: int) -> Dict[str, mx.array]:
        row = self.data.iloc[idx]
        text = self.text_engine.convert_row(row)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            "input_ids": mx.array(encoding["input_ids"]),
            "attention_mask": mx.array(encoding["attention_mask"]),
            "labels": mx.array(row[self.target_column])
        }
```

### Streaming Large Datasets

```python
from data.loaders.streaming import StreamingPipeline
from data.templates.converters import BERTTextConverter

# Setup converter
converter = BERTTextConverter(tokenizer, max_length=512)

# Create streaming pipeline
pipeline = StreamingPipeline(
    data_source="huge_dataset.parquet",
    batch_size=128,
    transform_fn=converter.transform,
    num_producers=8,
    memory_limit_gb=4.0
)

# Process in chunks
for batch in pipeline:
    # Batch is already tokenized and converted to MLX arrays
    outputs = model(batch["input_ids"], batch["attention_mask"])
```

## Performance Optimization

### Configuration Profiles

```python
from data import DataConfig, OptimizationProfile

# Development profile
dev_config = DataConfig.from_profile(OptimizationProfile.DEVELOPMENT)
# batch_size=8, workers=2, prefetch=1

# Training profile  
train_config = DataConfig.from_profile(OptimizationProfile.TRAINING)
# batch_size=32, workers=8, prefetch=4

# Competition profile
comp_config = DataConfig.from_profile(OptimizationProfile.COMPETITION)
# batch_size=64, workers=16, prefetch=8
```

### Memory Optimization Strategies

1. **Tensor Pooling**: Reuse common tensor shapes
2. **Batch Caching**: Cache frequently accessed batches
3. **Lazy Loading**: Load data only when needed
4. **Memory Monitoring**: Automatic cleanup on pressure

### Performance Tips

1. **Batch Size**: Use powers of 2 (32, 64, 128)
2. **Workers**: Set to number of CPU cores
3. **Prefetch**: 2-4x batch size for smooth loading
4. **Cache Size**: Balance between memory and speed

## Extending the Module

### Adding New Competitions

1. Create dataset class:
```python
class NewCompetitionDataset(KaggleDataset):
    def _load_data(self) -> pd.DataFrame:
        # Custom loading logic
        pass
    
    def _create_sample(self, idx: int) -> Dict[str, mx.array]:
        # Custom sample creation
        pass
```

2. Define specification:
```python
spec = DatasetSpec(
    competition_name="new-competition",
    competition_type=CompetitionType.REGRESSION,
    # ... other settings
)
```

3. Register with factory:
```python
from data import register_dataset
register_dataset("new-competition", NewCompetitionDataset, spec)
```

### Custom Text Templates

```python
from data.templates.engine import TemplateEngine

class CustomTemplateEngine(TemplateEngine):
    def create_template(self, row: Dict) -> str:
        # Custom template logic
        return f"Custom format: {row}"
```

### Custom Loaders

```python
from data.core.interfaces import DataLoader

class CustomLoader(DataLoader):
    def __iter__(self) -> Iterator[Batch]:
        # Custom iteration logic
        pass
```

## Best Practices

1. **Use Appropriate Loader**:
   - MLXDataLoader for standard datasets
   - StreamingPipeline for large datasets
   - Custom loaders for special requirements

2. **Text Conversion**:
   - Keep templates concise
   - Include relevant features only
   - Handle missing values gracefully

3. **Memory Management**:
   - Monitor memory usage
   - Use tensor pooling
   - Enable cleanup on pressure

4. **Performance**:
   - Profile data loading
   - Use prefetching
   - Optimize batch sizes

## Troubleshooting

### Common Issues

**Slow Data Loading**:
- Increase number of workers
- Enable prefetching
- Check disk I/O bottlenecks

**Out of Memory**:
- Reduce batch size
- Use streaming pipeline
- Enable aggressive cleanup

**Tokenization Errors**:
- Check max_length settings
- Verify text conversion
- Handle special characters

For more details, see the individual module documentation and implementation files.