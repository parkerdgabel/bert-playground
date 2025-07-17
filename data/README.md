# MLX Data Loading Architecture

This directory contains the MLX-native data loading infrastructure for Kaggle competitions.

## Overview

Our data loading architecture is designed to be:
- **Efficient**: Uses MLX's native streaming capabilities for optimal performance on Apple Silicon
- **Simple**: Clean, easy-to-understand API
- **Extensible**: Easy to add support for new Kaggle datasets

## Core Components

### 1. `mlx_dataloader.py` - Core DataLoader
The main `KaggleDataLoader` class that handles:
- Loading CSV data
- Converting tabular data to text
- Tokenization with HuggingFace tokenizers
- Efficient MLX streaming with prefetching
- Batching and shuffling

```python
from data import KaggleDataLoader, create_kaggle_dataloader

# Simple usage
loader = create_kaggle_dataloader(
    dataset_name="titanic",
    csv_path="train.csv",
    batch_size=32,
    max_length=128
)

# Iterate over batches
for batch in loader:
    # batch contains:
    # - input_ids: MLX array [batch_size, max_length]
    # - attention_mask: MLX array [batch_size, max_length]
    # - labels: MLX array [batch_size]
    pass
```

### 2. `datasets.py` - Dataset Registry
Centralized registry of dataset configurations:
- Predefined specs for common Kaggle competitions
- Easy registration of new datasets
- Metadata about columns, problem types, etc.

```python
from data import get_dataset_spec, register_dataset, DatasetSpec

# Get existing dataset spec
titanic_spec = get_dataset_spec("titanic")

# Register new dataset
new_spec = DatasetSpec(
    name="my_dataset",
    problem_type=ProblemType.BINARY_CLASSIFICATION,
    target_column="target",
    feature_columns=["col1", "col2", "col3"]
)
register_dataset(new_spec)
```

### 3. `text_generation.py` - Text Generation
Flexible text generation from tabular data:
- Generic tabular-to-text conversion
- Dataset-specific text generators
- Support for data augmentation

```python
from data import get_text_generator

# Get dataset-specific generator
generator = get_text_generator("titanic")

# Generate text from a data row
text = generator.generate(row)

# Generate multiple variations for augmentation
variations = generator.generate_augmented(row)
```

## Supported Datasets

Currently supported out-of-the-box:
- **Titanic**: Passenger survival prediction
- **House Prices**: Regression on house prices
- **Digit Recognizer**: MNIST digit classification

## Adding a New Dataset

1. **Register the dataset spec** in `datasets.py`:
```python
DATASET_SPECS['my_dataset'] = DatasetSpec(
    name='my_dataset',
    problem_type=ProblemType.BINARY_CLASSIFICATION,
    target_column='target',
    feature_columns=['feature1', 'feature2'],
    # ... other configuration
)
```

2. **Optional: Create custom text generator** in `text_generation.py`:
```python
class MyDatasetTextGenerator(TabularTextGenerator):
    def __init__(self):
        super().__init__(
            columns=['feature1', 'feature2'],
            column_descriptions={
                'feature1': 'First feature',
                'feature2': 'Second feature'
            }
        )
```

3. **Use the dataloader**:
```python
loader = create_kaggle_dataloader(
    dataset_name="my_dataset",
    csv_path="path/to/data.csv"
)
```

## Performance Considerations

- **Preprocessing**: Data is preprocessed once during initialization for efficiency
- **Streaming**: Uses MLX streams for memory-efficient data loading
- **Prefetching**: Multi-threaded prefetching for optimal throughput
- **Batch size**: Use powers of 2 for best MLX performance (16, 32, 64)

## Migration from Old DataLoaders

If you're migrating from the old dataloader implementations:

```python
# Old way
from data.dataloader import MLXTabularTextDataLoader
loader = MLXTabularTextDataLoader(...)

# New way  
from data import KaggleDataLoader
loader = KaggleDataLoader(...)
# or use the factory function
loader = create_kaggle_dataloader("titanic", ...)
```

The API is largely compatible - both provide the same batch format and iterator interface.