# Data Module Test Suite

Comprehensive test suite for the data module including loaders, templates, caching, and Kaggle dataset integration.

## Test Structure

```
tests/data/
├── unit/                    # Unit tests for individual components
│   ├── core/               # Core data abstractions
│   │   ├── test_base.py
│   │   ├── test_interfaces.py
│   │   ├── test_metadata.py
│   │   └── test_registry.py
│   ├── loaders/            # Data loader tests
│   │   ├── test_memory.py
│   │   ├── test_mlx_loader.py
│   │   └── test_streaming.py
│   ├── templates/          # Template engine tests
│   │   ├── test_converters.py
│   │   ├── test_engine.py
│   │   └── test_templates.py
│   └── kaggle/             # Kaggle-specific tests
│       └── test_kaggle_datasets.py
├── integration/            # Integration tests
│   ├── test_data_model_integration.py
│   ├── test_data_pipeline.py
│   └── test_caching.py
├── e2e/                    # End-to-end tests
│   ├── test_complete_data_workflow.py
│   └── test_kaggle_data_flow.py
├── fixtures/               # Shared test fixtures
│   ├── datasets.py        # Synthetic datasets
│   ├── configs.py         # Data configurations
│   └── utils.py          # Data testing utilities
├── conftest.py            # Pytest configuration
└── README.md              # This file
```

## Running Tests

### Run All Tests
```bash
# Run all tests with coverage
pytest tests/data/ -v --cov=data --cov-report=html

# Run with markers
pytest tests/data/ -m unit      # Unit tests only
pytest tests/data/ -m integration # Integration tests only
pytest tests/data/ -m e2e        # End-to-end tests only
```

### Run Specific Test Categories
```bash
# Core data tests
pytest tests/data/unit/core/ -v

# Loader tests
pytest tests/data/unit/loaders/ -v

# Template tests
pytest tests/data/unit/templates/ -v

# Kaggle tests
pytest tests/data/unit/kaggle/ -v
```

### Run Individual Test Files
```bash
# Test data interfaces
pytest tests/data/unit/core/test_interfaces.py -v

# Test MLX loader
pytest tests/data/unit/loaders/test_mlx_loader.py -v

# Test template engine
pytest tests/data/unit/templates/test_engine.py -v
```

### Run with Specific Markers
```bash
# Fast tests only (exclude slow tests)
pytest tests/data/ -v -m "not slow"

# Streaming tests
pytest tests/data/ -v -m streaming

# Kaggle tests
pytest tests/data/ -v -m kaggle
```

## Test Coverage

### Current Coverage Goals
- Unit tests: 90% coverage
- Integration tests: 80% coverage
- E2E tests: Core workflows covered
- Total: 85%+ coverage

### View Coverage Report
```bash
# Generate HTML coverage report
pytest tests/data/ --cov=data --cov-report=html

# Open report
open htmlcov/index.html
```

## Test Fixtures

### Datasets (`fixtures/datasets.py`)
- `SyntheticTabularDataset`: Basic tabular data generation
- `SyntheticTextDataset`: Text data generation
- `ImbalancedDataset`: Imbalanced classification data
- `LargeDataset`: Large dataset for memory testing
- `StreamingDataset`: Streaming data simulation
- `MultiModalDataset`: Mixed data types

### Configurations (`fixtures/configs.py`)
- `create_loader_config()`: Standard loader configuration
- `create_streaming_config()`: Streaming loader config
- `create_memory_config()`: Memory-optimized config
- `create_kaggle_config()`: Kaggle dataset config
- `create_template_config()`: Template engine config

### Utilities (`fixtures/utils.py`)
- Data comparison utilities
- Mock data generators
- Performance profiling
- Memory tracking
- Streaming simulation

## Writing New Tests

### Unit Test Template
```python
import pytest
from data.core import DatasetSpec, KaggleDataset
from tests.data.fixtures import create_synthetic_dataset

class TestDatasetSpec:
    """Test DatasetSpec functionality."""
    
    def test_initialization(self):
        """Test dataset spec initialization."""
        spec = DatasetSpec(
            name="test_dataset",
            task_type="classification",
            num_features=10,
            num_classes=2,
        )
        
        assert spec.name == "test_dataset"
        assert spec.task_type == "classification"
    
    def test_validation(self):
        """Test spec validation."""
        with pytest.raises(ValueError):
            DatasetSpec(
                name="",  # Invalid empty name
                task_type="invalid",  # Invalid task type
            )
    
    def test_serialization(self):
        """Test spec serialization."""
        spec = DatasetSpec(name="test", task_type="regression")
        json_data = spec.to_json()
        loaded_spec = DatasetSpec.from_json(json_data)
        
        assert loaded_spec == spec
```

### Integration Test Template
```python
class TestDataModelIntegration:
    """Test data integration with models."""
    
    def test_loader_with_model(self):
        """Test data loader works with model."""
        from models.bert import BertCore
        from data.loaders import MLXDataLoader
        
        loader = create_test_loader()
        model = create_test_model()
        
        for batch in loader:
            outputs = model(batch["input_ids"], batch["attention_mask"])
            assert outputs is not None
```

### E2E Test Template
```python
class TestCompleteDataWorkflow:
    """Test complete data workflows."""
    
    def test_kaggle_data_pipeline(self, tmp_path):
        """Test complete Kaggle data pipeline."""
        # Download data
        dataset = KaggleDataset.from_competition("titanic")
        dataset.download(tmp_path)
        
        # Create loader
        loader = dataset.create_loader(
            batch_size=32,
            shuffle=True,
        )
        
        # Process data
        for batch in loader:
            assert "features" in batch
            assert "labels" in batch
        
        # Create submission
        submission = dataset.create_submission(predictions)
        assert submission.is_valid()
```

## Data-Specific Testing Patterns

### Testing Data Loaders
```python
def test_loader_iteration():
    """Test loader can iterate through data."""
    loader = MLXDataLoader(
        dataset=create_synthetic_dataset(),
        batch_size=4,
        shuffle=True,
    )
    
    batches = list(loader)
    assert len(batches) > 0
    assert all("input_ids" in batch for batch in batches)
```

### Testing Streaming
```python
def test_streaming_pipeline():
    """Test streaming data pipeline."""
    stream = StreamingPipeline(
        source="data.csv",
        batch_size=32,
        prefetch=4,
    )
    
    # Test streaming
    with stream:
        for i, batch in enumerate(stream):
            assert batch.shape[0] <= 32
            if i > 10:  # Test early stopping
                break
```

### Testing Templates
```python
def test_template_conversion():
    """Test template conversion."""
    template = TabularToTextTemplate()
    
    row = {"age": 25, "gender": "M", "survived": 1}
    text = template.convert(row)
    
    assert "25" in text
    assert "male" in text.lower()
```

### Testing Caching
```python
def test_cache_behavior(tmp_path):
    """Test caching behavior."""
    cache_dir = tmp_path / "cache"
    loader = CachedDataLoader(
        dataset=create_large_dataset(),
        cache_dir=cache_dir,
    )
    
    # First load - should cache
    t1 = time.time()
    data1 = list(loader)
    time1 = time.time() - t1
    
    # Second load - should be faster
    t2 = time.time()
    data2 = list(loader)
    time2 = time.time() - t2
    
    assert time2 < time1 * 0.5  # At least 2x faster
    assert cache_dir.exists()
```

## Common Testing Patterns

### Testing Memory Efficiency
```python
def test_memory_efficiency():
    """Test loader memory usage."""
    loader = MLXDataLoader(
        dataset=create_large_dataset(),
        batch_size=1000,
        memory_efficient=True,
    )
    
    initial_memory = get_memory_usage()
    
    # Process batches
    for batch in loader:
        process_batch(batch)
    
    final_memory = get_memory_usage()
    assert final_memory - initial_memory < 1_000_000_000  # Less than 1GB
```

### Testing Data Augmentation
```python
def test_augmentation():
    """Test data augmentation."""
    augmenter = TextAugmenter(
        techniques=["synonym", "deletion", "swap"],
        augment_prob=0.5,
    )
    
    original = "The quick brown fox jumps over the lazy dog"
    augmented = augmenter(original)
    
    assert augmented != original  # Should be different
    assert len(augmented.split()) > 5  # Should preserve length roughly
```

### Testing Error Handling
```python
def test_corrupt_data_handling():
    """Test handling of corrupt data."""
    loader = RobustDataLoader(
        dataset=create_corrupt_dataset(),
        error_handling="skip",
    )
    
    valid_batches = []
    for batch in loader:
        valid_batches.append(batch)
    
    assert len(valid_batches) > 0  # Should skip corrupt, keep valid
```

## Performance Testing

### Throughput Testing
```python
@pytest.mark.benchmark
def test_loader_throughput(benchmark):
    """Test data loader throughput."""
    loader = MLXDataLoader(
        dataset=create_synthetic_dataset(size=10000),
        batch_size=64,
        num_workers=4,
    )
    
    def iterate():
        count = 0
        for batch in loader:
            count += batch["input_ids"].shape[0]
        return count
    
    result = benchmark(iterate)
    samples_per_second = result / benchmark.stats["mean"]
    
    assert samples_per_second > 1000  # At least 1000 samples/sec
```

## Debugging Tests

### Run Single Test with Output
```bash
pytest tests/data/unit/loaders/test_mlx_loader.py::TestMLXLoader::test_batching -v -s
```

### Run with Debugger
```bash
pytest tests/data/unit/core/test_interfaces.py --pdb
```

### Run with Logging
```bash
pytest tests/data/ -v --log-cli-level=DEBUG
```

## Troubleshooting

### Common Issues

1. **File Not Found**
   - Check data paths are correct
   - Ensure test fixtures create necessary files
   - Use tmp_path fixture for temporary files

2. **Memory Issues**
   - Use smaller datasets for unit tests
   - Mock large data operations
   - Test memory limits explicitly

3. **Slow Tests**
   - Mark slow tests with `@pytest.mark.slow`
   - Use synthetic data instead of real files
   - Implement data generation on-the-fly

4. **Flaky Streaming Tests**
   - Use deterministic data sources
   - Mock network operations
   - Set appropriate timeouts

5. **Cache Conflicts**
   - Always use temporary directories
   - Clear cache in test teardown
   - Use unique cache keys per test

## Contributing

When adding new data tests:
1. Follow existing test structure
2. Use appropriate fixtures
3. Test edge cases (empty data, corrupt files)
4. Add integration tests with models
5. Ensure deterministic behavior
6. Document data generation methods
7. Test both batch and streaming modes