# Training Module Test Suite

Comprehensive test suite for the declarative training module with MLX optimizations and Kaggle integration.

## Test Structure

```
tests/training/
├── unit/                    # Unit tests for individual components
│   ├── core/               # Core module tests
│   │   ├── test_protocols.py
│   │   ├── test_config.py
│   │   ├── test_base_trainer.py
│   │   ├── test_optimization.py
│   │   └── test_state.py
│   ├── callbacks/          # Callback tests
│   │   ├── test_base_callback.py
│   │   ├── test_early_stopping.py
│   │   ├── test_model_checkpoint.py
│   │   └── test_mlflow_callback.py
│   ├── metrics/            # Metrics tests
│   │   ├── test_base_metrics.py
│   │   ├── test_classification_metrics.py
│   │   └── test_regression_metrics.py
│   ├── kaggle/             # Kaggle-specific tests
│   │   └── test_kaggle_trainer.py
│   └── test_factory.py     # Factory pattern tests
├── integration/            # Integration tests
│   ├── test_trainer_callbacks.py
│   ├── test_trainer_metrics.py
│   └── test_checkpoint_recovery.py
├── e2e/                    # End-to-end tests
│   ├── test_classification_training.py
│   └── test_kaggle_competition.py
├── fixtures/               # Shared test fixtures
│   ├── models.py          # Mock models
│   ├── datasets.py        # Synthetic datasets
│   ├── configs.py         # Test configurations
│   └── utils.py           # Testing utilities
├── conftest.py            # Pytest configuration
└── README.md              # This file
```

## Running Tests

### Run All Tests
```bash
# Run all tests with coverage
pytest tests/training/ -v --cov=training --cov-report=html

# Run with markers
pytest tests/training/ -m unit      # Unit tests only
pytest tests/training/ -m integration # Integration tests only
pytest tests/training/ -m e2e        # End-to-end tests only
```

### Run Specific Test Categories
```bash
# Core module tests
pytest tests/training/unit/core/ -v

# Callback tests
pytest tests/training/unit/callbacks/ -v

# Metrics tests
pytest tests/training/unit/metrics/ -v

# Kaggle tests
pytest tests/training/unit/kaggle/ -v
```

### Run Individual Test Files
```bash
# Test protocols
pytest tests/training/unit/core/test_protocols.py -v

# Test configuration
pytest tests/training/unit/core/test_config.py -v

# Test base trainer
pytest tests/training/unit/core/test_base_trainer.py -v
```

### Run with Specific Markers
```bash
# Fast tests only (exclude slow tests)
pytest tests/training/ -v -m "not slow"

# MLflow tests
pytest tests/training/ -v -m mlflow

# Kaggle tests
pytest tests/training/ -v -m kaggle
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
pytest tests/training/ --cov=training --cov-report=html

# Open report
open htmlcov/index.html
```

## Test Fixtures

### Models (`fixtures/models.py`)
- `SimpleBinaryClassifier`: Basic binary classification model
- `SimpleMulticlassClassifier`: Multiclass classification model
- `SimpleRegressor`: Regression model
- `BrokenModel`: Model that raises errors (for error testing)
- `NaNModel`: Model that produces NaN values
- `MemoryIntensiveModel`: Large model for memory testing

### Datasets (`fixtures/datasets.py`)
- `SyntheticDataLoader`: Basic synthetic data
- `ImbalancedDataLoader`: Imbalanced classification data
- `NoisyDataLoader`: Data with label noise
- `VariableLengthDataLoader`: Variable length sequences
- `EmptyDataLoader`: Empty dataset (edge case)
- `SingleBatchDataLoader`: Single batch only

### Configurations (`fixtures/configs.py`)
- `create_test_config()`: Standard test configuration
- `create_minimal_config()`: Minimal config for fast tests
- `create_kaggle_config()`: Kaggle-specific configuration
- `create_distributed_config()`: Distributed training config
- Various scheduler and optimizer configs

### Utilities (`fixtures/utils.py`)
- Array comparison utilities
- Checkpoint creation/verification
- Mock MLflow helpers
- Metrics generation
- Training simulation

## Writing New Tests

### Unit Test Template
```python
import pytest
from training.module import Component
from tests.training.fixtures import create_test_config

class TestComponent:
    """Test Component functionality."""
    
    def test_initialization(self, tmp_path):
        """Test component initialization."""
        config = create_test_config(output_dir=tmp_path)
        component = Component(config)
        
        assert component.config == config
        # Add more assertions
    
    def test_specific_feature(self):
        """Test specific feature."""
        # Test implementation
        pass
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError):
            # Code that should raise error
            pass
```

### Integration Test Template
```python
class TestComponentIntegration:
    """Test component integration with other modules."""
    
    def test_component_with_callbacks(self, tmp_path):
        """Test component works with callbacks."""
        # Create components
        # Test interaction
        # Verify results
        pass
```

### E2E Test Template
```python
class TestEndToEnd:
    """Test complete workflows."""
    
    def test_full_training_workflow(self, tmp_path):
        """Test complete training workflow."""
        # Setup
        model = create_model()
        config = create_config()
        data = create_dataloader()
        
        # Train
        trainer = create_trainer(model, config)
        result = trainer.train(data)
        
        # Verify
        assert result.metrics["loss"] < 1.0
        assert result.best_checkpoint.exists()
```

## Common Testing Patterns

### Testing with Temporary Directories
```python
def test_with_temp_dir(tmp_path):
    """Use tmp_path fixture for temporary directories."""
    output_dir = tmp_path / "output"
    config = create_test_config(output_dir=output_dir)
    # Test code
```

### Testing with Mock Data
```python
def test_with_mock_data():
    """Use synthetic data loaders."""
    train_loader = SyntheticDataLoader(
        num_samples=100,
        batch_size=4,
        task_type="classification"
    )
    # Test code
```

### Testing Callbacks
```python
def test_callback():
    """Test callback functionality."""
    callback = MockCallback()
    trainer = create_trainer(callbacks=[callback])
    trainer.train(data_loader)
    
    assert "train_begin" in callback.events
```

### Testing Error Conditions
```python
def test_error_handling():
    """Test error conditions."""
    with pytest.raises(ValueError, match="specific error"):
        # Code that should raise error
        pass
```

## Debugging Tests

### Run Single Test with Output
```bash
pytest tests/training/unit/core/test_config.py::TestOptimizerConfig::test_validation -v -s
```

### Run with Debugger
```bash
pytest tests/training/unit/core/test_config.py --pdb
```

### Run with Logging
```bash
pytest tests/training/ -v --log-cli-level=DEBUG
```

## CI/CD Integration

### GitHub Actions Configuration
```yaml
- name: Run Training Module Tests
  run: |
    pytest tests/training/unit/ -v --cov=training
    pytest tests/training/integration/ -v -m "not slow"
```

### Pre-commit Hook
```bash
#!/bin/bash
# Run fast unit tests before commit
pytest tests/training/unit/ -v -m "not slow" --tb=short
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in PYTHONPATH
   - Check `conftest.py` adds correct paths

2. **MLX Device Issues**
   - Tests automatically use CPU device
   - Set `MLX_USE_DEFAULT_DEVICE=true`

3. **Slow Tests**
   - Use smaller datasets for unit tests
   - Mark slow tests with `@pytest.mark.slow`
   - Skip slow tests: `pytest -m "not slow"`

4. **Flaky Tests**
   - Set random seeds in fixtures
   - Use deterministic mock data
   - Avoid time-dependent assertions

## Contributing

When adding new tests:
1. Follow existing test structure
2. Use appropriate fixtures
3. Add docstrings to test methods
4. Mark tests appropriately (unit/integration/e2e)
5. Ensure tests are deterministic
6. Keep tests focused and independent