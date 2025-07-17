# MLX BERT Playground Test Suite

Comprehensive test suite for the MLX-based ModernBERT implementation.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── unit/                # Unit tests
│   ├── test_mlflow_central.py
│   ├── test_mlx_trainer.py
│   ├── test_data_loaders.py
│   ├── test_models.py
│   └── test_cli.py
├── integration/         # Integration tests
│   └── test_training_pipeline.py
└── fixtures/           # Test data and fixtures
```

## Running Tests

### Quick Start

```bash
# Run all fast tests (unit tests only)
./run_tests.sh

# Run all tests including slow and integration
./run_tests.sh --all

# Run specific test file
./run_tests.sh --test tests/unit/test_mlx_trainer.py

# Run specific test function
./run_tests.sh --test tests/unit/test_mlx_trainer.py::TestMLXTrainer::test_initialization

# Generate coverage report only
./run_tests.sh --coverage
```

### Test Markers

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.mlx` - Tests requiring MLX
- `@pytest.mark.mlflow` - Tests requiring MLflow

### Using pytest directly

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific markers
uv run pytest -m "not slow"
uv run pytest -m integration

# Run with verbose output
uv run pytest -vv

# Run in parallel
uv run pytest -n auto
```

## Test Coverage

Coverage reports are generated in HTML format:

```bash
# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Writing Tests

### Unit Tests

Unit tests should be fast, isolated, and test a single component:

```python
def test_specific_functionality(self, mock_dependency):
    """Test description."""
    # Arrange
    expected = "value"
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

### Integration Tests

Integration tests verify that components work together:

```python
@pytest.mark.integration
def test_full_workflow(self, sample_data):
    """Test complete workflow."""
    # Test multiple components working together
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_titanic_data` - Sample training data
- `sample_test_data` - Sample test data without labels
- `base_model` - Pre-configured ModernBERT model
- `titanic_classifier` - Classifier wrapper
- `training_config` - Minimal training configuration
- `mlx_trainer` - Pre-configured trainer instance

## Debugging Tests

```bash
# Run with full traceback
uv run pytest --tb=long

# Drop into debugger on failure
uv run pytest --pdb

# Show print statements
uv run pytest -s

# Run specific test with maximum verbosity
uv run pytest -vvs tests/unit/test_mlx_trainer.py::test_name
```

## Continuous Integration

Tests are designed to run in CI environments:

```bash
# CI-friendly output
uv run pytest --tb=short --maxfail=1

# Generate XML report for CI
uv run pytest --junitxml=test-results.xml
```

## Common Issues

1. **Import errors**: Ensure you're running from the project root
2. **MLX errors**: Some tests require Apple Silicon hardware
3. **Slow tests**: Use `--slow` flag to include slow tests
4. **Cleanup**: Tests automatically clean up temporary files

## Contributing

When adding new tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use appropriate markers (`@pytest.mark.slow`, etc.)
4. Add fixtures to `conftest.py` if reusable
5. Ensure tests are independent and can run in any order
6. Mock external dependencies in unit tests
7. Use descriptive test names that explain what is being tested