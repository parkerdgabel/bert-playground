# CLI Test Suite

This directory contains comprehensive tests for the BERT Playground CLI, following the project's established testing standards and patterns.

## Test Organization

```
tests/cli/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_validators.py   # Input validation functions
│   ├── test_decorators.py   # CLI decorators
│   ├── test_console.py      # Console output utilities
│   ├── test_config.py       # Configuration management
│   ├── test_contracts.py    # API contract validation
│   └── test_base_command.py # Base command class
├── integration/             # Integration tests for command workflows
│   ├── test_core_integration.py    # Train, predict, benchmark, info
│   ├── test_kaggle_integration.py  # Kaggle API integration
│   ├── test_mlflow_integration.py  # MLflow integration
│   └── test_model_integration.py   # Model management
├── e2e/                     # End-to-end workflow tests
│   ├── test_training_workflow.py   # Complete training pipelines
│   ├── test_kaggle_workflow.py     # Competition workflows
│   └── test_model_serving.py       # Model deployment
├── contract/                # API stability tests
│   ├── test_command_contracts.py   # Command interface contracts
│   └── test_parameter_contracts.py # Parameter validation
└── fixtures/                # Test utilities and helpers
    ├── commands.py          # Command-specific fixtures
    ├── mocks.py            # Mock objects
    ├── data.py             # Test data generators
    └── utils.py            # Testing utilities
```

## Running Tests

### Run all CLI tests
```bash
pytest tests/cli/
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/cli/unit/

# Integration tests only  
pytest tests/cli/integration/

# Run with markers
pytest -m unit tests/cli/
pytest -m integration tests/cli/
pytest -m e2e tests/cli/
```

### Run specific test files
```bash
pytest tests/cli/unit/test_validators.py
pytest tests/cli/integration/test_core_integration.py -v
```

### Run with coverage
```bash
pytest tests/cli/ --cov=cli --cov-report=html
```

## Test Patterns

### 1. Unit Tests

Unit tests focus on testing individual functions and classes in isolation:

```python
class TestValidatePath:
    """Test path validation."""
    
    def test_valid_existing_file(self, tmp_path):
        """Test validation of existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        
        result = validate_path(file_path, must_exist=True)
        assert result == file_path
```

### 2. Integration Tests

Integration tests verify command execution and component interaction:

```python
@pytest.mark.integration
class TestTrainCommand:
    """Test the train command integration."""
    
    def test_train_basic(self, runner, temp_project, titanic_data):
        """Test basic training command."""
        result = runner.invoke(app, [
            "train",
            "--train", str(titanic_data["train"]),
            "--val", str(titanic_data["val"])
        ])
        
        assert_success(result)
        assert "Training completed" in result.stdout
```

### 3. E2E Tests

End-to-end tests validate complete user workflows:

```python
@pytest.mark.e2e
class TestKaggleWorkflow:
    """Test complete Kaggle competition workflow."""
    
    def test_download_train_submit(self, runner, temp_project):
        """Test full competition workflow."""
        # Download data
        # Train model
        # Generate predictions
        # Submit to Kaggle
```

## Key Fixtures

### Global Fixtures (conftest.py)

- `runner`: CLI test runner
- `temp_project`: Temporary project structure
- `mock_kaggle_api`: Mocked Kaggle API
- `mock_mlflow`: Mocked MLflow tracking
- `titanic_data`: Sample dataset

### Command Fixtures (fixtures/commands.py)

- `train_args`: Common training arguments
- `predict_args`: Common prediction arguments
- `successful_train_result`: Mock training results

### Mock Objects (fixtures/mocks.py)

- `MockBertModel`: Simplified BERT model
- `MockTrainer`: Training process mock
- `MockKaggleAPI`: Kaggle API simulation

### Data Generators (fixtures/data.py)

- `generate_titanic_data()`: Create sample Titanic dataset
- `generate_training_config()`: Create config files
- `generate_model_checkpoint()`: Create mock checkpoints

### Test Utilities (fixtures/utils.py)

- `assert_success()`: Verify command success
- `assert_failure()`: Verify expected failures
- `parse_table_output()`: Extract table data
- `assert_file_created()`: Check file creation

## Testing Best Practices

### 1. Isolation

Each test should be independent and not rely on other tests:

```python
def setup_method(self):
    """Reset state before each test."""
    cli.utils.console._console = None
```

### 2. Mocking

Mock external dependencies to ensure fast, reliable tests:

```python
with patch("kaggle.api") as mock_api:
    mock_api.competition_list.return_value = [...]
    # Test code
```

### 3. Temporary Files

Use pytest's `tmp_path` fixture for file operations:

```python
def test_save_config(self, tmp_path):
    config_file = tmp_path / "config.yaml"
    save_config({"test": "value"}, config_file)
    assert config_file.exists()
```

### 4. Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_valid_batch_sizes(self, batch_size):
    result = validate_batch_size(batch_size)
    assert result == batch_size
```

### 5. Error Testing

Always test error conditions and edge cases:

```python
def test_invalid_path(self):
    with pytest.raises(typer.BadParameter, match="Path does not exist"):
        validate_path(Path("/nonexistent"), must_exist=True)
```

## Adding New Tests

When adding new CLI features:

1. **Add unit tests** for any new utility functions
2. **Add integration tests** for new commands
3. **Update fixtures** if new mocks are needed
4. **Add contract tests** to ensure API stability
5. **Document** any new testing patterns

## Debugging Tests

### Enable verbose output
```bash
pytest tests/cli/ -vv
```

### Run specific test by name
```bash
pytest tests/cli/ -k "test_train_basic"
```

### Drop into debugger on failure
```bash
pytest tests/cli/ --pdb
```

### Show print statements
```bash
pytest tests/cli/ -s
```

## Coverage Goals

- **Unit tests**: 90%+ coverage of utilities
- **Integration tests**: All command paths
- **E2E tests**: Critical workflows
- **Contract tests**: 100% of public API

## CI/CD Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Nightly builds

Failed tests will block merges to maintain code quality.