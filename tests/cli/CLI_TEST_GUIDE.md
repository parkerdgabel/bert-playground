# CLI Test Suite Guide

## Overview

This directory contains comprehensive unit tests for all CLI commands in the MLX BERT playground. The tests ensure that:

1. Commands execute without errors
2. Parameters are validated correctly
3. API contracts are maintained
4. Error conditions are handled gracefully
5. Output formatting is consistent

## Test Structure

```
tests/cli/
├── __init__.py
├── test_api_contracts.py      # API contract validation tests
├── test_benchmark_command.py   # Benchmark command tests
├── test_core_commands.py       # Train, predict, info command tests
├── test_kaggle_commands.py     # Kaggle integration tests
├── test_mlflow_commands.py     # MLflow integration tests
├── test_model_commands.py      # Model management tests
├── run_cli_tests.py           # Test runner script
└── CLI_TEST_GUIDE.md          # This file
```

## Running Tests

### Run All CLI Tests
```bash
# Using the test runner
uv run python tests/cli/run_cli_tests.py

# Using pytest directly
uv run pytest tests/cli/ -v
```

### Run Specific Test File
```bash
# Run only benchmark tests
uv run pytest tests/cli/test_benchmark_command.py -v

# Using the test runner
uv run python tests/cli/run_cli_tests.py --file test_benchmark_command.py
```

### Run Tests Matching Pattern
```bash
# Run all train-related tests
uv run pytest tests/cli/ -k "train" -v

# Using the test runner
uv run python tests/cli/run_cli_tests.py --pattern "train"
```

### Run with Coverage
```bash
# Generate coverage report
uv run python tests/cli/run_cli_tests.py --coverage

# Or using pytest
uv run pytest tests/cli/ --cov=cli --cov-report=html
```

## Test Categories

### 1. Core Commands (`test_core_commands.py`)

Tests for fundamental CLI operations:

- **Train Command**
  - Basic training with CSV files
  - Training with config files
  - MLX embeddings integration
  - Hyperparameter validation
  - Checkpoint resumption
  - Error handling

- **Predict Command**
  - Basic prediction
  - Output file generation
  - Probability output
  - Batch size configuration
  - Missing checkpoint handling

- **Info Command**
  - System information display
  - Package version listing
  - Git status integration
  - JSON output format

### 2. Benchmark Command (`test_benchmark_command.py`)

Tests for performance benchmarking:

- Basic benchmark execution
- Memory tracking
- Gradient computation verification
- Model type variations
- Output formatting
- Error conditions

### 3. Kaggle Commands (`test_kaggle_commands.py`)

Tests for Kaggle integration:

- **Competitions Command**
  - List competitions
  - Filter by category/search
  - Active/all filtering
  - Tag display
  - Timestamp handling

- **Download Command**
  - Competition download
  - Custom output paths
  - Force overwrite
  - Extract options

- **Submit Command**
  - Basic submission
  - Checkpoint references
  - Error handling

### 4. MLflow Commands (`test_mlflow_commands.py`)

Tests for experiment tracking:

- **Health Command**
  - Health check execution
  - Failure reporting
  - No-parameter contract

- **Experiments Command**
  - List experiments
  - Create experiments
  - Show run counts

- **Runs Command**
  - List runs by experiment
  - Status filtering
  - Artifact display
  - Run deletion

- **UI Command**
  - UI launch
  - Custom ports
  - Backend URI configuration

### 5. Model Commands (`test_model_commands.py`)

Tests for model management:

- **Inspect Command**
  - Checkpoint inspection
  - Weight display
  - JSON output

- **Convert Command**
  - ONNX conversion
  - Quantization
  - Output paths

- **Merge Command**
  - Basic merging
  - Multiple models
  - Custom weights
  - Merge strategies

- **Compare Command**
  - Model comparison
  - Detailed output

### 6. API Contracts (`test_api_contracts.py`)

Tests ensuring stable interfaces:

- Parameter validation
- Return value structure
- Backward compatibility
- Contract documentation

## Writing New Tests

### Test Template

```python
class TestNewCommand:
    """Test suite for new-command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    def test_basic_functionality(self, runner):
        """Test basic command execution."""
        result = runner.invoke(app, ["new-command", "--option", "value"])
        
        assert result.exit_code == 0
        assert "Expected output" in result.stdout
    
    @patch('cli.commands.new.SomeDependency')
    def test_with_mocking(self, mock_dep, runner):
        """Test with mocked dependencies."""
        mock_dep.return_value.method.return_value = "result"
        
        result = runner.invoke(app, ["new-command"])
        
        assert result.exit_code == 0
        mock_dep.assert_called_once()
```

### Best Practices

1. **Use Fixtures**: Create reusable test data and mocks
2. **Mock External Dependencies**: Don't make real API calls or file system changes
3. **Test Error Conditions**: Ensure graceful failure handling
4. **Verify Output**: Check both exit codes and stdout content
5. **Test Parameter Validation**: Ensure invalid inputs are rejected
6. **Use Descriptive Names**: Make test purpose clear from the name

## Common Test Patterns

### Mocking File System
```python
@patch('cli.commands.module.Path')
def test_file_operations(self, mock_path, runner):
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.read_text.return_value = "content"
```

### Mocking API Calls
```python
@patch('cli.commands.module.requests.get')
def test_api_integration(self, mock_get, runner):
    mock_get.return_value.json.return_value = {"data": "value"}
```

### Testing with Temporary Files
```python
def test_with_temp_files(self, runner, tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("column1,column2\nvalue1,value2\n")
    
    result = runner.invoke(app, ["command", "--input", str(test_file)])
```

### Verifying Complex Outputs
```python
def test_table_output(self, runner):
    result = runner.invoke(app, ["command"])
    
    # Check for table structure
    assert "│" in result.stdout  # Table borders
    assert "Column1" in result.stdout
    assert "─" in result.stdout  # Table lines
```

## Debugging Failed Tests

### View Full Output
```bash
# Show full stdout/stderr
uv run pytest tests/cli/test_file.py::test_name -vv -s
```

### Run Single Test
```bash
# Run specific test method
uv run pytest tests/cli/test_file.py::TestClass::test_method -v
```

### Interactive Debugging
```python
# Add breakpoint in test
import pdb; pdb.set_trace()

# Or use pytest debugging
uv run pytest tests/cli/test_file.py --pdb
```

## CI/CD Integration

The test suite is integrated with GitHub Actions:

1. **On Push**: Tests run on changes to CLI or test files
2. **On PR**: Tests run and comment on failures
3. **Coverage**: Reports are generated and uploaded

See `.github/workflows/cli-contract-tests.yml` for configuration.

## Maintenance

### Adding New Commands

When adding a new CLI command:

1. Create unit tests in appropriate test file
2. Add contract tests if command calls utilities
3. Update this guide with test descriptions
4. Run full test suite to ensure no regressions

### Updating Existing Commands

When modifying CLI commands:

1. Update relevant tests to match new behavior
2. Add tests for new functionality
3. Ensure contract tests still pass
4. Document any breaking changes

## Test Metrics

Target metrics for CLI test suite:

- **Coverage**: >90% of CLI code
- **Execution Time**: <30 seconds for full suite
- **Reliability**: Zero flaky tests
- **Documentation**: All commands have tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in PYTHONPATH
2. **Mock Failures**: Check mock call counts and arguments
3. **Async Issues**: Use proper async test decorators
4. **Path Issues**: Use Path objects consistently

### Getting Help

1. Check test output for detailed error messages
2. Review similar tests for patterns
3. Use test runner's detailed report
4. Check CLI_API_CONTRACTS.md for interface specs