# API Contract Tests Implementation Summary

## Overview

I've implemented a comprehensive API contract testing system for the CLI to ensure interface stability between CLI commands and utility functions. This prevents the type of failures we encountered after the CLI refactoring.

## What Was Created

### 1. Contract Test Suite (`tests/cli/test_api_contracts.py`)

Comprehensive test suite covering:
- **Core Commands**: train, predict, benchmark, info
- **Kaggle Commands**: competitions, download, submit
- **MLflow Commands**: health, experiments, runs
- **Model Commands**: inspect, convert, merge

Each test validates:
- Parameter names and types
- Return value structures
- API compatibility
- Error handling contracts

### 2. Contract Documentation (`docs/CLI_API_CONTRACTS.md`)

Detailed specification of:
- All API contracts between CLI and utilities
- Expected parameter types and formats
- Return value structures
- Breaking change protocol
- Version management guidelines

### 3. Contract Validation Utilities (`cli/utils/contracts.py`)

Runtime validation tools including:
- `@validate_contract`: Decorator for parameter/return type validation
- `@backward_compatible`: Support for deprecated parameter names
- `@validate_pydantic_contract`: Pydantic-based validation
- `ContractViolation`: Exception for contract violations
- Contract test base classes

### 4. Test Runner (`run_contract_tests.py`)

Automated test runner that:
- Runs all contract tests
- Checks contract test coverage
- Reports which commands lack tests
- Provides actionable feedback

### 5. CI/CD Integration (`.github/workflows/cli-contract-tests.yml`)

GitHub Actions workflow that:
- Runs on CLI/utility changes
- Tests across Python versions
- Comments on PRs if contracts break
- Uploads test artifacts

## Key Features

### 1. Parameter Validation
```python
@validate_contract(
    expected_params={
        'model_type': str,
        'batch_size': int,
        'learning_rate': float
    },
    return_type=dict
)
def create_model(model_type: str, **kwargs) -> dict:
    ...
```

### 2. Backward Compatibility
```python
@backward_compatible({
    'page_size': 'page',  # old_name: new_name
})
def list_competitions(page: int):
    # Automatically handles old parameter names with deprecation warnings
```

### 3. Pydantic Contracts
```python
class TrainParams(ParameterContract):
    model_type: str
    batch_size: int = 32
    learning_rate: float = 2e-5

@validate_pydantic_contract(TrainParams)
def train_command(model_type: str, batch_size: int = 32, **kwargs):
    ...
```

## Contract Examples

### MLflow Health Check
```python
# Contract: run_full_check takes NO parameters
results = health_checker.run_full_check()

# Each result must have:
# - status: "PASS" or "FAIL"
# - message: str
# - suggestions: List[str] (if FAIL)
```

### Kaggle Competitions
```python
# Contract: use 'page' not 'page_size'
competitions = kaggle.list_competitions(
    category="tabular",
    search="classification",
    sort_by="prize",
    page=1  # NOT page_size!
)
```

### Model Creation
```python
# Contract: model must return dict with 'loss' key
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
assert 'loss' in outputs
```

## Benefits

1. **Early Detection**: Catches API breaks before deployment
2. **Documentation**: Contracts serve as API documentation
3. **Deprecation Management**: Graceful handling of API evolution
4. **Type Safety**: Runtime validation of parameters
5. **CI/CD Integration**: Automated testing on every change

## Usage

### Running Tests Locally
```bash
# Run contract tests
uv run pytest tests/cli/test_api_contracts.py -v

# Run with coverage report
uv run python run_contract_tests.py
```

### Adding New Contracts
1. Document the contract in `CLI_API_CONTRACTS.md`
2. Add test case to `test_api_contracts.py`
3. Optionally add runtime validation using decorators

### Handling Breaking Changes
1. Add deprecation warning in current version
2. Support both old and new APIs for 2 versions
3. Document migration path
4. Update contract tests

## Next Steps

1. **Add More Tests**: Cover remaining CLI commands
2. **Runtime Validation**: Apply contract decorators to actual commands
3. **Contract Generation**: Auto-generate from type hints
4. **Performance Monitoring**: Track contract validation overhead
5. **API Versioning**: Implement explicit version management

## Conclusion

The API contract testing system provides a safety net for CLI development, ensuring that changes to utility functions don't break CLI commands. This is especially important in a modular architecture where components evolve independently.