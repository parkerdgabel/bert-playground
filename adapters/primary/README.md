# Primary Adapters

This directory contains the primary adapters that handle external interfaces for the K-BERT application. Primary adapters are the entry points for external actors (users, systems) to interact with the application.

## Structure

```
primary/
├── cli/              # Command-line interface adapters
│   ├── app.py       # Main CLI application
│   ├── base.py      # Base classes and utilities
│   ├── train_adapter.py    # Training command adapter
│   └── predict_adapter.py  # Prediction command adapter
├── api/              # REST API adapters (future)
└── web/              # Web UI adapters (future)
```

## CLI Adapters

The CLI adapters provide a thin layer that:

1. **Parse Arguments**: Convert command-line arguments into structured data
2. **Create DTOs**: Build request DTOs from parsed arguments
3. **Call Use Cases**: Delegate to application use cases for business logic
4. **Format Output**: Display results in a user-friendly format

### Key Principles

- **No Business Logic**: Adapters contain zero business logic
- **DTO-Based**: All communication uses DTOs from `application/dto/`
- **Error Handling**: Consistent error display and exit codes
- **Progress Indication**: Visual feedback for long-running operations
- **Configuration**: Support for hierarchical configuration

### Example: Train Adapter

```python
# The train adapter only:
1. Parses CLI arguments
2. Creates TrainingRequestDTO
3. Calls TrainModelUseCase
4. Displays results

# No training logic exists in the adapter!
```

## API Adapters (Future)

The `api/` directory is prepared for future REST API implementation. It will follow the same principles:

- Thin adapters that parse HTTP requests
- Convert to DTOs
- Call use cases
- Return HTTP responses

## Web Adapters (Future)

The `web/` directory is prepared for future web UI implementation. It will provide:

- Server-side rendering or SPA support
- WebSocket support for real-time updates
- Same DTO-based communication

## Adding New Commands

To add a new CLI command:

1. Create a new adapter file in `cli/` (e.g., `evaluate_adapter.py`)
2. Import and use the corresponding DTO from `application/dto/`
3. Import and use the corresponding use case from `application/use_cases/`
4. Register the command in `cli/app.py`

Example structure:
```python
from application.dto.evaluation import EvaluationRequestDTO
from application.use_cases.evaluate_model import EvaluateModelUseCase

async def evaluate_command(...):
    # 1. Parse arguments
    # 2. Create DTO
    # 3. Call use case
    # 4. Display results
```

## Testing

Primary adapters should be tested with:

- **Unit tests**: Test argument parsing and DTO creation
- **Integration tests**: Test interaction with use cases
- **Contract tests**: Ensure CLI interface stability