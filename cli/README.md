# K-BERT CLI with Hexagonal Architecture

This directory contains the new CLI implementation using the hexagonal architecture pattern.

## Structure

```
cli/
├── __init__.py              # Package initialization
├── main.py                  # Main entry point with full functionality
├── main_simple.py           # Simplified entry point for testing
├── bootstrap.py             # Full dependency injection setup
├── bootstrap_simple.py      # Simplified bootstrap for testing
├── commands/
│   ├── __init__.py
│   ├── train.py            # Train command adapter
│   ├── evaluate.py         # Evaluate command adapter
│   ├── predict.py          # Predict command adapter
│   └── info.py             # Info command adapter
└── config/
    ├── __init__.py
    └── loader.py           # Configuration loading utilities
```

## Design Principles

1. **Thin Adapter Layer**: The CLI commands are thin adapters that:
   - Parse command-line arguments
   - Convert arguments to DTOs
   - Delegate to application layer commands
   - Format and display results

2. **No Business Logic**: All business logic resides in:
   - Application layer (commands, use cases)
   - Domain layer (services, entities)

3. **Dependency Injection**: The bootstrap module:
   - Wires up all adapters (compute, storage, monitoring, etc.)
   - Registers services with the DI container
   - Provides commands with their dependencies

## Usage

### Using the simplified CLI (for testing):

```bash
# Show help
uv run python -m cli.main_simple --help

# Show version
uv run python -m cli.main_simple version

# Show system info
uv run python -m cli.main_simple info

# Initialize configuration
uv run python -m cli.main_simple config init

# Show configuration
uv run python -m cli.main_simple config show
```

### Using the full CLI (when all adapters are ready):

```bash
# Train a model
k-bert train --config configs/production.yaml

# Evaluate a model
k-bert evaluate --model output/best_model --data data/test.csv

# Generate predictions
k-bert predict --model output/best_model --input data/test.csv --output predictions.csv

# Show system info
k-bert info --all
```

## Command Structure

### Train Command
- Loads configuration from multiple sources
- Creates `TrainingRequestDTO`
- Calls `TrainModelCommand.execute()`
- Displays progress and results

### Evaluate Command
- Loads model and data configuration
- Creates `EvaluationRequestDTO`
- Calls `EvaluateModelCommand.execute()`
- Displays metrics and confusion matrix

### Predict Command
- Loads model and input data
- Creates `PredictionRequestDTO`
- Calls `PredictCommand.execute()`
- Saves predictions in specified format

### Info Command
- Shows system information
- Displays configuration
- Lists available models and adapters
- Shows run details

## Configuration

The CLI uses a hierarchical configuration system:

1. **User Config**: `~/.k-bert/config.yaml`
2. **Project Config**: `./k-bert.yaml`
3. **Command Config**: `--config file.yaml`
4. **CLI Arguments**: Direct overrides

Configuration is merged in order, with later sources overriding earlier ones.

## Integration with Hexagonal Architecture

The CLI integrates with the hexagonal architecture as follows:

```
CLI (Primary Adapter)
    ↓
Application Layer (Commands/Use Cases)
    ↓
Domain Layer (Services/Entities)
    ↓
Secondary Adapters (Compute/Storage/Monitoring)
```

Each CLI command:
1. Acts as a primary adapter
2. Converts user input to application DTOs
3. Calls application commands
4. Handles responses and displays results

## Next Steps

1. Fix import issues in the full bootstrap
2. Implement missing application commands
3. Add more configuration validation
4. Add progress bars for long operations
5. Implement interactive mode
6. Add shell completion support