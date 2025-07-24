# CLAUDE.md - k-bert Development Guide

This file provides essential guidance to Claude Code when working with the k-bert MLX-based ModernBERT implementation for Kaggle competitions.

## Project Overview

k-bert is a state-of-the-art BERT implementation using Apple's MLX framework, designed to solve Kaggle tabular problems using text-based approaches. The project emphasizes configuration-driven development, plugin architecture, and efficient training on Apple Silicon.

### Core Technologies
- **MLX**: Apple's ML framework optimized for Apple Silicon
- **ModernBERT**: Answer.AI's 2024 BERT variant with RoPE, GeGLU, alternating attention
- **uv**: Fast Python package manager
- **Typer**: Modern CLI framework
- **Pydantic**: Configuration validation
- **MLflow**: Experiment tracking
- **Loguru**: Structured logging

## Project Structure

```
bert-playground/
├── domain/                 # Core business logic (Hexagonal Architecture)
│   ├── entities/          # Domain entities
│   ├── services/          # Domain services
│   └── value_objects/     # Value objects
├── application/            # Application layer (Use Cases)
│   ├── use_cases/         # Application use cases
│   ├── services/          # Application services
│   └── dto/               # Data transfer objects
├── infrastructure/         # Infrastructure layer
│   ├── bootstrap.py       # Application initialization
│   ├── config/            # Configuration management
│   ├── di/                # Dependency injection system
│   └── plugins/           # Plugin system
├── adapters/               # Adapters (Hexagonal Architecture)
│   ├── primary/           # Driving adapters (CLI, API)
│   │   └── cli/           # CLI application (k-bert command)
│   └── secondary/         # Driven adapters
│       ├── compute/       # MLX compute adapters
│       ├── storage/       # Storage adapters
│       ├── tokenizer/     # Tokenizer adapters
│       └── monitoring/    # Monitoring adapters
├── ports/                  # Port definitions (interfaces)
│   ├── primary/           # Driving ports
│   └── secondary/         # Driven ports
├── models/                 # Model implementations
│   ├── bert/              # BERT architectures
│   ├── heads/             # Task-specific heads
│   ├── lora/              # LoRA adapters
│   └── factory.py         # Model creation
├── data/                   # Data pipeline
│   ├── base/              # Base classes and protocols
│   ├── loaders/           # MLX-optimized data loading
│   ├── templates/         # Text conversion templates
│   └── factory.py         # Dataset creation
├── training/              # Training infrastructure
│   ├── base/              # Base trainer classes
│   ├── callbacks/         # Training callbacks
│   └── metrics/           # Evaluation metrics
├── configs/               # Example configurations
└── tests/                 # Comprehensive test suite
```

## Essential Commands

```bash
# Installation
uv sync

# Configuration-first workflow
k-bert config init            # Initialize user config
k-bert project init myproj    # Create new project
k-bert run                    # Run from project directory

# Core training commands
k-bert train --config configs/production.yaml
k-bert predict --checkpoint output/run_*/checkpoints/final --output predictions.csv
k-bert benchmark --batch-size 64

# System information
k-bert info --all
```

## Configuration System

The project uses a hierarchical configuration system:

1. **User Config**: `~/.k-bert/config.yaml` - Personal preferences
2. **Project Config**: `k-bert.yaml` - Project-specific settings
3. **Command Config**: `--config file.yaml` - Run-specific overrides
4. **CLI Arguments**: Direct command-line overrides

### Key Configuration Sections
- `models`: Model architecture and hyperparameters
- `training`: Training settings (epochs, batch size, learning rate)
- `data`: Data loading and preprocessing
- `checkpoint`: Checkpointing and model saving
- `logging`: Logging configuration
- `mlflow`: Experiment tracking settings

## Plugin System

k-bert supports custom components through a plugin system:

```python
# Project structure with plugins
myproject/
├── k-bert.yaml
├── src/
│   ├── heads/         # Custom task heads
│   ├── augmenters/    # Data augmentation
│   └── features/      # Feature engineering
└── data/
```

Plugins are automatically discovered and registered when running from a project directory.

## MLX Optimization

- **Batch Size**: Use 32-64 for optimal performance
- **Compilation**: Enabled by default, disable with `use_compilation: false`
- **Prefetching**: Set `prefetch_size: 4-8` for data loading
- **LoRA**: Use for memory-efficient fine-tuning
- **Pre-tokenization**: Enable with `--use-pretokenized` for faster loading

## Architecture Overview

### Hexagonal Architecture (Ports and Adapters)

The project follows hexagonal architecture principles to ensure clean separation of concerns:

1. **Domain Layer** (`domain/`): Pure business logic with no external dependencies
   - Domain services contain core algorithms and business rules
   - Entities represent core domain concepts
   - No framework or infrastructure code

2. **Application Layer** (`application/`): Orchestrates domain logic
   - Use cases define application workflows
   - Application services coordinate between domain and infrastructure
   - DTOs handle data transformation

3. **Ports** (`ports/`): Define interfaces (contracts)
   - Primary ports: Interfaces for driving adapters (e.g., CLI commands)
   - Secondary ports: Interfaces for driven adapters (e.g., storage, compute)
   - All ports are Protocol-based for type safety

4. **Adapters** (`adapters/`): Implement port interfaces
   - Primary adapters: CLI, REST API (future)
   - Secondary adapters: MLX compute, file storage, HuggingFace tokenizers
   - Adapters are swappable via configuration

5. **Infrastructure** (`infrastructure/`): Cross-cutting concerns
   - Bootstrap: Application initialization
   - DI Container: Dependency injection and lifecycle management
   - Configuration: Hierarchical config system
   - Plugins: Extension mechanism

### Dependency Injection System

The project uses a decorator-based DI system for automatic component discovery and wiring:

#### Component Decorators

```python
from infrastructure.di import service, adapter, port, use_case, repository

# Domain service
@service(scope="singleton")
class TrainingService:
    def __init__(self, compute: ComputeBackend):
        self.compute = compute

# Adapter implementation
@adapter(port=StorageService, priority=100)
class FileStorageAdapter:
    def save(self, path: str, data: Any) -> None:
        # Implementation

# Port definition
@port
class StorageService(Protocol):
    def save(self, path: str, data: Any) -> None: ...
```

#### Key Features

1. **Auto-discovery**: Components are automatically discovered via decorators
2. **Scope Management**: Singleton or transient lifecycle
3. **Circular Dependency Detection**: Validates dependency graph at startup
4. **Configuration-driven**: Adapters can be selected via configuration
5. **Type Safety**: Full type checking with Protocol support

#### Available Decorators

- `@service`: Domain services (business logic)
- `@adapter`: Port implementations
- `@port`: Port interfaces (protocols)
- `@use_case`: Application use cases
- `@repository`: Data repositories
- `@factory`: Factory classes
- `@component`: Generic components

#### Container Usage

```python
# Bootstrap automatically initializes the container
from infrastructure.bootstrap import initialize_application

container = initialize_application()

# Resolve services
training_service = container.resolve(TrainingService)
storage = container.resolve(StorageService)  # Gets configured adapter

# Health check
health = container.health_check()
print(f"Services: {health['services_count']}")
print(f"Adapters: {health['adapters']}")
```

## Development Best Practices

### Repository Hygiene
- **Temporary Files**: ALWAYS use `/tmp` for debug scripts and temporary files
- **Documentation**: Only create docs when explicitly requested
- **Clean State**: Leave the repository in a polished state
- **Commits**: Make focused commits with clear messages

### Architecture Guidelines
- **Domain Purity**: Domain layer must not depend on infrastructure
- **Port-Adapter Pattern**: Always define ports before implementing adapters
- **DI Decorators**: Use appropriate decorators for automatic registration
- **Configuration Over Code**: Prefer configuration for adapter selection

### Testing Requirements
- **Mandatory Testing**: ALWAYS run tests after changes
- **Test Structure**: Mirror code structure in tests/
  - `unit/`: Individual component tests
  - `integration/`: Component interaction tests
  - `e2e/`: Complete workflow tests
  - `contract/`: CLI stability tests
- **Test New Features**: Add tests for all new functionality

### Code Style
- Follow existing patterns and conventions
- Use type hints and docstrings
- Prefer configuration over hardcoding
- Use protocols for extensibility

## Key Implementation Details

### Model Factory
- Uses `modernbert_with_head` as default model type
- Automatically selects appropriate head based on labels
- Supports classic BERT, ModernBERT, and custom architectures

### Data Pipeline
- Protocol-based design for flexibility
- MLX-optimized loaders with zero-copy operations
- Automatic caching for pre-tokenized data
- Template system for tabular-to-text conversion

### Training Infrastructure
- Declarative configuration with YAML/JSON
- Callback system for extensibility
- Automatic MLflow integration
- Comprehensive checkpointing with resume support

### CLI Design
- Typer-based with rich console output
- Hierarchical command structure
- Comprehensive error handling
- Configuration-first approach

## Testing Commands

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest tests/cli/unit/
uv run pytest tests/cli/integration/
uv run pytest tests/cli/contract/

# Run with coverage
uv run pytest --cov=bert_playground --cov-report=html

# Run specific test
uv run pytest tests/cli/unit/test_config_manager.py -v
```

## Important Notes

- Always use `uv run` for Python execution
- Package name is `k-bert` on PyPI
- Entry point is `k-bert` command (replaces `bert_cli.py`)
- Configuration files use `k-bert.yaml` naming
- MLX performs best with batch sizes that are powers of 2
- Pre-tokenization cache stored in `data/.tokenizer_cache/`
- Training logs automatically saved to `{run_dir}/training.log`

## Quick Reference

### File Locations
- User config: `~/.k-bert/config.yaml`
- Project config: `./k-bert.yaml`
- Plugin source: `./src/`
- Output directory: `./output/`
- MLflow tracking: `./output/mlruns/`

### Environment Variables
- `K_BERT_CONFIG_PATH`: Override config location
- `LOGURU_LEVEL`: Set logging level (DEBUG, INFO, etc.)
- `K_BERT_CACHE_DIR`: Override cache directory

### Common Workflows
1. **New Project**: `k-bert config init` → `k-bert project init` → customize → `k-bert run`
2. **Training**: Edit config → `k-bert train --config myconfig.yaml`
3. **Experimentation**: Use different configs with MLflow tracking
4. **Plugin Development**: Add to src/ → test → use in config