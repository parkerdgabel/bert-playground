# Training Architecture Refactor

## Overview

The training architecture has been refactored to follow hexagonal architecture principles with clear separation of concerns between domain, application, and infrastructure layers.

## Key Changes

### 1. Domain Layer (Pure Business Logic)

**New Components:**
- `PureTrainingService` - Contains only training decisions and strategies
- `TrainingSession` entity - Represents a training session with its state
- `Hyperparameters` value object - Immutable training configuration
- Domain events for training lifecycle

**Key Characteristics:**
- No framework dependencies (no MLX, PyTorch, etc.)
- Pure business logic for training decisions
- Makes decisions like: when to stop, when to checkpoint, when to evaluate
- Analyzes training progress and suggests adjustments

### 2. Application Layer (Orchestration)

**New Components:**
- `TrainingOrchestrator` - Coordinates the training workflow
- `TrainModelUseCaseV2` - Clean use case interface
- `TrainingExecutor` port - Interface for training execution
- `TrainingMonitor` port - Interface for progress monitoring

**Responsibilities:**
- Orchestrates between domain logic and infrastructure
- Manages the training loop flow
- Handles checkpoint management
- Coordinates evaluation and monitoring

### 3. Infrastructure Layer (Technical Implementation)

**New Components:**
- `MLXTrainingExecutor` - MLX implementation of training execution
- Handles actual forward/backward passes
- Manages optimizers and schedulers
- Performs gradient computations

**Responsibilities:**
- Framework-specific implementations
- Low-level training operations
- Memory management
- Model compilation

## Training Flow

```
1. CLI Command
   ↓
2. TrainModelUseCaseV2 (Application)
   - Validates request
   - Converts DTOs to domain objects
   ↓
3. TrainingOrchestrator (Application)
   - Creates training plan via PureTrainingService
   - Initializes training via MLXTrainingExecutor
   ↓
4. Training Loop
   - MLXTrainingExecutor executes steps
   - PureTrainingService makes decisions
   - TrainingOrchestrator coordinates
   ↓
5. Results returned through layers
```

## Benefits

1. **Testability**: Each layer can be tested independently
2. **Flexibility**: Easy to swap MLX for PyTorch by implementing new adapter
3. **Maintainability**: Clear responsibilities for each component
4. **Business Logic Preservation**: Training strategies remain pure and framework-agnostic

## Example: Adding PyTorch Support

To add PyTorch support, you would only need to:
1. Create `PyTorchTrainingExecutor` implementing `TrainingExecutor` port
2. Register it in the DI container
3. No changes needed to domain or use case layers

## Key Design Decisions

1. **Domain Service is Pure**: `PureTrainingService` only makes decisions, doesn't execute
2. **Orchestrator Pattern**: Application layer orchestrates between domain and infrastructure
3. **Port Interfaces**: Clear contracts between layers
4. **State Management**: Training state is a domain concept, but storage is infrastructure

## Migration Guide

To use the new architecture:

```python
# Old way
from application.use_cases.train_model import TrainModelUseCase

# New way
from application.use_cases.train_model_v2 import TrainModelUseCaseV2
```

The new use case has a simpler interface and delegates complexity to the orchestrator.