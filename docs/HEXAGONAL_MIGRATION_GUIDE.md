# K-BERT Hexagonal Architecture Migration Guide

This guide helps you migrate from the legacy architecture to the new hexagonal architecture with dependency injection.

## Overview

The new architecture provides:
- Clear separation between domain logic and infrastructure
- Dependency injection for better testability
- Plugin system for extensibility
- Event-driven communication
- Standardized error handling

## Key Changes

### 1. Import Path Changes

**Old:**
```python
from bert_playground.data.loaders import MLXDataLoader
from bert_playground.models.factory import ModelFactory
from bert_playground.training.trainer import Trainer
```

**New:**
```python
from infrastructure.bootstrap import get_service
from infrastructure.ports.compute import ComputeBackend
from models.factory_facade import ModelFactory
from training.components.training_orchestrator import TrainingOrchestrator
```

### 2. Service Access Pattern

**Old (Direct Instantiation):**
```python
# Direct creation of services
config_manager = ConfigManager()
model_factory = ModelFactory()
trainer = Trainer(config)
```

**New (Dependency Injection):**
```python
from infrastructure.bootstrap import initialize_application, get_service

# Initialize application once
container = initialize_application()

# Get services from container
config_provider = get_service(ConfigurationProvider)
model_factory = get_service(ModelFactory)
training_orchestrator = get_service(TrainingOrchestrator)
```

### 3. Configuration Management

**Old:**
```python
config = ConfigManager().load_config("config.yaml")
merged_config = ConfigManager().merge_configs(base, override)
```

**New:**
```python
from infrastructure.ports.config import ConfigurationProvider

config_provider = get_service(ConfigurationProvider)
config = config_provider.load_config("config.yaml")
merged = config_provider.merge_configs(base, override)
```

### 4. Training Workflow

**Old:**
```python
trainer = Trainer(
    model=model,
    train_data=train_loader,
    val_data=val_loader,
    config=config
)
trainer.train()
```

**New:**
```python
from training.commands.train import TrainCommand

# Create command with dependencies injected
train_command = TrainCommand(
    model_factory=get_service(ModelFactory),
    dataset_factory=get_service(DatasetFactory),
    training_orchestrator=get_service(TrainingOrchestrator),
    storage=get_service(StorageService)
)

# Execute training
result = train_command.execute(config)
```

### 5. Plugin Development

**Old (Manual Registration):**
```python
# Custom components registered manually
ModelFactory.register_head("custom", CustomHead)
```

**New (Plugin System):**
```python
from infrastructure.plugins.base import Plugin
from infrastructure.plugins.decorators import plugin

@plugin(
    name="custom_head",
    version="1.0.0",
    category="model_head"
)
class CustomHeadPlugin(Plugin):
    def activate(self):
        # Auto-registered on discovery
        pass
```

### 6. Error Handling

**Old:**
```python
try:
    result = some_operation()
except Exception as e:
    print(f"Error: {e}")
    raise
```

**New:**
```python
from infrastructure.exceptions import ApplicationError, handle_errors

@handle_errors(ApplicationError)
def some_operation():
    # Automatic error handling and logging
    pass
```

### 7. Event System

**New Feature - Event-Driven Communication:**
```python
from infrastructure.events.bus import get_event_bus
from infrastructure.events.events import TrainingStartedEvent

# Subscribe to events
event_bus = get_event_bus()
event_bus.subscribe(TrainingStartedEvent, on_training_started)

# Publish events
event_bus.publish(TrainingStartedEvent(model_name="bert-base"))
```

## Migration Steps

### Step 1: Update Imports

Replace all old imports with new port/adapter imports:

```python
# Replace direct imports
- from bert_playground.training.trainer import Trainer
+ from training.components.training_orchestrator import TrainingOrchestrator
+ from infrastructure.bootstrap import get_service
```

### Step 2: Initialize Application

Add application initialization at the start of your script:

```python
from infrastructure.bootstrap import initialize_application

# Initialize once at startup
container = initialize_application()
```

### Step 3: Use Dependency Injection

Replace direct instantiation with service resolution:

```python
# Old
model_factory = ModelFactory()

# New
model_factory = get_service(ModelFactory)
```

### Step 4: Update Configuration Access

Use the configuration provider port:

```python
# Old
config = load_yaml("config.yaml")

# New
config_provider = get_service(ConfigurationProvider)
config = config_provider.load_config("config.yaml")
```

### Step 5: Adopt Command Pattern

Use command objects for operations:

```python
# Old
trainer.train()

# New
train_command = container.resolve(TrainCommand)
result = train_command.execute(config)
```

## Example: Complete Training Script

**Old Architecture:**
```python
from bert_playground.config import ConfigManager
from bert_playground.models.factory import ModelFactory
from bert_playground.data.loaders import create_data_loader
from bert_playground.training.trainer import Trainer

# Load configuration
config = ConfigManager().load_config("config.yaml")

# Create model
model = ModelFactory.create_model(config.model)

# Create data loaders
train_loader = create_data_loader(config.data.train)
val_loader = create_data_loader(config.data.validation)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()
```

**New Architecture:**
```python
from pathlib import Path
from infrastructure.bootstrap import initialize_application, get_service
from training.commands.train import TrainCommand
from infrastructure.ports.config import ConfigurationProvider

# Initialize application
container = initialize_application()

# Load configuration
config_provider = get_service(ConfigurationProvider)
config = config_provider.load_config("config.yaml")

# Create and execute training command
train_command = container.resolve(TrainCommand)
result = train_command.execute(config)

# Result contains all training artifacts
print(f"Model saved to: {result.model_path}")
print(f"Final metrics: {result.metrics}")
```

## CLI Migration

The CLI now uses the new architecture internally. No changes needed for end users:

```bash
# Still works the same
k-bert train --config config.yaml
k-bert predict --model model.safetensors --data test.csv
```

## Testing with New Architecture

```python
import pytest
from infrastructure.di.container import Container
from infrastructure.testing import create_test_container

def test_training_command():
    # Create test container with mocks
    container = create_test_container()
    
    # Register test doubles
    container.register(ModelFactory, MockModelFactory, instance=True)
    
    # Test command
    command = container.resolve(TrainCommand)
    result = command.execute(test_config)
    
    assert result.success
```

## Troubleshooting

### Issue: Service Not Found

```python
# Error: No registration found for type X
```

**Solution:** Ensure the application is initialized:
```python
container = initialize_application()
```

### Issue: Circular Dependencies

**Solution:** Use lazy injection or factory pattern:
```python
# Use factory instead of direct dependency
self.model_factory = get_service(ModelFactory)
model = self.model_factory.create_model(config)
```

### Issue: Plugin Not Loading

**Solution:** Check plugin structure:
```python
# Plugin must be decorated
@plugin(name="my_plugin", version="1.0.0")
class MyPlugin(Plugin):
    pass
```

## Benefits of New Architecture

1. **Testability**: Easy to mock dependencies
2. **Flexibility**: Swap implementations without changing code
3. **Maintainability**: Clear separation of concerns
4. **Extensibility**: Plugin system for custom components
5. **Observability**: Built-in monitoring and events

## Further Resources

- [Hexagonal Architecture Guide](./HEXAGONAL_ARCHITECTURE_GUIDE.md)
- [Plugin Development Guide](./PLUGIN_DEVELOPMENT_GUIDE.md)
- [Event System Guide](./EVENT_SYSTEM_GUIDE.md)
- [DI Container Documentation](./DI_CONTAINER_GUIDE.md)