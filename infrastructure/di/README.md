# k-bert Dependency Injection System

A clean, protocol-based dependency injection system for k-bert built on top of Kink, providing enhanced features for managing dependencies in the k-bert MLX-based BERT implementation.

## Features

- **Protocol-based Service Registration**: Register implementations for Python protocols/interfaces
- **Multiple Lifecycles**: Singleton, transient, and factory patterns
- **Auto-wiring**: Automatic dependency resolution based on type hints
- **Configuration Injection**: Built-in support for configuration management
- **Hierarchical Containers**: Parent-child container relationships
- **Decorators**: Clean decorator-based registration
- **Testing Support**: Easy mocking and testing with isolated containers

## Quick Start

### Basic Usage

```python
from infrastructure.di import injectable, singleton, get_container

# Register a transient service
@injectable
class DataLoader:
    def load(self, path: str):
        return f"Loading from {path}"

# Register a singleton service
@singleton
class ConfigManager:
    def __init__(self):
        self.config = {"app": "k-bert"}

# Resolve services
container = get_container()
loader = container.resolve(DataLoader)
config = container.resolve(ConfigManager)
```

### Protocol-based Registration

```python
from typing import Protocol
from infrastructure.di import injectable

class DatabaseProtocol(Protocol):
    def query(self, sql: str) -> list:
        ...

@injectable(bind_to=DatabaseProtocol)
class SQLiteDatabase:
    def query(self, sql: str) -> list:
        return ["result1", "result2"]

# Resolve by protocol
db = container.resolve(DatabaseProtocol)  # Returns SQLiteDatabase instance
```

### Auto-wiring Dependencies

```python
@injectable
class UserService:
    def __init__(self, db: DatabaseProtocol, config: ConfigManager):
        self.db = db
        self.config = config

# Dependencies are automatically injected
user_service = container.resolve(UserService)
```

### Factory Functions

```python
from infrastructure.di import provider

@provider
def create_logger() -> Logger:
    """Factory function for creating configured loggers."""
    return Logger("app.log")

# Singleton factory
@provider(singleton=True)
def create_database() -> Database:
    """Creates a singleton database connection."""
    return Database("sqlite:///:memory:")
```

## Decorators

### `@injectable`

Marks a class as injectable with optional parameters:

```python
@injectable                           # Transient lifecycle
@injectable(singleton=True)           # Singleton lifecycle
@injectable(bind_to=ProtocolType)     # Bind to protocol
```

### `@singleton`

Shorthand for singleton registration:

```python
@singleton
class CacheService:
    pass

@singleton(bind_to=CacheProtocol)
class RedisCache:
    pass
```

### `@provider`

Registers factory functions:

```python
@provider
def create_service() -> Service:
    return Service()

@provider(bind_to=ServiceProtocol, singleton=True)
def create_singleton_service():
    return ServiceImpl()
```

## Registration Functions

For manual registration without decorators:

```python
from infrastructure.di import (
    register_service,
    register_singleton,
    register_factory,
    register_instance,
)

# Register service
register_service(ServiceType, ServiceImpl)

# Register singleton
register_singleton(CacheType, CacheImpl)

# Register factory
register_factory(LoggerType, create_logger)

# Register instance
config = {"key": "value"}
register_instance(dict, config)
```

## Configuration Injection

```python
container = get_container()

# Inject configuration
container.inject_config("database.host", "localhost")
container.inject_config("database.port", 5432)

# Retrieve configuration
host = container.get_config("database.host")
port = container.get_config("database.port", default=3306)
```

## Testing with DI

Create isolated containers for testing:

```python
from infrastructure.di import Container

def test_user_service():
    # Create test container
    test_container = Container()
    
    # Register mocks
    class MockDatabase:
        def query(self, sql):
            return ["test_user"]
    
    test_container.register(DatabaseProtocol, MockDatabase)
    
    # Test with mocked dependencies
    service = test_container.auto_wire(UserService)
    assert service.get_user("123") == "test_user"
```

## Child Containers

Create hierarchical containers for scoped dependencies:

```python
parent = get_container()
child = parent.create_child()

# Register in parent
parent.register(GlobalService, GlobalServiceImpl)

# Register in child (can override parent)
child.register(RequestService, RequestServiceImpl)

# Child can resolve both
global_svc = child.resolve(GlobalService)
request_svc = child.resolve(RequestService)
```

## Lifecycle Management

### Singleton
- Single instance per container
- Lazy initialization
- Thread-safe creation

### Transient
- New instance for each resolution
- No caching
- Useful for stateful services

### Factory
- Custom creation logic
- Can be singleton or transient
- Supports parameterized creation

## Integration with k-bert

The DI system is designed to wire up k-bert components cleanly:

```python
from infrastructure.di import injectable, singleton

@singleton(bind_to=ConfigManagerProtocol)
class KBertConfigManager:
    """Manages k-bert configuration."""
    pass

@injectable(bind_to=ModelFactoryProtocol)
class MLXModelFactory:
    """Creates MLX models."""
    def __init__(self, config: ConfigManagerProtocol):
        self.config = config

@injectable(bind_to=TrainerProtocol)
class MLXTrainer:
    """Trains models using MLX."""
    def __init__(
        self,
        config: ConfigManagerProtocol,
        model_factory: ModelFactoryProtocol,
    ):
        self.config = config
        self.model_factory = model_factory
```

## Best Practices

1. **Use Protocols**: Define protocols for your services to enable easy testing and swapping of implementations
2. **Prefer Constructor Injection**: Use constructor parameters for dependencies rather than property injection
3. **Singleton for Stateless Services**: Use singleton lifecycle for stateless services like factories and managers
4. **Transient for Stateful Services**: Use transient lifecycle for services that maintain state
5. **Avoid Service Locator**: Don't pass the container around; use constructor injection instead
6. **Test with Isolated Containers**: Create new containers for tests to avoid cross-test contamination

## Advanced Features

### Custom Providers

Create custom providers for complex lifecycle management:

```python
from infrastructure.di.providers import Provider

class CustomProvider(Provider):
    def get(self, container):
        # Custom creation logic
        return MyService()
    
    def reset(self):
        # Custom cleanup
        pass
```

### Configuration-based Services

Use `ConfigurationProvider` for services created from configuration:

```python
from infrastructure.di.providers import ConfigurationProvider

provider = ConfigurationProvider(
    DatabaseConnection,
    "database",
    lambda config: DatabaseConnection(config["connection_string"])
)
```

## Thread Safety

- Singleton creation is thread-safe with double-check locking
- Container operations are thread-safe
- Transient and factory providers are stateless

## Performance Considerations

- Singleton resolution is O(1) after first creation
- Transient resolution involves object creation overhead
- Protocol resolution adds one additional lookup
- Auto-wiring uses reflection but results are cached

## Troubleshooting

### Common Issues

1. **Circular Dependencies**: Refactor to break the cycle or use factory pattern
2. **Missing Dependencies**: Ensure all dependencies are registered before resolution
3. **Protocol Not Found**: Make sure to use `bind_to` parameter when registering implementations
4. **Singleton Not Working**: Check that you're using the same container instance

### Debug Logging

The DI system uses loguru for debug logging:

```python
import logging
logging.getLogger("core.di").setLevel(logging.DEBUG)
```

## Examples

See `core/di/example.py` and `core/di/integration_example.py` for comprehensive examples of using the DI system.