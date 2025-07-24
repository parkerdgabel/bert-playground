# Enhanced Dependency Injection with Decorators

This document describes the enhanced decorator-based dependency injection system that extends the existing DI infrastructure with modern patterns inspired by Spring, FastAPI, and other frameworks.

## Overview

The enhanced DI system provides:
- 🎯 **Decorator-based component marking** - Clean, declarative component registration
- 🔧 **Advanced autowiring** - Support for Optional, List, Set, and qualified dependencies
- 🔍 **Auto-discovery** - Automatic component scanning and registration
- 🏗️ **Hexagonal architecture support** - Built-in layer validation and enforcement
- ⚙️ **Configuration injection** - Inject configuration values directly into constructors
- 🔄 **Lifecycle management** - Post-construct, pre-destroy, and scope control

## Quick Start

```python
from infrastructure.di import (
    service, repository, adapter, use_case,
    get_container, auto_discover_and_register
)

# Define a domain service
@service
class UserService:
    def validate_user(self, user: dict) -> bool:
        return user.get("age", 0) >= 18

# Define a repository
@repository
class UserRepository:
    def __init__(self):
        self._users = {}
    
    def save(self, user: dict) -> str:
        user_id = str(len(self._users) + 1)
        self._users[user_id] = user
        return user_id

# Define a use case with auto-wired dependencies
@use_case
class CreateUserUseCase:
    def __init__(self, user_service: UserService, user_repo: UserRepository):
        self.user_service = user_service
        self.user_repo = user_repo
    
    def execute(self, user_data: dict) -> str:
        if not self.user_service.validate_user(user_data):
            raise ValueError("Invalid user")
        return self.user_repo.save(user_data)

# Auto-discover and use
container = get_container()
auto_discover_and_register(container.core_container, ["your.module.path"])

# Or manually register
container.core_container.register_decorator(UserService)
container.core_container.register_decorator(UserRepository)
container.core_container.register_decorator(CreateUserUseCase)

# Resolve and use
use_case = container.resolve(CreateUserUseCase)
user_id = use_case.execute({"name": "Alice", "age": 25})
```

## Component Decorators

### Layer-Specific Decorators

```python
# Domain layer
@service(scope=Scope.SINGLETON)  # Pure business logic
class PricingService:
    pass

# Infrastructure layer
@repository  # Data persistence
class OrderRepository:
    pass

@adapter(EmailPort, name="smtp")  # External system adapters
class SmtpEmailAdapter:
    pass

# Application layer
@use_case  # Use case orchestration
class PlaceOrderUseCase:
    pass

# Other components
@factory(produces=Logger)  # Factory classes
class LoggerFactory:
    pass

@handler  # Event/command handlers
class OrderEventHandler:
    pass
```

### Lifecycle Management

```python
@service(scope=Scope.SINGLETON)  # Singleton (default for services)
class CacheService:
    pass

@service(scope=Scope.TRANSIENT)  # New instance each time
class RequestHandler:
    pass

@service
@lazy()  # Lazy initialization
class ExpensiveService:
    @post_construct
    def initialize(self):
        """Called after construction and dependency injection."""
        self.connect()
    
    @pre_destroy
    def cleanup(self):
        """Called before destruction."""
        self.disconnect()
```

## Advanced Autowiring

### Optional Dependencies

```python
@service
class NotificationService:
    def __init__(self, 
                 email: EmailPort,
                 sms: Optional[SmsPort] = None):
        self.email = email
        self.sms = sms  # Will be None if not available
```

### Collection Injection

```python
@service
class PluginManager:
    def __init__(self, plugins: List[Plugin]):
        """Injects all registered Plugin implementations."""
        self.plugins = plugins

@service
class HandlerRegistry:
    def __init__(self, handlers: Set[EventHandler]):
        """Injects unique set of handlers."""
        self.handlers = handlers
```

### Qualified Injection

```python
from typing import Annotated

@adapter(CachePort, name="redis")
class RedisCache:
    pass

@adapter(CachePort, name="memory")
@primary()  # Default when no qualifier specified
class MemoryCache:
    pass

@service
class DataService:
    def __init__(self,
                 default_cache: CachePort,  # Uses @primary
                 redis: Annotated[CachePort, qualifier("redis")]):
        self.default_cache = default_cache
        self.redis = redis
```

### Configuration Injection

```python
@service
class DatabaseService:
    def __init__(self,
                 host: Annotated[str, value("db.host", "localhost")],
                 port: Annotated[int, value("db.port", 5432)]):
        self.host = host
        self.port = port

# Inject configuration
container.core_container.inject_config("db.host", "prod.db.example.com")
container.core_container.inject_config("db.port", 5433)
```

## Conditional Registration

### Profile-Based

```python
@service
@profile("development")
class MockPaymentService:
    pass

@service
@profile("production")
class StripePaymentService:
    pass

# Activate profiles during scanning
scanner = ComponentScanner(container)
scanner.set_active_profiles({"production"})
```

### Custom Conditions

```python
@service
@conditional(lambda: os.getenv("FEATURE_X_ENABLED") == "true")
class FeatureXService:
    pass
```

## Auto-Discovery

```python
from infrastructure.di import auto_discover_and_register

# Scan packages for decorated components
auto_discover_and_register(
    container.core_container,
    package_paths=[
        "domain.services",
        "application.use_cases",
        "adapters.primary",
        "adapters.secondary"
    ],
    profiles={"production"},
    validate=True  # Check for circular dependencies
)
```

## Architecture Validation

The scanner can validate hexagonal architecture rules:

```python
scanner = ComponentScanner(container)
scanner.scan_packages(["your.app"])

# Validate architecture
errors = scanner.validate_architecture()
if errors:
    for error in errors:
        print(f"Architecture violation: {error}")

# Generate report
print(scanner.generate_report())
```

## Best Practices

1. **Use appropriate decorators** for each architectural layer
2. **Prefer constructor injection** over property injection
3. **Use interfaces (Protocols)** for loose coupling
4. **Leverage @primary** for default implementations
5. **Use profiles** for environment-specific components
6. **Validate architecture** regularly to maintain clean boundaries

## Migration from Existing Code

The new decorators are fully backward compatible:

```python
# Old style (still works)
@injectable(singleton=True)
class OldService:
    pass

# New style (recommended)
@service
class NewService:
    pass
```

## Examples

See the following files for complete examples:
- `decorator_example.py` - Comprehensive usage examples
- `tests/infrastructure/di/test_decorators.py` - Decorator tests
- `tests/infrastructure/di/test_enhanced_container.py` - Container tests
- `tests/infrastructure/di/test_scanner.py` - Scanner tests