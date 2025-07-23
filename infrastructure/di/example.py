"""Example usage of the k-bert dependency injection system.

This file demonstrates various ways to use the DI container with protocols,
decorators, and different lifecycle strategies.
"""

from typing import Protocol
from infrastructure.di import (
    injectable,
    singleton,
    provider,
    get_container,
    register_service,
    register_instance,
)


# Define protocols/interfaces

class DatabaseProtocol(Protocol):
    """Protocol for database access."""
    
    def query(self, sql: str) -> list:
        """Execute a query."""
        ...


class CacheProtocol(Protocol):
    """Protocol for caching."""
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        ...
        
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        ...


class ConfigProtocol(Protocol):
    """Protocol for configuration access."""
    
    def get_value(self, key: str) -> Any:
        """Get configuration value."""
        ...


# Implementations using decorators

@injectable(bind_to=DatabaseProtocol)
class SqliteDatabase:
    """SQLite database implementation."""
    
    def __init__(self):
        self.connection = "sqlite:///:memory:"
        
    def query(self, sql: str) -> list:
        print(f"Executing SQL: {sql}")
        return ["result1", "result2"]


@singleton(bind_to=CacheProtocol)
class InMemoryCache:
    """In-memory cache implementation (singleton)."""
    
    def __init__(self):
        self._cache = {}
        print("Creating InMemoryCache singleton")
        
    def get(self, key: str) -> Any:
        return self._cache.get(key)
        
    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


@injectable
class ConfigService:
    """Configuration service with auto-wired dependencies."""
    
    def __init__(self, cache: CacheProtocol):
        self.cache = cache
        self._config = {"app_name": "k-bert", "version": "0.1.0"}
        
    def get_value(self, key: str) -> Any:
        # Check cache first
        cached = self.cache.get(f"config:{key}")
        if cached is not None:
            return cached
            
        # Get from config and cache it
        value = self._config.get(key)
        if value is not None:
            self.cache.set(f"config:{key}", value)
        return value


# Factory function example

@provider
def create_logger() -> "Logger":
    """Factory function to create configured logger."""
    print("Creating logger via factory")
    return Logger("app.log")


class Logger:
    """Simple logger class."""
    
    def __init__(self, filename: str):
        self.filename = filename
        
    def log(self, message: str):
        print(f"[{self.filename}] {message}")


# Service using multiple dependencies

@injectable
class UserService:
    """Service demonstrating dependency injection of multiple services."""
    
    def __init__(
        self,
        db: DatabaseProtocol,
        cache: CacheProtocol,
        config: ConfigService,
        logger: Logger,
    ):
        self.db = db
        self.cache = cache
        self.config = config
        self.logger = logger
        
    def get_user(self, user_id: str):
        # Check cache
        cached_user = self.cache.get(f"user:{user_id}")
        if cached_user:
            self.logger.log(f"User {user_id} found in cache")
            return cached_user
            
        # Query database
        users = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        if users:
            user = users[0]
            self.cache.set(f"user:{user_id}", user)
            self.logger.log(f"User {user_id} loaded from database")
            return user
            
        return None


def demo_basic_usage():
    """Demonstrate basic DI usage."""
    print("=== Basic DI Usage Demo ===\n")
    
    # Get container
    container = get_container()
    
    # Resolve services
    db = container.resolve(DatabaseProtocol)
    print(f"Database type: {type(db).__name__}")
    print(f"Query result: {db.query('SELECT * FROM users')}\n")
    
    # Singleton cache - same instance
    cache1 = container.resolve(CacheProtocol)
    cache2 = container.resolve(CacheProtocol)
    print(f"Cache instances are same: {cache1 is cache2}")
    
    cache1.set("test", "value")
    print(f"Cache get result: {cache2.get('test')}\n")
    
    # Logger factory - new instance each time
    logger1 = container.resolve(Logger)
    logger2 = container.resolve(Logger)
    print(f"Logger instances are same: {logger1 is logger2}\n")
    
    # Auto-wired service
    user_service = container.resolve(UserService)
    user_service.get_user("123")


def demo_manual_registration():
    """Demonstrate manual registration."""
    print("\n=== Manual Registration Demo ===\n")
    
    container = get_container()
    
    # Register a custom implementation
    class PostgresDatabase:
        def query(self, sql: str) -> list:
            return [f"PostgreSQL: {sql}"]
    
    # Override the default database
    register_service(DatabaseProtocol, PostgresDatabase, singleton=True)
    
    # Register an instance
    custom_config = {"custom": "config"}
    register_instance(dict, custom_config)
    
    # Resolve and use
    db = container.resolve(DatabaseProtocol)
    print(f"Database type after override: {type(db).__name__}")
    print(f"Query result: {db.query('SELECT 1')}")
    
    config_dict = container.resolve(dict)
    print(f"Config dict: {config_dict}")


def demo_configuration_injection():
    """Demonstrate configuration injection."""
    print("\n=== Configuration Injection Demo ===\n")
    
    container = get_container()
    
    # Inject configuration values
    container.inject_config("database.host", "localhost")
    container.inject_config("database.port", 5432)
    container.inject_config("app.debug", True)
    
    # Retrieve configuration
    print(f"Database host: {container.get_config('database.host')}")
    print(f"Database port: {container.get_config('database.port')}")
    print(f"Debug mode: {container.get_config('app.debug')}")
    print(f"Missing config: {container.get_config('missing', 'default')}")


def demo_child_containers():
    """Demonstrate hierarchical containers."""
    print("\n=== Child Container Demo ===\n")
    
    parent = get_container()
    child = parent.create_child()
    
    # Register in parent
    parent.register(str, "parent-value", instance=True)
    
    # Register in child (overrides parent)
    child.register(str, "child-value", instance=True)
    
    # Resolve from both
    print(f"Parent resolves: {parent.resolve(str)}")
    print(f"Child resolves: {child.resolve(str)}")
    
    # Register only in child
    child.register(int, 42, instance=True)
    print(f"Child has int: {child.has(int)}")
    print(f"Parent has int: {parent.has(int)}")


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_manual_registration()
    demo_configuration_injection()
    demo_child_containers()
    
    # Clean up
    from infrastructure.di import reset_container
    reset_container()
    print("\n=== Container reset ===")