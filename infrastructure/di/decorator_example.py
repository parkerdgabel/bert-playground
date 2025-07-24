"""Example demonstrating enhanced decorator patterns for dependency injection.

This example shows how to use the new decorators with hexagonal architecture
principles, including qualifiers, configuration injection, and auto-discovery.
"""

from typing import List, Optional, Protocol
from typing_extensions import Annotated

from infrastructure.di import (
    # Decorators
    service, repository, adapter, use_case, port,
    qualifier, primary, value, profile, post_construct, pre_destroy,
    
    # Container and scanner
    get_container, reset_container, auto_discover_and_register,
    
    # Types
    Scope,
)


# Define ports (interfaces) for hexagonal architecture

@port
class NotificationPort(Protocol):
    """Port for sending notifications."""
    
    def send(self, recipient: str, message: str) -> bool:
        """Send a notification."""
        ...


@port  
class UserRepositoryPort(Protocol):
    """Port for user data persistence."""
    
    def find_by_id(self, user_id: str) -> Optional[dict]:
        """Find user by ID."""
        ...
        
    def save(self, user: dict) -> str:
        """Save user and return ID."""
        ...


@port
class CachePort(Protocol):
    """Port for caching."""
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        ...
        
    def set(self, key: str, value: str, ttl: int = 300) -> None:
        """Set value in cache with TTL."""
        ...


# Domain services (pure business logic)

@service(scope=Scope.SINGLETON)
class UserDomainService:
    """Domain service for user business logic."""
    
    def __init__(self):
        self.validation_rules = {
            "min_age": 18,
            "max_name_length": 100,
        }
        print(f"Creating {self.__class__.__name__}")
        
    def validate_user(self, user: dict) -> bool:
        """Validate user data according to business rules."""
        age = user.get("age", 0)
        name = user.get("name", "")
        
        return (
            age >= self.validation_rules["min_age"] and
            len(name) <= self.validation_rules["max_name_length"]
        )
        
    def calculate_user_score(self, user: dict) -> int:
        """Calculate user score based on business logic."""
        base_score = 100
        if user.get("verified", False):
            base_score += 50
        if user.get("premium", False):
            base_score += 100
        return base_score


# Infrastructure adapters

@adapter(NotificationPort, name="email", priority=10)
@primary()
class EmailNotificationAdapter:
    """Email implementation of notification port."""
    
    def __init__(self, smtp_host: Annotated[str, value("email.smtp_host", "localhost")]):
        self.smtp_host = smtp_host
        print(f"Creating {self.__class__.__name__} with SMTP host: {smtp_host}")
        
    def send(self, recipient: str, message: str) -> bool:
        print(f"Sending email to {recipient} via {self.smtp_host}: {message}")
        return True


@adapter(NotificationPort, name="sms", priority=5)
@profile("production")
class SMSNotificationAdapter:
    """SMS implementation of notification port."""
    
    def __init__(self):
        print(f"Creating {self.__class__.__name__}")
        
    def send(self, recipient: str, message: str) -> bool:
        print(f"Sending SMS to {recipient}: {message}")
        return True


@adapter(NotificationPort, name="push")
@profile("mobile")
class PushNotificationAdapter:
    """Push notification implementation."""
    
    def send(self, recipient: str, message: str) -> bool:
        print(f"Sending push notification to {recipient}: {message}")
        return True


@repository
class InMemoryUserRepository:
    """In-memory implementation of user repository."""
    
    def __init__(self):
        self._users = {}
        self._next_id = 1
        print(f"Creating {self.__class__.__name__}")
        
    @post_construct
    def initialize(self):
        """Initialize repository with sample data."""
        print("Initializing repository with sample data")
        self._users["1"] = {"id": "1", "name": "Alice", "age": 25, "verified": True}
        self._users["2"] = {"id": "2", "name": "Bob", "age": 30, "premium": True}
        
    def find_by_id(self, user_id: str) -> Optional[dict]:
        return self._users.get(user_id)
        
    def save(self, user: dict) -> str:
        user_id = str(self._next_id)
        self._next_id += 1
        user["id"] = user_id
        self._users[user_id] = user
        return user_id
        
    @pre_destroy
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up repository")
        self._users.clear()


@adapter(CachePort, name="memory")
class InMemoryCacheAdapter:
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        self._cache = {}
        print(f"Creating {self.__class__.__name__}")
        
    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
        
    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._cache[key] = value


@adapter(CachePort, name="redis")
@profile("production")
class RedisCacheAdapter:
    """Redis cache implementation (mock)."""
    
    def __init__(self, 
                 host: Annotated[str, value("redis.host", "localhost")],
                 port: Annotated[int, value("redis.port", 6379)]):
        self.host = host
        self.port = port
        print(f"Creating {self.__class__.__name__} connected to {host}:{port}")
        
    def get(self, key: str) -> Optional[str]:
        print(f"Getting {key} from Redis")
        return None
        
    def set(self, key: str, value: str, ttl: int = 300) -> None:
        print(f"Setting {key} in Redis with TTL={ttl}")


# Application layer use cases

@use_case
class CreateUserUseCase:
    """Use case for creating a new user."""
    
    def __init__(self,
                 user_service: UserDomainService,
                 user_repo: UserRepositoryPort,
                 notification: NotificationPort,
                 cache: Annotated[CachePort, qualifier("memory")],
                 notification_channels: Optional[List[NotificationPort]] = None):
        self.user_service = user_service
        self.user_repo = user_repo
        self.notification = notification
        self.cache = cache
        self.notification_channels = notification_channels or []
        print(f"Creating {self.__class__.__name__} with {len(self.notification_channels)} channels")
        
    def execute(self, user_data: dict) -> dict:
        """Create a new user."""
        # Validate
        if not self.user_service.validate_user(user_data):
            raise ValueError("Invalid user data")
            
        # Calculate score
        user_data["score"] = self.user_service.calculate_user_score(user_data)
        
        # Save
        user_id = self.user_repo.save(user_data)
        
        # Cache
        self.cache.set(f"user:{user_id}", user_data["name"])
        
        # Notify
        message = f"Welcome {user_data['name']}!"
        self.notification.send(user_data.get("email", ""), message)
        
        # Notify through all channels
        for channel in self.notification_channels:
            channel.send(user_data.get("email", ""), message)
        
        return {"id": user_id, "status": "created"}


@use_case
class GetUserUseCase:
    """Use case for retrieving a user."""
    
    def __init__(self,
                 user_repo: UserRepositoryPort,
                 cache: CachePort):
        self.user_repo = user_repo
        self.cache = cache
        print(f"Creating {self.__class__.__name__}")
        
    def execute(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        # Check cache first
        cached_name = self.cache.get(f"user:{user_id}")
        if cached_name:
            print(f"Cache hit for user {user_id}: {cached_name}")
            
        # Get from repository
        user = self.user_repo.find_by_id(user_id)
        
        # Update cache if found
        if user and not cached_name:
            self.cache.set(f"user:{user_id}", user["name"])
            
        return user


def demo_basic_usage():
    """Demonstrate basic decorator usage."""
    print("=== Basic Decorator Usage Demo ===\n")
    
    # Reset container
    reset_container()
    container = get_container()
    
    # Inject configuration
    container.core_container.inject_config("email.smtp_host", "mail.example.com")
    container.core_container.inject_config("redis.host", "redis.example.com")
    container.core_container.inject_config("redis.port", 6380)
    
    # Manual registration of decorated components
    for cls in [
        UserDomainService,
        EmailNotificationAdapter,
        InMemoryUserRepository,
        InMemoryCacheAdapter,
        CreateUserUseCase,
        GetUserUseCase,
    ]:
        container.core_container.register_decorator(cls)
        
    # Resolve and use
    create_use_case = container.resolve(CreateUserUseCase)
    get_use_case = container.resolve(GetUserUseCase)
    
    # Create a user
    print("\nCreating user...")
    result = create_use_case.execute({
        "name": "Charlie",
        "age": 28,
        "email": "charlie@example.com",
        "verified": True,
    })
    print(f"Result: {result}")
    
    # Get the user
    print("\nGetting user...")
    user = get_use_case.execute(result["id"])
    print(f"User: {user}")


def demo_auto_discovery():
    """Demonstrate auto-discovery with profiles."""
    print("\n\n=== Auto-Discovery Demo ===\n")
    
    # Reset container
    reset_container()
    container = get_container()
    
    # Inject configuration
    container.core_container.inject_config("email.smtp_host", "smtp.gmail.com")
    
    # This would normally scan actual packages
    # For demo, we'll manually register our demo components
    print("Simulating auto-discovery...")
    
    # Register all components defined in this module
    import sys
    current_module = sys.modules[__name__]
    
    from infrastructure.di import get_component_metadata
    import inspect
    
    discovered = []
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if get_component_metadata(obj):
            discovered.append(obj)
            container.core_container.register_decorator(obj)
            
    print(f"Discovered {len(discovered)} components")
    
    # Test resolution
    create_use_case = container.resolve(CreateUserUseCase)
    print("\nSuccessfully resolved CreateUserUseCase with all dependencies")


def demo_qualifiers():
    """Demonstrate qualifier-based injection."""
    print("\n\n=== Qualifier Demo ===\n")
    
    # Reset container
    reset_container()
    container = get_container()
    
    # Register multiple cache implementations with qualifiers
    container.core_container.register_decorator(InMemoryCacheAdapter)
    container.core_container.register(
        CachePort, 
        InMemoryCacheAdapter,
        metadata=get_component_metadata(InMemoryCacheAdapter)
    )
    container.core_container._qualifiers["memory"][CachePort] = InMemoryCacheAdapter
    
    # Register other components
    for cls in [UserDomainService, EmailNotificationAdapter, 
                InMemoryUserRepository, CreateUserUseCase]:
        container.core_container.register_decorator(cls)
        
    # Resolve - will use the qualified cache
    create_use_case = container.resolve(CreateUserUseCase)
    print("Successfully resolved CreateUserUseCase with qualified cache")


def demo_profiles():
    """Demonstrate profile-based registration."""
    print("\n\n=== Profile Demo ===\n")
    
    # Reset container
    reset_container()
    container = get_container()
    
    # Set production profile
    print("Setting production profile...")
    
    # Register components that match production profile
    for cls in [UserDomainService, EmailNotificationAdapter, SMSNotificationAdapter,
                RedisCacheAdapter, InMemoryUserRepository]:
        metadata = get_component_metadata(cls)
        if metadata and (not metadata.profiles or "production" in metadata.profiles):
            container.core_container.register_decorator(cls)
            
    # Check available notification adapters
    print("\nAvailable NotificationPort implementations:")
    for impl in [EmailNotificationAdapter, SMSNotificationAdapter]:
        if container.core_container.has(impl):
            print(f"  - {impl.__name__}")


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_auto_discovery()
    demo_qualifiers()
    demo_profiles()
    
    # Clean up
    reset_container()
    print("\n\n=== Container reset ===")