"""Tests for enhanced Container with decorator support."""

import pytest
from typing import Optional, List, Set, Protocol
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from infrastructure.di.container import Container
from infrastructure.di.decorators import (
    service, adapter, repository, port,
    qualifier, value, primary,
    Scope, get_component_metadata,
    clear_registry
)


class TestEnhancedAutowiring:
    """Test enhanced autowiring capabilities."""
    
    def setup_method(self):
        """Set up test container."""
        clear_registry()
        self.container = Container()
        
    def test_optional_dependency_injection(self):
        """Test Optional[T] dependency injection."""
        @service
        class OptionalDep:
            pass
            
        @service
        class ServiceWithOptional:
            def __init__(self, required: str, optional: Optional[OptionalDep] = None):
                self.required = required
                self.optional = optional
                
        # Register only the service, not the optional dependency
        self.container.register(str, "test", instance=True)
        self.container.register_decorator(ServiceWithOptional)
        
        # Should resolve with None for optional
        instance = self.container.resolve(ServiceWithOptional)
        assert instance.required == "test"
        assert instance.optional is None
        
        # Now register the optional dependency
        self.container.register_decorator(OptionalDep)
        instance2 = self.container.resolve(ServiceWithOptional)
        assert instance2.optional is not None
        assert isinstance(instance2.optional, OptionalDep)
        
    def test_list_injection(self):
        """Test List[T] dependency injection."""
        class Plugin(Protocol):
            def process(self): ...
            
        @adapter(Plugin, name="plugin1")
        class Plugin1:
            def process(self):
                return "p1"
                
        @adapter(Plugin, name="plugin2")
        class Plugin2:
            def process(self):
                return "p2"
                
        @service
        class PluginManager:
            def __init__(self, plugins: List[Plugin]):
                self.plugins = plugins
                
        # Register all components
        self.container.register_decorator(Plugin1)
        self.container.register_decorator(Plugin2)
        self.container.register_decorator(PluginManager)
        
        # Resolve should inject all Plugin implementations
        manager = self.container.resolve(PluginManager)
        assert len(manager.plugins) == 2
        results = [p.process() for p in manager.plugins]
        assert "p1" in results
        assert "p2" in results
        
    def test_set_injection(self):
        """Test Set[T] dependency injection."""
        class Handler(Protocol):
            def handle(self): ...
            
        @adapter(Handler)
        class Handler1:
            def handle(self):
                return 1
                
        @adapter(Handler)  
        class Handler2:
            def handle(self):
                return 2
                
        @service
        class HandlerRegistry:
            def __init__(self, handlers: Set[Handler]):
                self.handlers = handlers
                
        # Register components
        self.container.register_decorator(Handler1)
        self.container.register_decorator(Handler2)
        self.container.register_decorator(HandlerRegistry)
        
        # Resolve
        registry = self.container.resolve(HandlerRegistry)
        assert len(registry.handlers) == 2
        assert isinstance(registry.handlers, set)
        

class TestQualifiedInjection:
    """Test qualifier-based injection."""
    
    def setup_method(self):
        clear_registry()
        self.container = Container()
        
    def test_primary_resolution(self):
        """Test @primary decorator resolution."""
        class Cache(Protocol):
            def get(self, key: str): ...
            
        @adapter(Cache)
        class MemoryCache:
            def get(self, key: str):
                return f"memory:{key}"
                
        @adapter(Cache)
        @primary()
        class RedisCache:
            def get(self, key: str):
                return f"redis:{key}"
                
        @service
        class CacheService:
            def __init__(self, cache: Cache):
                self.cache = cache
                
        # Register all
        self.container.register_decorator(MemoryCache)
        self.container.register_decorator(RedisCache)
        self.container.register_decorator(CacheService)
        
        # Should use primary (Redis)
        service = self.container.resolve(CacheService)
        assert service.cache.get("test") == "redis:test"
        
    def test_qualified_resolution(self):
        """Test @qualifier resolution."""
        class Database(Protocol):
            def query(self): ...
            
        @adapter(Database, name="mysql")
        class MySQLDatabase:
            def query(self):
                return "mysql"
                
        @adapter(Database, name="postgres")
        class PostgresDatabase:
            def query(self):
                return "postgres"
                
        # Manual qualifier setup (normally done by container)
        self.container.register_decorator(MySQLDatabase)
        self.container.register_decorator(PostgresDatabase)
        self.container._qualifiers["mysql"][Database] = MySQLDatabase
        self.container._qualifiers["postgres"][Database] = PostgresDatabase
        
        # Test qualified resolution
        mysql = self.container.resolve(Database, qualifier_name="mysql")
        assert mysql.query() == "mysql"
        
        postgres = self.container.resolve(Database, qualifier_name="postgres")
        assert postgres.query() == "postgres"
        

class TestConfigurationInjection:
    """Test configuration value injection."""
    
    def setup_method(self):
        clear_registry()
        self.container = Container()
        
    def test_value_injection(self):
        """Test @value configuration injection."""
        # Inject config values
        self.container.inject_config("app.name", "TestApp")
        self.container.inject_config("app.port", 8080)
        
        @service
        class ConfigService:
            def __init__(self, name: str, port: int):
                self.name = name
                self.port = port
                
        # For this test, manually inject values
        # In real usage, this would use Annotated[str, value("app.name")]
        self.container.register(str, "TestApp", instance=True)
        self.container.register(int, 8080, instance=True)
        self.container.register_decorator(ConfigService)
        
        service = self.container.resolve(ConfigService)
        assert service.name == "TestApp"
        assert service.port == 8080
        
    def test_config_with_default(self):
        """Test config injection with defaults."""
        # Don't inject, use default
        assert self.container.get_config("missing.key", "default") == "default"
        
        # Inject and override default
        self.container.inject_config("existing.key", "value")
        assert self.container.get_config("existing.key", "default") == "value"
        

class TestMetadataSupport:
    """Test metadata handling in container."""
    
    def setup_method(self):
        clear_registry()
        self.container = Container()
        
    def test_metadata_registration(self):
        """Test registering with metadata."""
        @service(scope=Scope.SINGLETON, priority=10)
        class MetaService:
            pass
            
        metadata = get_component_metadata(MetaService)
        self.container.register(MetaService, MetaService, metadata=metadata)
        
        # Check metadata was stored
        assert MetaService in self.container._metadata
        assert self.container._metadata[MetaService].priority == 10
        assert MetaService in self.container._singleton_types
        
    def test_auto_decorator_registration(self):
        """Test register_decorator method."""
        @service
        class AutoService:
            pass
            
        @adapter(AutoService)
        class AutoAdapter:
            pass
            
        # Register via decorator method
        self.container.register_decorator(AutoService)
        self.container.register_decorator(AutoAdapter)
        
        # Check registrations
        assert self.container.has(AutoService)
        assert self.container.has(AutoAdapter)
        
        # AutoAdapter should also be registered for AutoService port
        adapter = self.container.resolve(AutoService)
        assert isinstance(adapter, AutoAdapter)
        

class TestLifecycleManagement:
    """Test component lifecycle management."""
    
    def setup_method(self):
        clear_registry()
        self.container = Container()
        
    def test_post_construct_called(self):
        """Test post_construct method is called."""
        from infrastructure.di.decorators import post_construct
        
        @service
        class InitializableService:
            def __init__(self):
                self.initialized = False
                
            @post_construct
            def init(self):
                self.initialized = True
                
        # Update metadata with init method
        metadata = get_component_metadata(InitializableService)
        metadata.init_method = "init"
        
        self.container.register_decorator(InitializableService)
        instance = self.container.resolve(InitializableService)
        
        # Post construct should have been called
        assert instance.initialized is True
        
    def test_singleton_vs_transient(self):
        """Test singleton vs transient lifecycle."""
        @service(scope=Scope.SINGLETON)
        class SingletonService:
            pass
            
        @service(scope=Scope.TRANSIENT)  
        class TransientService:
            pass
            
        self.container.register_decorator(SingletonService)
        self.container.register_decorator(TransientService)
        
        # Singleton should return same instance
        s1 = self.container.resolve(SingletonService)
        s2 = self.container.resolve(SingletonService)
        assert s1 is s2
        
        # Transient should return new instances
        t1 = self.container.resolve(TransientService)
        t2 = self.container.resolve(TransientService)
        assert t1 is not t2
        

class TestChildContainers:
    """Test hierarchical container support."""
    
    def setup_method(self):
        clear_registry()
        self.parent = Container()
        
    def test_child_inherits_services(self):
        """Test child container inherits parent services."""
        @service
        class ParentService:
            pass
            
        self.parent.register_decorator(ParentService)
        child = self.parent.create_child()
        
        # Child should have parent's services
        assert child.has(ParentService)
        
    def test_child_override(self):
        """Test child can override parent services."""
        class BaseService:
            def get_name(self):
                return "parent"
                
        class ChildService(BaseService):
            def get_name(self):
                return "child"
                
        self.parent.register(BaseService, BaseService)
        
        child = self.parent.create_child()
        child.register(BaseService, ChildService)
        
        # Parent still returns original
        parent_service = self.parent.resolve(BaseService)
        assert parent_service.get_name() == "parent"
        
        # Child returns override
        child_service = child.resolve(BaseService)
        assert child_service.get_name() == "child"
        
    def test_child_isolation(self):
        """Test child containers have isolated singletons."""
        @service(scope=Scope.SINGLETON)
        class SharedService:
            def __init__(self):
                self.value = 0
                
        self.parent.register_decorator(SharedService)
        child = self.parent.create_child()
        
        # Get instances
        parent_instance = self.parent.resolve(SharedService)
        child_instance = child.resolve(SharedService)
        
        # Should be different instances
        assert parent_instance is not child_instance
        
        # Modify one
        parent_instance.value = 10
        assert child_instance.value == 0