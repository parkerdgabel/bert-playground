"""Tests for service registration helpers and decorators."""

import pytest
from typing import Protocol
from core.di import (
    injectable,
    singleton,
    provider,
    register_service,
    register_singleton,
    register_factory,
    register_instance,
    get_container,
    reset_container,
)


class TestDecorators:
    """Test suite for DI decorators."""
    
    def setup_method(self):
        """Reset container before each test."""
        reset_container()
        
    def teardown_method(self):
        """Clean up after each test."""
        reset_container()
        
    def test_injectable_decorator_basic(self):
        """Test basic @injectable decorator."""
        @injectable
        class TestService:
            def __init__(self):
                self.value = "test"
                
        container = get_container()
        instance = container.resolve(TestService)
        assert isinstance(instance, TestService)
        assert instance.value == "test"
        
    def test_injectable_decorator_with_singleton(self):
        """Test @injectable with singleton parameter."""
        @injectable(singleton=True)
        class SingletonService:
            pass
            
        container = get_container()
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        assert instance1 is instance2
        
    def test_injectable_decorator_with_protocol(self):
        """Test @injectable with bind_to parameter."""
        class ServiceProtocol(Protocol):
            def operation(self) -> str:
                ...
                
        @injectable(bind_to=ServiceProtocol)
        class ServiceImpl:
            def operation(self) -> str:
                return "implemented"
                
        container = get_container()
        instance = container.resolve(ServiceProtocol)
        assert isinstance(instance, ServiceImpl)
        assert instance.operation() == "implemented"
        
    def test_singleton_decorator(self):
        """Test @singleton decorator."""
        @singleton
        class SingletonService:
            def __init__(self):
                self.id = id(self)
                
        container = get_container()
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        assert instance1 is instance2
        assert instance1.id == instance2.id
        
    def test_singleton_decorator_with_protocol(self):
        """Test @singleton with bind_to parameter."""
        class CacheProtocol(Protocol):
            def get(self, key: str) -> any:
                ...
                
        @singleton(bind_to=CacheProtocol)
        class InMemoryCache:
            def __init__(self):
                self.data = {}
                
            def get(self, key: str) -> any:
                return self.data.get(key)
                
        container = get_container()
        cache1 = container.resolve(CacheProtocol)
        cache2 = container.resolve(CacheProtocol)
        assert cache1 is cache2
        assert isinstance(cache1, InMemoryCache)
        
    def test_provider_decorator_basic(self):
        """Test @provider decorator with return type annotation."""
        class Logger:
            def __init__(self, name: str):
                self.name = name
                
        @provider
        def create_logger() -> Logger:
            return Logger("test-logger")
            
        container = get_container()
        logger = container.resolve(Logger)
        assert isinstance(logger, Logger)
        assert logger.name == "test-logger"
        
    def test_provider_decorator_with_bind_to(self):
        """Test @provider with bind_to parameter."""
        class DatabaseProtocol(Protocol):
            def connect(self) -> str:
                ...
                
        class MockDatabase:
            def connect(self) -> str:
                return "connected"
                
        @provider(bind_to=DatabaseProtocol)
        def create_database():
            return MockDatabase()
            
        container = get_container()
        db = container.resolve(DatabaseProtocol)
        assert isinstance(db, MockDatabase)
        assert db.connect() == "connected"
        
    def test_provider_decorator_singleton(self):
        """Test @provider with singleton parameter."""
        class Counter:
            def __init__(self):
                self.count = 0
                
        @provider(singleton=True)
        def create_counter() -> Counter:
            return Counter()
            
        container = get_container()
        counter1 = container.resolve(Counter)
        counter2 = container.resolve(Counter)
        assert counter1 is counter2
        
    def test_provider_without_return_annotation_raises(self):
        """Test that provider without return annotation raises error."""
        with pytest.raises(ValueError) as exc_info:
            @provider
            def bad_provider():
                return "something"
                
        assert "must have a return type annotation" in str(exc_info.value)
        
    def test_nested_decorator_usage(self):
        """Test services with dependencies using decorators."""
        @injectable
        class ConfigService:
            def __init__(self):
                self.config = {"app": "test"}
                
        @singleton
        class CacheService:
            def __init__(self):
                self.cache = {}
                
        @injectable
        class AppService:
            def __init__(self, config: ConfigService, cache: CacheService):
                self.config = config
                self.cache = cache
                
        container = get_container()
        app = container.resolve(AppService)
        assert isinstance(app, AppService)
        assert isinstance(app.config, ConfigService)
        assert isinstance(app.cache, CacheService)
        
        # Verify cache is singleton
        cache2 = container.resolve(CacheService)
        assert app.cache is cache2


class TestRegistrationFunctions:
    """Test suite for registration helper functions."""
    
    def setup_method(self):
        """Reset container before each test."""
        reset_container()
        
    def teardown_method(self):
        """Clean up after each test."""
        reset_container()
        
    def test_register_service(self):
        """Test register_service function."""
        class TestService:
            pass
            
        register_service(TestService, TestService)
        
        container = get_container()
        instance = container.resolve(TestService)
        assert isinstance(instance, TestService)
        
    def test_register_service_with_protocol(self):
        """Test registering implementation for protocol."""
        class ServiceProtocol(Protocol):
            def action(self) -> str:
                ...
                
        class ServiceImpl:
            def action(self) -> str:
                return "done"
                
        register_service(ServiceProtocol, ServiceImpl)
        
        container = get_container()
        instance = container.resolve(ServiceProtocol)
        assert isinstance(instance, ServiceImpl)
        assert instance.action() == "done"
        
    def test_register_singleton(self):
        """Test register_singleton function."""
        class SingletonService:
            pass
            
        register_singleton(SingletonService, SingletonService)
        
        container = get_container()
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        assert instance1 is instance2
        
    def test_register_factory(self):
        """Test register_factory function."""
        class FactoryService:
            def __init__(self, value: str):
                self.value = value
                
        def create_service():
            return FactoryService("factory-created")
            
        register_factory(FactoryService, create_service)
        
        container = get_container()
        instance = container.resolve(FactoryService)
        assert isinstance(instance, FactoryService)
        assert instance.value == "factory-created"
        
    def test_register_instance(self):
        """Test register_instance function."""
        test_dict = {"key": "value", "number": 42}
        register_instance(dict, test_dict)
        
        container = get_container()
        resolved = container.resolve(dict)
        assert resolved is test_dict
        assert resolved["key"] == "value"
        assert resolved["number"] == 42
        
    def test_mixed_registration_methods(self):
        """Test mixing different registration methods."""
        # Use decorator
        @injectable
        class ServiceA:
            pass
            
        # Use function
        class ServiceB:
            pass
        register_singleton(ServiceB, ServiceB)
        
        # Use factory
        class ServiceC:
            def __init__(self, a: ServiceA, b: ServiceB):
                self.a = a
                self.b = b
                
        def create_service_c():
            container = get_container()
            return ServiceC(
                container.resolve(ServiceA),
                container.resolve(ServiceB)
            )
            
        register_factory(ServiceC, create_service_c)
        
        # Resolve all
        container = get_container()
        c = container.resolve(ServiceC)
        assert isinstance(c, ServiceC)
        assert isinstance(c.a, ServiceA)
        assert isinstance(c.b, ServiceB)
        
        # Verify B is singleton
        b2 = container.resolve(ServiceB)
        assert c.b is b2