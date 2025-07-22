"""Tests for provider implementations."""

import pytest
from threading import Thread
from core.di import Container
from core.di.providers import (
    SingletonProvider,
    TransientProvider,
    FactoryProvider,
    ConfigurationProvider,
)


class TestSingletonProvider:
    """Test suite for SingletonProvider."""
    
    def test_singleton_creation(self):
        """Test that singleton provider creates only one instance."""
        class TestService:
            def __init__(self):
                self.id = id(self)
                
        provider = SingletonProvider(TestService)
        container = Container()
        
        # Get instance multiple times
        instance1 = provider.get(container)
        instance2 = provider.get(container)
        instance3 = provider.get(container)
        
        # All should be same instance
        assert instance1 is instance2
        assert instance2 is instance3
        assert instance1.id == instance2.id == instance3.id
        
    def test_singleton_with_factory(self):
        """Test singleton provider with factory function."""
        counter = 0
        
        def create_service():
            nonlocal counter
            counter += 1
            return {"instance": counter}
            
        provider = SingletonProvider(create_service)
        container = Container()
        
        # Get instance multiple times
        instance1 = provider.get(container)
        instance2 = provider.get(container)
        
        # Should be same instance, factory called only once
        assert instance1 is instance2
        assert counter == 1
        assert instance1["instance"] == 1
        
    def test_singleton_thread_safety(self):
        """Test that singleton is thread-safe."""
        class TestService:
            pass
            
        provider = SingletonProvider(TestService)
        container = Container()
        instances = []
        
        def get_instance():
            instances.append(provider.get(container))
            
        # Create multiple threads
        threads = [Thread(target=get_instance) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)
        
    def test_singleton_reset(self):
        """Test resetting singleton provider."""
        class TestService:
            pass
            
        provider = SingletonProvider(TestService)
        container = Container()
        
        # Get instance
        instance1 = provider.get(container)
        
        # Reset provider
        provider.reset()
        
        # Get instance again
        instance2 = provider.get(container)
        
        # Should be different instances
        assert instance1 is not instance2


class TestTransientProvider:
    """Test suite for TransientProvider."""
    
    def test_transient_creation(self):
        """Test that transient provider creates new instances."""
        class TestService:
            def __init__(self):
                self.id = id(self)
                
        provider = TransientProvider(TestService)
        container = Container()
        
        # Get instance multiple times
        instance1 = provider.get(container)
        instance2 = provider.get(container)
        instance3 = provider.get(container)
        
        # All should be different instances
        assert instance1 is not instance2
        assert instance2 is not instance3
        assert instance1.id != instance2.id != instance3.id
        
    def test_transient_with_factory(self):
        """Test transient provider with factory function."""
        counter = 0
        
        def create_service():
            nonlocal counter
            counter += 1
            return {"instance": counter}
            
        provider = TransientProvider(create_service)
        container = Container()
        
        # Get instance multiple times
        instance1 = provider.get(container)
        instance2 = provider.get(container)
        
        # Should be different instances, factory called each time
        assert instance1 is not instance2
        assert counter == 2
        assert instance1["instance"] == 1
        assert instance2["instance"] == 2
        
    def test_transient_reset_noop(self):
        """Test that reset does nothing for transient provider."""
        class TestService:
            pass
            
        provider = TransientProvider(TestService)
        
        # Reset should not raise error
        provider.reset()


class TestFactoryProvider:
    """Test suite for FactoryProvider."""
    
    def test_factory_basic(self):
        """Test basic factory provider."""
        def create_service():
            return {"type": "service"}
            
        provider = FactoryProvider(create_service)
        container = Container()
        
        instance = provider.get(container)
        assert instance == {"type": "service"}
        
    def test_factory_with_dependencies(self):
        """Test factory with auto-wired dependencies."""
        class DependencyA:
            pass
            
        class DependencyB:
            pass
            
        class ServiceWithDeps:
            def __init__(self, a: DependencyA, b: DependencyB):
                self.a = a
                self.b = b
                
        def create_service(dep_a: DependencyA, dep_b: DependencyB) -> ServiceWithDeps:
            return ServiceWithDeps(dep_a, dep_b)
            
        # Set up container with dependencies
        container = Container()
        container.register(DependencyA, DependencyA)
        container.register(DependencyB, DependencyB)
        
        provider = FactoryProvider(create_service)
        instance = provider.get(container)
        
        assert isinstance(instance, ServiceWithDeps)
        assert isinstance(instance.a, DependencyA)
        assert isinstance(instance.b, DependencyB)
        
    def test_factory_with_defaults(self):
        """Test factory with default parameters."""
        def create_service(required: str, optional: int = 42):
            return {"required": required, "optional": optional}
            
        container = Container()
        container.register(str, "test-value", instance=True)
        
        provider = FactoryProvider(create_service)
        instance = provider.get(container)
        
        assert instance["required"] == "test-value"
        assert instance["optional"] == 42


class TestConfigurationProvider:
    """Test suite for ConfigurationProvider."""
    
    def test_configuration_provider_basic(self):
        """Test basic configuration provider."""
        class ConfiguredService:
            def __init__(self, host: str, port: int):
                self.host = host
                self.port = port
                
        container = Container()
        container.inject_config("database", {"host": "localhost", "port": 5432})
        
        provider = ConfigurationProvider(ConfiguredService, "database")
        instance = provider.get(container)
        
        assert isinstance(instance, ConfiguredService)
        assert instance.host == "localhost"
        assert instance.port == 5432
        
    def test_configuration_provider_with_factory(self):
        """Test configuration provider with custom factory."""
        class DatabaseConnection:
            def __init__(self, connection_string: str):
                self.connection_string = connection_string
                
        def create_from_config(config: dict) -> DatabaseConnection:
            conn_str = f"{config['host']}:{config['port']}/{config['database']}"
            return DatabaseConnection(conn_str)
            
        container = Container()
        container.inject_config("db", {
            "host": "localhost",
            "port": 5432,
            "database": "mydb"
        })
        
        provider = ConfigurationProvider(
            DatabaseConnection,
            "db",
            create_from_config
        )
        instance = provider.get(container)
        
        assert isinstance(instance, DatabaseConnection)
        assert instance.connection_string == "localhost:5432/mydb"
        
    def test_configuration_provider_singleton(self):
        """Test that configuration provider acts as singleton."""
        class ConfiguredService:
            def __init__(self, value: str):
                self.value = value
                self.id = id(self)
                
        container = Container()
        container.inject_config("service", {"value": "test"})
        
        provider = ConfigurationProvider(ConfiguredService, "service")
        
        # Get multiple times
        instance1 = provider.get(container)
        instance2 = provider.get(container)
        
        # Should be same instance
        assert instance1 is instance2
        assert instance1.id == instance2.id
        
    def test_configuration_provider_reset(self):
        """Test resetting configuration provider."""
        class ConfiguredService:
            def __init__(self, value: str):
                self.value = value
                
        container = Container()
        container.inject_config("service", {"value": "test"})
        
        provider = ConfigurationProvider(ConfiguredService, "service")
        
        # Get instance
        instance1 = provider.get(container)
        
        # Reset
        provider.reset()
        
        # Get again
        instance2 = provider.get(container)
        
        # Should be different instances
        assert instance1 is not instance2