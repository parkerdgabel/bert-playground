"""Tests for the DI container implementation."""

import pytest
from typing import Protocol
from infrastructure.di import Container, get_container, reset_container


class TestContainer:
    """Test suite for Container class."""
    
    def setup_method(self):
        """Reset container before each test."""
        reset_container()
        
    def teardown_method(self):
        """Clean up after each test."""
        reset_container()
        
    def test_basic_registration_and_resolution(self):
        """Test basic service registration and resolution."""
        container = Container()
        
        class TestService:
            pass
            
        # Register service
        container.register(TestService, TestService)
        
        # Resolve service
        instance = container.resolve(TestService)
        assert isinstance(instance, TestService)
        
    def test_singleton_lifecycle(self):
        """Test singleton lifecycle management."""
        container = Container()
        
        class SingletonService:
            pass
            
        # Register as singleton
        container.register(SingletonService, SingletonService, singleton=True)
        
        # Resolve multiple times
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        
        # Should be same instance
        assert instance1 is instance2
        
    def test_transient_lifecycle(self):
        """Test transient lifecycle management."""
        container = Container()
        
        class TransientService:
            pass
            
        # Register as transient (default)
        container.register(TransientService, TransientService)
        
        # Resolve multiple times
        instance1 = container.resolve(TransientService)
        instance2 = container.resolve(TransientService)
        
        # Should be different instances
        assert instance1 is not instance2
        
    def test_factory_registration(self):
        """Test factory function registration."""
        container = Container()
        
        class FactoryCreatedService:
            def __init__(self, value: str):
                self.value = value
                
        # Factory function
        def create_service():
            return FactoryCreatedService("factory-created")
            
        # Register factory
        container.register(FactoryCreatedService, create_service, factory=True)
        
        # Resolve
        instance = container.resolve(FactoryCreatedService)
        assert isinstance(instance, FactoryCreatedService)
        assert instance.value == "factory-created"
        
    def test_instance_registration(self):
        """Test instance registration."""
        container = Container()
        
        # Create instance
        test_instance = {"key": "value"}
        
        # Register instance
        container.register(dict, test_instance, instance=True)
        
        # Resolve should return same instance
        resolved = container.resolve(dict)
        assert resolved is test_instance
        
    def test_protocol_registration(self):
        """Test protocol-based registration."""
        container = Container()
        
        class ServiceProtocol(Protocol):
            def operation(self) -> str:
                ...
                
        class ServiceImplementation:
            def operation(self) -> str:
                return "implemented"
                
        # Register implementation for protocol
        container.register(ServiceProtocol, ServiceImplementation)
        
        # Resolve protocol should return implementation
        instance = container.resolve(ServiceProtocol)
        assert isinstance(instance, ServiceImplementation)
        assert instance.operation() == "implemented"
        
    def test_auto_wiring(self):
        """Test automatic dependency injection."""
        container = Container()
        
        class DependencyA:
            pass
            
        class DependencyB:
            pass
            
        class ServiceWithDeps:
            def __init__(self, dep_a: DependencyA, dep_b: DependencyB):
                self.dep_a = dep_a
                self.dep_b = dep_b
                
        # Register dependencies
        container.register(DependencyA, DependencyA)
        container.register(DependencyB, DependencyB)
        
        # Auto-wire should work
        instance = container.auto_wire(ServiceWithDeps)
        assert isinstance(instance, ServiceWithDeps)
        assert isinstance(instance.dep_a, DependencyA)
        assert isinstance(instance.dep_b, DependencyB)
        
    def test_auto_wiring_with_defaults(self):
        """Test auto-wiring with default parameters."""
        container = Container()
        
        class ServiceWithDefaults:
            def __init__(self, required: str, optional: int = 42):
                self.required = required
                self.optional = optional
                
        # Register only required dependency
        container.register(str, "test-value", instance=True)
        
        # Auto-wire should use default for optional
        instance = container.auto_wire(ServiceWithDefaults)
        assert instance.required == "test-value"
        assert instance.optional == 42
        
    def test_configuration_injection(self):
        """Test configuration value injection."""
        container = Container()
        
        # Inject configuration
        container.inject_config("database.host", "localhost")
        container.inject_config("database.port", 5432)
        
        # Retrieve configuration
        assert container.get_config("database.host") == "localhost"
        assert container.get_config("database.port") == 5432
        assert container.get_config("missing.key", "default") == "default"
        
    def test_child_containers(self):
        """Test hierarchical container resolution."""
        parent = Container()
        child = parent.create_child()
        
        class ParentService:
            pass
            
        class ChildService:
            pass
            
        # Register in parent
        parent.register(ParentService, ParentService)
        
        # Register in child
        child.register(ChildService, ChildService)
        
        # Child can resolve both
        assert child.has(ParentService)
        assert child.has(ChildService)
        
        # Parent can only resolve its own
        assert parent.has(ParentService)
        assert not parent.has(ChildService)
        
        # Resolution works correctly
        parent_instance = child.resolve(ParentService)
        assert isinstance(parent_instance, ParentService)
        
    def test_child_container_override(self):
        """Test child container overriding parent registrations."""
        parent = Container()
        child = parent.create_child()
        
        # Register in parent
        parent.register(str, "parent-value", instance=True)
        
        # Override in child
        child.register(str, "child-value", instance=True)
        
        # Resolution returns correct values
        assert parent.resolve(str) == "parent-value"
        assert child.resolve(str) == "child-value"
        
    def test_resolve_unregistered_raises_error(self):
        """Test that resolving unregistered service raises error."""
        container = Container()
        
        class UnregisteredService:
            pass
            
        with pytest.raises(KeyError) as exc_info:
            container.resolve(UnregisteredService)
            
        assert "UnregisteredService not registered" in str(exc_info.value)
        
    def test_clear_container(self):
        """Test clearing container registrations."""
        container = Container()
        
        class TestService:
            pass
            
        # Register and verify
        container.register(TestService, TestService)
        assert container.has(TestService)
        
        # Clear and verify
        container.clear()
        assert not container.has(TestService)
        
    def test_global_container(self):
        """Test global container singleton."""
        container1 = get_container()
        container2 = get_container()
        
        # Should be same instance
        assert container1 is container2
        
        # Reset should create new instance
        reset_container()
        container3 = get_container()
        assert container3 is not container1