"""Tests for service registration helpers."""

import pytest
from typing import Protocol
from infrastructure.di import (
    register_service,
    register_singleton,
    register_factory,
    register_instance,
    get_container,
    reset_container,
)


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
        # Use manual registration
        class ServiceA:
            pass
        register_service(ServiceA, ServiceA)
            
        # Use singleton registration
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