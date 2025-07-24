"""Tests for enhanced DI decorators."""

import pytest
from typing import Protocol, Optional, List

from infrastructure.di.decorators import (
    service, adapter, repository, use_case, port,
    primary, qualifier, value, profile, 
    post_construct, pre_destroy,
    Scope, ComponentType, ComponentMetadata,
    get_component_metadata, get_registered_components,
    clear_registry
)


class TestBasicDecorators:
    """Test basic decorator functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def test_service_decorator(self):
        """Test @service decorator."""
        @service
        class TestService:
            pass
            
        metadata = get_component_metadata(TestService)
        assert metadata is not None
        assert metadata.component_type == ComponentType.DOMAIN_SERVICE
        assert metadata.scope == Scope.SINGLETON
        assert "domain" in metadata.tags
        assert "service" in metadata.tags
        
    def test_service_with_options(self):
        """Test @service with custom options."""
        @service(name="custom", scope=Scope.TRANSIENT, priority=10)
        class CustomService:
            pass
            
        metadata = get_component_metadata(CustomService)
        assert metadata.name == "custom"
        assert metadata.scope == Scope.TRANSIENT
        assert metadata.priority == 10
        
    def test_adapter_decorator(self):
        """Test @adapter decorator."""
        class TestPort(Protocol):
            def do_something(self): ...
            
        @adapter(TestPort, name="test_impl", priority=5)
        class TestAdapter:
            def do_something(self):
                return "done"
                
        metadata = get_component_metadata(TestAdapter)
        assert metadata.component_type == ComponentType.ADAPTER
        assert metadata.port_type == TestPort
        assert metadata.name == "test_impl"
        assert metadata.priority == 5
        assert TestPort in metadata.interfaces
        
    def test_repository_decorator(self):
        """Test @repository decorator."""
        @repository
        class UserRepository:
            pass
            
        metadata = get_component_metadata(UserRepository)
        assert metadata.component_type == ComponentType.REPOSITORY
        assert metadata.scope == Scope.SINGLETON
        assert "repository" in metadata.tags
        assert "infrastructure" in metadata.tags
        
    def test_use_case_decorator(self):
        """Test @use_case decorator."""
        @use_case(name="create_order")
        class CreateOrderUseCase:
            pass
            
        metadata = get_component_metadata(CreateOrderUseCase)
        assert metadata.component_type == ComponentType.USE_CASE
        assert metadata.name == "create_order"
        assert metadata.scope == Scope.TRANSIENT
        assert "application" in metadata.tags
        

class TestQualifierDecorators:
    """Test qualifier and primary decorators."""
    
    def setup_method(self):
        clear_registry()
        
    def test_primary_decorator(self):
        """Test @primary decorator."""
        @adapter(Protocol)
        @primary()
        class PrimaryAdapter:
            pass
            
        metadata = get_component_metadata(PrimaryAdapter)
        assert metadata.priority == 1000
        assert metadata.qualifiers.get("primary") is True
        
    def test_qualifier_creation(self):
        """Test qualifier creation."""
        q = qualifier("test")
        assert hasattr(q, "name")
        assert q.name == "test"
        

class TestLifecycleDecorators:
    """Test lifecycle decorators."""
    
    def setup_method(self):
        clear_registry()
        
    def test_post_construct_decorator(self):
        """Test @post_construct decorator."""
        @service
        class ServiceWithInit:
            def __init__(self):
                self.initialized = False
                
            @post_construct
            def init(self):
                self.initialized = True
                
        # Note: post_construct is marked but not automatically called
        # The container is responsible for calling it
        instance = ServiceWithInit()
        assert hasattr(instance.init, "_di_post_construct")
        assert instance.init._di_post_construct is True
        
    def test_pre_destroy_decorator(self):
        """Test @pre_destroy decorator."""
        @service
        class ServiceWithCleanup:
            @pre_destroy
            def cleanup(self):
                pass
                
        instance = ServiceWithCleanup()
        assert hasattr(instance.cleanup, "_di_pre_destroy")
        assert instance.cleanup._di_pre_destroy is True
        

class TestConditionalDecorators:
    """Test conditional registration decorators."""
    
    def setup_method(self):
        clear_registry()
        
    def test_profile_decorator(self):
        """Test @profile decorator."""
        @profile("test")
        @service
        class TestOnlyService:
            pass
            
        metadata = get_component_metadata(TestOnlyService)
        assert "test" in metadata.profiles
        assert metadata.should_register({"test"})
        assert not metadata.should_register({"production"})
        
    def test_multiple_profiles(self):
        """Test @profiles decorator."""
        from infrastructure.di.decorators import profiles
        
        @profiles("dev", "test")
        @service
        class DevTestService:
            pass
            
        metadata = get_component_metadata(DevTestService)
        assert "dev" in metadata.profiles
        assert "test" in metadata.profiles
        assert metadata.should_register({"dev"})
        assert metadata.should_register({"test"})
        assert not metadata.should_register({"production"})
        

class TestComponentRegistry:
    """Test component registry functions."""
    
    def setup_method(self):
        clear_registry()
        
    def test_get_registered_components(self):
        """Test getting registered components."""
        @service
        class Service1:
            pass
            
        @repository
        class Repo1:
            pass
            
        components = get_registered_components()
        assert len(components) == 2
        assert Service1 in components
        assert Repo1 in components
        
    def test_filter_by_type(self):
        """Test filtering components by type."""
        @service
        class Service1:
            pass
            
        @service
        class Service2:
            pass
            
        @repository  
        class Repo1:
            pass
            
        services = get_registered_components(
            component_type=ComponentType.DOMAIN_SERVICE
        )
        assert len(services) == 2
        assert Service1 in services
        assert Service2 in services
        assert Repo1 not in services
        
    def test_filter_by_tags(self):
        """Test filtering components by tags."""
        @service
        class DomainService:
            pass
            
        @repository
        class InfraRepo:
            pass
            
        domain_components = get_registered_components(tags={"domain"})
        assert DomainService in domain_components
        assert InfraRepo not in domain_components
        
        infra_components = get_registered_components(tags={"infrastructure"})
        assert InfraRepo in infra_components
        assert DomainService not in infra_components


class TestDependencyExtraction:
    """Test automatic dependency extraction."""
    
    def setup_method(self):
        clear_registry()
        
    def test_extract_dependencies(self):
        """Test extracting constructor dependencies."""
        class Dep1:
            pass
            
        class Dep2:
            pass
            
        @service
        class ServiceWithDeps:
            def __init__(self, dep1: Dep1, dep2: Dep2):
                self.dep1 = dep1
                self.dep2 = dep2
                
        metadata = get_component_metadata(ServiceWithDeps)
        assert Dep1 in metadata.dependencies
        assert Dep2 in metadata.dependencies
        assert len(metadata.dependencies) == 2
        
    def test_optional_dependencies(self):
        """Test optional dependency extraction."""
        @service
        class ServiceWithOptional:
            def __init__(self, required: str, optional: Optional[int] = None):
                self.required = required
                self.optional = optional
                
        metadata = get_component_metadata(ServiceWithOptional)
        # Should still detect the types
        assert str in metadata.dependencies
        assert int in metadata.dependencies  # Optional[int] -> int


class TestInterfaceExtraction:
    """Test automatic interface extraction."""
    
    def setup_method(self):
        clear_registry()
        
    def test_protocol_extraction(self):
        """Test extracting Protocol interfaces."""
        class UserPort(Protocol):
            def get_user(self, id: str): ...
            
        @adapter(UserPort)
        class UserAdapter(UserPort):
            def get_user(self, id: str):
                return {"id": id}
                
        metadata = get_component_metadata(UserAdapter)
        assert UserPort in metadata.interfaces
        
    def test_name_based_extraction(self):
        """Test extracting interfaces by naming convention."""
        class SomeService:
            pass
            
        class SomePort:
            pass
            
        class SomeInterface:
            pass
            
        @service
        class Implementation(SomeService, SomePort, SomeInterface):
            pass
            
        metadata = get_component_metadata(Implementation)
        # Should detect Port and Service suffixes
        assert SomeService in metadata.interfaces
        assert SomePort in metadata.interfaces
        assert SomeInterface in metadata.interfaces