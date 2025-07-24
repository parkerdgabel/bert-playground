"""Tests for component scanner and auto-discovery."""

import pytest
from pathlib import Path
import tempfile
import os

from infrastructure.di.scanner import (
    ComponentScanner, DependencyGraph, auto_discover_and_register
)
from infrastructure.di.container import Container
from infrastructure.di.decorators import (
    service, repository, adapter, use_case,
    ComponentType, get_component_metadata,
    clear_registry
)


class TestDependencyGraph:
    """Test dependency graph for circular detection."""
    
    def test_no_cycle(self):
        """Test graph with no cycles."""
        graph = DependencyGraph()
        
        # A -> B -> C
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        assert not graph.has_cycle()
        
    def test_simple_cycle(self):
        """Test graph with simple cycle."""
        graph = DependencyGraph()
        
        # A -> B -> C -> A
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        assert graph.has_cycle()
        
    def test_self_cycle(self):
        """Test self-referencing cycle."""
        graph = DependencyGraph()
        
        # A -> A
        graph.add_edge("A", "A")
        
        assert graph.has_cycle()
        
    def test_topological_sort(self):
        """Test topological sorting."""
        graph = DependencyGraph()
        
        # C depends on B, B depends on A
        graph.add_edge("C", "B")
        graph.add_edge("B", "A")
        
        order = graph.topological_sort()
        
        # A should come before B, B before C
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")
        

class TestComponentScanner:
    """Test component scanner functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_registry()
        self.container = Container()
        self.scanner = ComponentScanner(self.container)
        
    def test_scan_components(self):
        """Test scanning discovers decorated components."""
        # Create test components
        @service
        class TestService:
            pass
            
        @repository
        class TestRepo:
            pass
            
        # Manually add to discovered (simulating scan)
        self.scanner.discovered_components = [TestService, TestRepo]
        
        # Register all
        self.scanner.register_all(validate=False)
        
        # Check registration
        assert self.container.has(TestService)
        assert self.container.has(TestRepo)
        
    def test_profile_filtering(self):
        """Test profile-based component filtering."""
        from infrastructure.di.decorators import profile
        
        @service
        @profile("test")
        class TestOnlyService:
            pass
            
        @service
        @profile("prod")
        class ProdOnlyService:
            pass
            
        @service
        class AlwaysService:
            pass
            
        # Set active profile
        self.scanner.set_active_profiles({"test"})
        
        # Check filtering
        metadata1 = get_component_metadata(TestOnlyService)
        metadata2 = get_component_metadata(ProdOnlyService)
        metadata3 = get_component_metadata(AlwaysService)
        
        assert metadata1.should_register({"test"})
        assert not metadata2.should_register({"test"})
        assert metadata3.should_register({"test"})
        
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        @service
        class ServiceA:
            def __init__(self, b: "ServiceB"):
                self.b = b
                
        @service
        class ServiceB:
            def __init__(self, a: ServiceA):
                self.a = a
                
        self.scanner.discovered_components = [ServiceA, ServiceB]
        
        # Build graph
        self.scanner._build_dependency_graph()
        
        # Should detect cycle
        with pytest.raises(RuntimeError, match="Circular dependency"):
            self.scanner.register_all(validate=True)
            
    def test_dependency_order_registration(self):
        """Test components are registered in dependency order."""
        registration_order = []
        
        # Mock register to track order
        original_register = self.container.register_decorator
        
        def track_register(cls):
            registration_order.append(cls.__name__)
            return original_register(cls)
            
        self.container.register_decorator = track_register
        
        # Create dependency chain
        @service
        class BaseService:
            pass
            
        @service
        class MiddleService:
            def __init__(self, base: BaseService):
                self.base = base
                
        @service
        class TopService:
            def __init__(self, middle: MiddleService):
                self.middle = middle
                
        self.scanner.discovered_components = [TopService, MiddleService, BaseService]
        self.scanner.register_all(validate=True)
        
        # Check registration order
        assert registration_order.index("BaseService") < registration_order.index("MiddleService")
        assert registration_order.index("MiddleService") < registration_order.index("TopService")
        
    def test_architecture_validation(self):
        """Test hexagonal architecture validation."""
        # Create invalid dependency (domain -> infrastructure)
        @service  # domain layer
        class DomainService:
            def __init__(self, repo: "InfraRepository"):
                self.repo = repo
                
        @repository  # infrastructure layer
        class InfraRepository:
            pass
            
        self.scanner.discovered_components = [DomainService, InfraRepository]
        
        errors = self.scanner.validate_architecture()
        assert len(errors) == 1
        assert "violates hexagonal architecture" in errors[0]
        
    def test_valid_architecture(self):
        """Test valid architectural dependencies."""
        # Valid: infrastructure -> domain
        @service
        class DomainService:
            pass
            
        @repository
        class InfraRepository:
            def __init__(self, service: DomainService):
                self.service = service
                
        # Valid: application -> domain
        @use_case
        class AppUseCase:
            def __init__(self, service: DomainService):
                self.service = service
                
        self.scanner.discovered_components = [
            DomainService, InfraRepository, AppUseCase
        ]
        
        errors = self.scanner.validate_architecture()
        assert len(errors) == 0
        
    def test_generate_report(self):
        """Test report generation."""
        @service
        class Service1:
            pass
            
        @service
        class Service2:
            pass
            
        @repository
        class Repo1:
            pass
            
        @use_case
        class UseCase1:
            pass
            
        self.scanner.discovered_components = [
            Service1, Service2, Repo1, UseCase1
        ]
        
        report = self.scanner.generate_report()
        
        # Check report content
        assert "Component Discovery Report" in report
        assert "DOMAIN_SERVICE (2)" in report
        assert "REPOSITORY (1)" in report
        assert "USE_CASE (1)" in report
        assert "Total components: 4" in report
        

class TestAutoDiscovery:
    """Test auto-discovery convenience function."""
    
    def setup_method(self):
        clear_registry()
        self.container = Container()
        
    def test_auto_discover_and_register(self):
        """Test the convenience function."""
        # Create components
        @service
        class AutoService:
            pass
            
        @repository  
        class AutoRepo:
            pass
            
        # Mock package scanning
        # In real usage, this would scan actual packages
        scanner = ComponentScanner(self.container)
        scanner.discovered_components = [AutoService, AutoRepo]
        
        # Manually trigger registration
        scanner.register_all()
        
        # Verify registration
        assert self.container.has(AutoService)
        assert self.container.has(AutoRepo)
        
    def test_with_profiles(self):
        """Test auto-discovery with profiles."""
        from infrastructure.di.decorators import profile
        
        @service
        @profile("dev")
        class DevService:
            pass
            
        @service
        @profile("prod")
        class ProdService:
            pass
            
        # Create scanner with dev profile
        scanner = ComponentScanner(self.container)
        scanner.set_active_profiles({"dev"})
        
        # Add components
        scanner.discovered_components = [DevService, ProdService]
        
        # Only DevService should be registered
        registered = []
        original = self.container.register_decorator
        
        def track(cls):
            registered.append(cls)
            return original(cls)
            
        self.container.register_decorator = track
        scanner.register_all()
        
        assert DevService in registered
        assert ProdService not in registered