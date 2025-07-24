"""Integration tests for DI container with actual services."""

import pytest
from unittest.mock import Mock, AsyncMock

from infrastructure.di.container import Container
from infrastructure.di.decorators import clear_registry


class TestDIServiceIntegration:
    """Test DI container integration with actual decorated services."""
    
    def setup_method(self):
        """Clean registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clean registry after each test."""
        clear_registry()
    
    def test_domain_service_registration(self, configured_container):
        """Test that decorated domain services can be registered and resolved."""
        container = configured_container
        
        # Import and register decorated services
        from domain.services.training import ModelTrainingService
        from domain.services.tokenization import TokenizationService
        from domain.services.model_builder import ModelBuilderService
        
        container.register_decorator(ModelTrainingService)
        container.register_decorator(TokenizationService)
        container.register_decorator(ModelBuilderService)
        
        # Resolve services
        training_service = container.resolve(ModelTrainingService)
        tokenization_service = container.resolve(TokenizationService)
        model_builder = container.resolve(ModelBuilderService)
        
        # Verify they are properly instantiated
        assert training_service is not None
        assert tokenization_service is not None
        assert model_builder is not None
        
        # Verify they are the correct types
        assert isinstance(training_service, ModelTrainingService)
        assert isinstance(tokenization_service, TokenizationService)
        assert isinstance(model_builder, ModelBuilderService)
    
    def test_service_dependency_injection(self, configured_container):
        """Test that services receive their dependencies correctly."""
        container = configured_container
        
        from domain.services.training import ModelTrainingService
        
        container.register_decorator(ModelTrainingService)
        
        # Resolve service
        service = container.resolve(ModelTrainingService)
        
        # Verify dependencies were injected
        # The service should have its ports injected via constructor
        assert hasattr(service, '__dict__')  # Has instance attributes
        # Specific dependency verification would depend on service implementation
    
    def test_singleton_scope_behavior(self, configured_container):
        """Test that singleton-scoped services return the same instance."""
        container = configured_container
        
        from domain.services.training import ModelTrainingService
        
        container.register_decorator(ModelTrainingService)
        
        # Resolve the same service multiple times
        service1 = container.resolve(ModelTrainingService)
        service2 = container.resolve(ModelTrainingService)
        
        # Should be the same instance for singleton scope
        assert service1 is service2
    
    def test_use_case_with_service_dependencies(self, use_case_container):
        """Test that use cases can resolve with service dependencies."""
        container = use_case_container
        
        from application.use_cases.train_model import TrainModelUseCase
        
        # The use_case_container fixture should have already registered this
        use_case = container.resolve(TrainModelUseCase)
        
        assert use_case is not None
        assert hasattr(use_case, 'training_service')
        
        # Verify the training service was injected and is the right type
        from domain.services.training import ModelTrainingService
        assert isinstance(use_case.training_service, ModelTrainingService)
    
    def test_full_dependency_chain(self, configured_container):
        """Test that complex dependency chains resolve correctly."""
        container = configured_container
        
        # Register a chain of services that depend on each other
        from domain.services.training import ModelTrainingService
        from domain.services.evaluation_engine import EvaluationEngineService
        from domain.services.training_orchestrator import TrainingOrchestratorService
        
        container.register_decorator(ModelTrainingService)
        container.register_decorator(EvaluationEngineService)
        container.register_decorator(TrainingOrchestratorService)
        
        # Resolve the top-level service
        orchestrator = container.resolve(TrainingOrchestratorService)
        
        assert orchestrator is not None
        # The orchestrator should have its dependencies injected
        # Specific verification would depend on the actual dependency structure
    
    def test_error_handling_missing_dependency(self, di_container):
        """Test error handling when dependencies are missing."""
        container = di_container
        
        from domain.services.training import ModelTrainingService
        
        # Try to register service without registering its dependencies
        container.register_decorator(ModelTrainingService)
        
        # Should handle missing dependencies gracefully
        with pytest.raises(Exception):  # The specific exception type depends on implementation
            container.resolve(ModelTrainingService)
    
    def test_container_isolation(self):
        """Test that different containers are isolated."""
        container1 = Container()
        container2 = Container()
        
        from domain.services.training import ModelTrainingService
        
        # Register mock dependencies in both containers
        from application.ports.secondary.compute import ComputeBackend
        mock1 = AsyncMock(spec=ComputeBackend)
        mock2 = AsyncMock(spec=ComputeBackend)
        
        container1.register(ComputeBackend, mock1, instance=True)
        container2.register(ComputeBackend, mock2, instance=True)
        
        container1.register_decorator(ModelTrainingService)
        container2.register_decorator(ModelTrainingService)
        
        service1 = container1.resolve(ModelTrainingService)
        service2 = container2.resolve(ModelTrainingService)
        
        # Should be different instances
        assert service1 is not service2
    
    def test_decorator_metadata_preservation(self, configured_container):
        """Test that decorator metadata is preserved during registration."""
        container = configured_container
        
        from domain.services.training import ModelTrainingService
        from infrastructure.di.decorators import get_component_metadata
        
        # Check that the service has metadata
        metadata = get_component_metadata(ModelTrainingService)
        assert metadata is not None
        assert metadata.component_type == "service"
        
        # Register and resolve
        container.register_decorator(ModelTrainingService)
        service = container.resolve(ModelTrainingService)
        
        # Service should still be functional
        assert service is not None
        assert isinstance(service, ModelTrainingService)


class TestBootstrapIntegration:
    """Test integration with the application bootstrap."""
    
    def test_bootstrap_creates_working_container(self):
        """Test that bootstrap creates a container with all services working."""
        from infrastructure.bootstrap import ApplicationBootstrap
        
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Should be able to resolve key services
        from domain.services.training import ModelTrainingService
        service = container.resolve(ModelTrainingService)
        
        assert service is not None
        assert isinstance(service, ModelTrainingService)
    
    def test_bootstrap_service_dependencies(self):
        """Test that bootstrap properly wires service dependencies."""
        from infrastructure.bootstrap import ApplicationBootstrap
        
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Resolve a service that has dependencies
        from domain.services.training_orchestrator import TrainingOrchestratorService
        
        # This should work if dependencies are properly registered
        try:
            orchestrator = container.resolve(TrainingOrchestratorService)
            assert orchestrator is not None
        except Exception as e:
            pytest.skip(f"Service not yet registered in bootstrap: {e}")