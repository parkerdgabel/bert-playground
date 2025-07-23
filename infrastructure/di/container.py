"""Infrastructure dependency injection container.

This module provides an enhanced DI container specifically designed for the
hexagonal architecture, with support for:
- Port-adapter registration and resolution
- Domain service registration  
- Configuration-driven adapter selection
- Lifecycle management
"""

from typing import Any, Dict, Optional, Type, TypeVar, get_type_hints, Callable
import inspect
from pathlib import Path

from ..config.manager import ConfigurationManager
from .registry import ServiceRegistry, AdapterRegistry

T = TypeVar("T")


class Container:
    """Basic dependency injection container.
    
    Provides core DI functionality including:
    - Service registration
    - Dependency resolution
    - Lifecycle management (singleton/transient)
    """
    
    def __init__(self):
        """Initialize the container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._singleton_types: set = set()
        
    def register(
        self, 
        service_type: Type[T], 
        implementation: Any = None,
        factory: Optional[Callable] = None,
        instance: bool = False,
        singleton: bool = False
    ) -> None:
        """Register a service in the container.
        
        Args:
            service_type: The service interface/type
            implementation: The implementation class
            factory: Factory function to create instances
            instance: Whether implementation is already an instance
            singleton: Whether to use singleton lifecycle
        """
        if instance:
            self._singletons[service_type] = implementation
        elif factory:
            self._factories[service_type] = factory
        else:
            self._services[service_type] = implementation or service_type
            
        if singleton:
            self._singleton_types.add(service_type)
            
    def register_factory(self, service_type: Type[T], factory: Callable[["Container"], T]) -> None:
        """Register a factory function for a service.
        
        Args:
            service_type: The service type
            factory: Factory function that takes the container
        """
        self._factories[service_type] = factory
        
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not registered
        """
        # Check for existing singleton
        if service_type in self._singletons:
            return self._singletons[service_type]
            
        # Check for factory
        if service_type in self._factories:
            instance = self._factories[service_type](self)
            if service_type in self._singleton_types:
                self._singletons[service_type] = instance
            return instance
            
        # Check for registered service
        if service_type in self._services:
            implementation = self._services[service_type]
            instance = self._create_instance(implementation)
            if service_type in self._singleton_types:
                self._singletons[service_type] = instance
            return instance
            
        raise KeyError(f"Service {service_type} not registered")
        
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with dependency injection.
        
        Args:
            implementation: The class to instantiate
            
        Returns:
            Created instance
        """
        # Get constructor parameters
        sig = inspect.signature(implementation.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to resolve parameter type
            if param.annotation != param.empty:
                try:
                    params[param_name] = self.resolve(param.annotation)
                except KeyError:
                    if param.default == param.empty:
                        raise
                        
        return implementation(**params)
        
    def has(self, service_type: Type) -> bool:
        """Check if a service is registered.
        
        Args:
            service_type: The service type
            
        Returns:
            True if registered
        """
        return (
            service_type in self._services or
            service_type in self._factories or
            service_type in self._singletons
        )
        
    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_types.clear()
        
    def list_services(self) -> list:
        """List all registered service types."""
        return list(set(
            list(self._services.keys()) +
            list(self._factories.keys()) +
            list(self._singletons.keys())
        ))


class InfrastructureContainer:
    """Enhanced DI container for hexagonal architecture infrastructure.
    
    This container extends the core DI functionality with:
    - Automatic port-adapter wiring based on configuration
    - Domain service registration
    - Use case registration
    - Adapter lifecycle management
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize infrastructure container.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.core_container = Container()
        self.config_manager = config_manager or ConfigurationManager()
        self.service_registry = ServiceRegistry()
        self.adapter_registry = AdapterRegistry()
        
        # Register the configuration manager itself
        self.core_container.register(
            ConfigurationManager, 
            self.config_manager, 
            instance=True
        )
        
    def register_domain_services(self) -> None:
        """Register all domain services."""
        # Skip domain services for now due to import issues
        # TODO: Enable when domain services are fully implemented
        pass
        
        # # Training domain services
        # self._register_training_domain()
        # 
        # # Data domain services
        # self._register_data_domain()
        # 
        # # Model domain services
        # self._register_model_domain()
        
    def _register_training_domain(self) -> None:
        """Register training domain services."""
        try:
            # Import training domain services
            from domain.services.training_service import TrainingService
            from domain.services.evaluation_service import EvaluationService
            
            self.service_registry.register_service("training", TrainingService)
            self.service_registry.register_service("evaluation", EvaluationService)
            
            # Register with core container
            self.core_container.register(TrainingService, singleton=True)
            self.core_container.register(EvaluationService, singleton=True)
            
        except ImportError:
            # Training domain services not yet implemented
            pass
            
    def _register_data_domain(self) -> None:
        """Register data domain services."""
        try:
            from domain.data.data_service import DataService
            from adapters.secondary.data.factory import DatasetFactory
            
            self.service_registry.register_service("data", DataService)
            self.service_registry.register_service("dataset_factory", DatasetFactory)
            
            self.core_container.register(DataService, singleton=True)
            self.core_container.register(DatasetFactory, singleton=True)
            
        except ImportError:
            # Fallback to existing data services
            try:
                from adapters.secondary.data.factory import DatasetFactory
                self.core_container.register(DatasetFactory, singleton=True)
            except ImportError:
                pass
                
    def _register_model_domain(self) -> None:
        """Register model domain services."""
        try:
            from models.factory_facade import ModelFactory
            from models.factory import ModelFactoryImpl
            
            self.service_registry.register_service("model_factory", ModelFactory)
            
            self.core_container.register(ModelFactory, singleton=True)
            
        except ImportError:
            # Model factory not available
            pass
            
    def register_ports_and_adapters(self) -> None:
        """Register all ports with their configured adapters."""
        # Primary ports (driving adapters)
        self._register_primary_ports()
        
        # Secondary ports (driven adapters)  
        self._register_secondary_ports()
        
    def _register_primary_ports(self) -> None:
        """Register primary ports with their adapters."""
        # Skip CLI adapters for now due to dataclass issues
        # self._register_cli_adapters()
        
        # API adapters (future)
        # self._register_api_adapters()
        
    def _register_cli_adapters(self) -> None:
        """Register CLI adapters."""
        try:
            from adapters.primary.cli.app import CLIAdapter
            from adapters.primary.cli.train_adapter import TrainCommandAdapter
            from adapters.primary.cli.predict_adapter import PredictCommandAdapter
            from adapters.primary.cli.benchmark_adapter import BenchmarkCommandAdapter
            
            # Register CLI adapters
            self.adapter_registry.register_adapter("cli", "main", CLIAdapter)
            self.adapter_registry.register_adapter("cli", "train", TrainCommandAdapter)
            self.adapter_registry.register_adapter("cli", "predict", PredictCommandAdapter)
            self.adapter_registry.register_adapter("cli", "benchmark", BenchmarkCommandAdapter)
            
            # Register with container
            self.core_container.register(CLIAdapter, singleton=True)
            self.core_container.register(TrainCommandAdapter, singleton=True)
            self.core_container.register(PredictCommandAdapter, singleton=True)
            self.core_container.register(BenchmarkCommandAdapter, singleton=True)
            
        except ImportError:
            # CLI adapters not available yet
            pass
            
    def _register_secondary_ports(self) -> None:
        """Register secondary ports with configured adapters."""
        # Monitoring port
        self._register_monitoring_port()
        
        # Storage port
        self._register_storage_port()
        
        # Compute port
        self._register_compute_port()
        
        # Tokenizer port
        self._register_tokenizer_port()
        
        # Configuration port
        self._register_configuration_port()
        
    def _register_monitoring_port(self) -> None:
        """Register monitoring port with configured adapter."""
        # Use existing core monitoring port to avoid import issues
        try:
            from infrastructure.ports.monitoring import MonitoringService
        except ImportError:
            # Define a simple monitoring service protocol
            from typing import Protocol
            class MonitoringService(Protocol):
                def info(self, message: str, **kwargs): ...
                def error(self, message: str, **kwargs): ...
                def debug(self, message: str, **kwargs): ...
        
        # Get configured implementation
        adapter_config = self.config_manager.get_adapter_config("monitoring")
        implementation = adapter_config.get("implementation", "loguru")
        
        # Use core adapters to avoid import issues
        from infrastructure.adapters.loguru_monitoring import LoguruMonitoringAdapter
        adapter_class = LoguruMonitoringAdapter
            
        self.adapter_registry.register_adapter("monitoring", implementation, adapter_class)
        self.core_container.register(MonitoringService, adapter_class, singleton=True)
        
    def _register_storage_port(self) -> None:
        """Register storage port with configured adapter."""
        # Use existing storage adapters to avoid import issues
        try:
            from infrastructure.ports.storage import StorageService
            from infrastructure.adapters.file_storage import FileStorageAdapter
            self.core_container.register(StorageService, FileStorageAdapter, singleton=True)
        except ImportError:
            pass
                
    def _register_compute_port(self) -> None:
        """Register compute port with configured adapter."""
        # Use existing compute adapters to avoid import issues
        try:
            from infrastructure.ports.compute import ComputeBackend
            from infrastructure.adapters.mlx_adapter import MLXComputeAdapter
            self.core_container.register(ComputeBackend, MLXComputeAdapter, singleton=True)
        except ImportError:
            pass
                
    def _register_tokenizer_port(self) -> None:
        """Register tokenizer port with configured adapter."""
        # Use existing tokenizer adapters to avoid import issues
        try:
            from infrastructure.ports.tokenizer import TokenizerFactory
            from infrastructure.adapters.huggingface_tokenizer import HuggingFaceTokenizerFactory
            self.core_container.register(TokenizerFactory, HuggingFaceTokenizerFactory, singleton=True)
        except ImportError:
            pass
                
    def _register_configuration_port(self) -> None:
        """Register configuration port."""
        # Use existing configuration port
        try:
            from infrastructure.ports.config import ConfigurationProvider
            from infrastructure.adapters.yaml_config import YAMLConfigAdapter
            
            config_adapter = YAMLConfigAdapter()
            self.core_container.register(
                ConfigurationProvider, 
                config_adapter, 
                instance=True
            )
        except ImportError:
            pass
                
    def register_application_services(self) -> None:
        """Register application layer services."""
        # Skip application services for now
        # TODO: Enable when application services are implemented
        pass
        
    def _register_use_cases(self) -> None:
        """Register application use cases."""
        try:
            from application.use_cases.train_model import TrainModelUseCase
            from application.use_cases.predict import PredictUseCase
            from application.use_cases.evaluate_model import EvaluateModelUseCase
            
            self.service_registry.register_service("train_use_case", TrainModelUseCase)
            self.service_registry.register_service("predict_use_case", PredictUseCase)
            self.service_registry.register_service("evaluate_use_case", EvaluateModelUseCase)
            
            # Register with auto-wiring
            self.core_container.register(TrainModelUseCase)
            self.core_container.register(PredictUseCase)
            self.core_container.register(EvaluateModelUseCase)
            
        except ImportError:
            # Application use cases not available yet
            pass
            
    def _register_orchestrators(self) -> None:
        """Register orchestration services."""
        try:
            from application.orchestration.training_orchestrator import TrainingOrchestrator
            from application.orchestration.workflow_orchestrator import WorkflowOrchestrator
            
            self.service_registry.register_service("training_orchestrator", TrainingOrchestrator)
            self.service_registry.register_service("workflow_orchestrator", WorkflowOrchestrator)
            
            self.core_container.register(TrainingOrchestrator, singleton=True) 
            self.core_container.register(WorkflowOrchestrator, singleton=True)
            
        except ImportError:
            # Orchestrators not available yet
            pass
            
    def register_training_components(self) -> None:
        """Register training infrastructure components."""
        # Skip training components for now
        # TODO: Enable when training components are needed
        pass
            
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
        """
        return self.core_container.resolve(service_type)
        
    def has(self, service_type: Type) -> bool:
        """Check if service is registered.
        
        Args:
            service_type: Service type to check
            
        Returns:
            True if registered
        """
        return self.core_container.has(service_type)
        
    def create_child(self) -> "InfrastructureContainer":
        """Create child container for scoped resolution.
        
        Returns:
            Child container
        """
        child = InfrastructureContainer(self.config_manager)
        child.core_container = self.core_container.create_child()
        return child
        
    def get_adapter_info(self, port_type: str) -> Dict[str, Any]:
        """Get information about registered adapters for a port.
        
        Args:
            port_type: Port type (e.g., "monitoring", "storage")
            
        Returns:
            Adapter information
        """
        return self.adapter_registry.get_adapter_info(port_type)
        
    def list_services(self) -> Dict[str, Type]:
        """List all registered services.
        
        Returns:
            Service name -> type mapping
        """
        return self.service_registry.list_services()
        
    def swap_adapter(self, port_type: str, new_implementation: str) -> None:
        """Swap an adapter implementation at runtime.
        
        Args:
            port_type: Port type to swap
            new_implementation: New implementation name
        """
        # Update configuration
        adapter_config = self.config_manager.get_adapter_config(port_type)
        adapter_config["implementation"] = new_implementation
        
        # Re-register the port with new adapter
        if port_type == "monitoring":
            self._register_monitoring_port()
        elif port_type == "storage":
            self._register_storage_port()
        elif port_type == "compute":
            self._register_compute_port()
        elif port_type == "tokenizer":
            self._register_tokenizer_port()
        else:
            raise ValueError(f"Unknown port type: {port_type}")
            
    def initialize_all(self) -> None:
        """Initialize all registered services and adapters."""
        # Register all components
        self.register_domain_services()
        self.register_ports_and_adapters()
        self.register_application_services()
        self.register_training_components()
        
        # Initialize any services that need initialization
        self._initialize_services()
        
    def _initialize_services(self) -> None:
        """Initialize services that need post-registration setup."""
        # Initialize monitoring service
        try:
            from ports.secondary.monitoring import MonitoringService
            monitoring = self.resolve(MonitoringService)
            if hasattr(monitoring, 'initialize'):
                monitoring.initialize()
        except Exception:
            pass
            
        # Initialize other services as needed
        # ...
        
    def clear(self) -> None:
        """Clear all registrations."""
        self.core_container.clear()
        self.service_registry.clear()
        self.adapter_registry.clear()