"""Infrastructure dependency injection container.

This module provides an enhanced DI container specifically designed for the
hexagonal architecture, with support for:
- Port-adapter registration and resolution
- Domain service registration  
- Configuration-driven adapter selection
- Lifecycle management
"""

from typing import (
    Any, Dict, Optional, Type, TypeVar, get_type_hints, Callable,
    Union, List, Set, get_origin, get_args
)
import inspect
from pathlib import Path
from collections import defaultdict

from ..config.manager import ConfigurationManager
from .registry import ServiceRegistry, AdapterRegistry
from .decorators import (
    ComponentMetadata, ComponentType, Scope, 
    get_component_metadata, qualifier, value
)

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
        
        # Enhanced metadata tracking
        self._metadata: Dict[Type, ComponentMetadata] = {}
        self._qualifiers: Dict[str, Dict[Type, Type]] = defaultdict(dict)
        self._primary_implementations: Dict[Type, Type] = {}
        self._config_values: Dict[str, Any] = {}
        
    def register(
        self, 
        service_type: Type[T], 
        implementation: Any = None,
        factory: Optional[Callable] = None,
        instance: bool = False,
        singleton: bool = False,
        metadata: Optional[ComponentMetadata] = None
    ) -> None:
        """Register a service in the container.
        
        Args:
            service_type: The service interface/type
            implementation: The implementation class
            factory: Factory function to create instances
            instance: Whether implementation is already an instance
            singleton: Whether to use singleton lifecycle
            metadata: Optional component metadata
        """
        # Store metadata if provided
        if metadata:
            self._metadata[service_type] = metadata
            
            # Handle qualifiers
            for qualifier_name, qualifier_value in metadata.qualifiers.items():
                if qualifier_name == "primary" and qualifier_value:
                    self._primary_implementations[service_type] = implementation or service_type
                elif isinstance(qualifier_name, str):
                    self._qualifiers[qualifier_name][service_type] = implementation or service_type
        
        # Auto-detect metadata from decorated classes
        elif implementation and hasattr(implementation, "_di_metadata"):
            comp_metadata = get_component_metadata(implementation)
            if comp_metadata:
                self._metadata[service_type] = comp_metadata
                
                # Override singleton based on scope
                if comp_metadata.scope == Scope.SINGLETON:
                    singleton = True
                elif comp_metadata.scope == Scope.TRANSIENT:
                    singleton = False
                    
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
        
    def resolve(self, service_type: Type[T], qualifier_name: Optional[str] = None) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The service type to resolve
            qualifier_name: Optional qualifier name for specific implementation
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not registered
        """
        # Handle qualified resolution
        if qualifier_name:
            qualified_types = self._qualifiers.get(qualifier_name, {})
            if service_type in qualified_types:
                return self.resolve(qualified_types[service_type])
                
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
            
            # Call post-construct method if defined
            metadata = self._metadata.get(service_type)
            if metadata and metadata.init_method:
                init_method = getattr(instance, metadata.init_method, None)
                if init_method and callable(init_method):
                    init_method()
                    
            if service_type in self._singleton_types:
                self._singletons[service_type] = instance
            return instance
            
        # Try to resolve primary implementation
        if service_type in self._primary_implementations:
            return self.resolve(self._primary_implementations[service_type])
            
        raise KeyError(f"Service {service_type} not registered")
        
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with enhanced dependency injection.
        
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
                param_type = param.annotation
                origin = get_origin(param_type)
                
                # Handle Optional types
                if origin is Union:
                    args = get_args(param_type)
                    # Check if it's Optional (Union with None)
                    non_none_types = [t for t in args if t is not type(None)]
                    if len(args) == 2 and type(None) in args and non_none_types:
                        # It's Optional[T]
                        try:
                            params[param_name] = self.resolve(non_none_types[0])
                        except KeyError:
                            # Optional dependency not found, use None
                            params[param_name] = None
                        continue
                        
                # Handle List types (multiple implementations)
                elif origin is list or origin is List:
                    args = get_args(param_type)
                    if args:
                        item_type = args[0]
                        # Find all implementations of the type
                        implementations = []
                        for service_type, impl in self._services.items():
                            if self._is_assignable(impl, item_type):
                                implementations.append(self.resolve(service_type))
                        params[param_name] = implementations
                        continue
                        
                # Handle Set types (unique implementations)
                elif origin is set or origin is Set:
                    args = get_args(param_type)
                    if args:
                        item_type = args[0]
                        # Find all implementations of the type
                        implementations = set()
                        for service_type, impl in self._services.items():
                            if self._is_assignable(impl, item_type):
                                implementations.add(self.resolve(service_type))
                        params[param_name] = implementations
                        continue
                
                # Handle Annotated types (qualifiers and values)
                elif hasattr(param_type, "__metadata__"):
                    # Extract base type and metadata
                    base_type = get_args(param_type)[0]
                    metadata = param_type.__metadata__
                    
                    for meta in metadata:
                        # Handle qualifier
                        if hasattr(meta, "name") and hasattr(meta, "__class__") and meta.__class__.__name__ == "Qualifier":
                            params[param_name] = self.resolve(base_type, qualifier_name=meta.name)
                            break
                        # Handle value injection
                        elif hasattr(meta, "key") and hasattr(meta, "__class__") and meta.__class__.__name__ == "Value":
                            params[param_name] = self._config_values.get(meta.key, meta.default)
                            break
                    else:
                        # No special metadata, resolve normally
                        params[param_name] = self.resolve(base_type)
                    continue
                    
                # Normal type resolution
                try:
                    params[param_name] = self.resolve(param_type)
                except KeyError:
                    if param.default == param.empty:
                        raise
                        
        return implementation(**params)
    
    def _is_assignable(self, implementation: Type, interface: Type) -> bool:
        """Check if implementation is assignable to interface."""
        try:
            return issubclass(implementation, interface)
        except TypeError:
            return False
        
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
    
    def inject_config(self, key: str, value: Any) -> None:
        """Inject a configuration value for @value decorators.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config_values[key] = value
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get an injected configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self._config_values.get(key, default)
    
    def register_decorator(self, cls: Type[T]) -> Type[T]:
        """Register a decorated component automatically.
        
        Args:
            cls: The decorated class
            
        Returns:
            The class (for decorator chaining)
        """
        metadata = get_component_metadata(cls)
        if not metadata:
            return cls
            
        # Determine scope
        singleton = metadata.scope == Scope.SINGLETON
        
        # Register for each interface
        if metadata.interfaces:
            for interface in metadata.interfaces:
                self.register(interface, cls, singleton=singleton, metadata=metadata)
        else:
            # Register as self
            self.register(cls, cls, singleton=singleton, metadata=metadata)
            
        # Handle adapters
        if metadata.port_type:
            self.register(metadata.port_type, cls, singleton=singleton, metadata=metadata)
            
        return cls
    
    def auto_wire(self, implementation: Type[T]) -> T:
        """Create an instance with automatic dependency injection.
        
        This is an alias for _create_instance for backward compatibility.
        
        Args:
            implementation: The class to instantiate
            
        Returns:
            Created instance
        """
        return self._create_instance(implementation)
    
    def create_child(self) -> "Container":
        """Create a child container for scoped resolution.
        
        Returns:
            Child container
        """
        child = Container()
        # Copy services but not singletons
        child._services = self._services.copy()
        child._factories = self._factories.copy()
        child._singleton_types = self._singleton_types.copy()
        child._metadata = self._metadata.copy()
        child._qualifiers = self._qualifiers.copy()
        child._primary_implementations = self._primary_implementations.copy()
        child._config_values = self._config_values.copy()
        return child


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
        
        # Checkpointing port
        self._register_checkpointing_port()
        
        # Optimization port
        self._register_optimization_port()
        
        # Plugins port
        self._register_plugins_port()
        
        # Neural port
        self._register_neural_port()
        
        # Data port
        self._register_data_port()
        
        # Metrics port
        self._register_metrics_port()
        
    def _register_monitoring_port(self) -> None:
        """Register monitoring port with configured adapter."""
        # Use existing core monitoring port to avoid import issues
        try:
            from ports.secondary.monitoring import MonitoringService
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
        from adapters.secondary.monitoring.loguru import LoguruMonitoringAdapter
        adapter_class = LoguruMonitoringAdapter
            
        self.adapter_registry.register_adapter("monitoring", implementation, adapter_class)
        self.core_container.register(MonitoringService, adapter_class, singleton=True)
        
    def _register_storage_port(self) -> None:
        """Register storage port with configured adapter."""
        # Use existing storage adapters to avoid import issues
        try:
            from ports.secondary.storage import StorageService
            from adapters.secondary.storage.file_storage import FileStorageAdapter
            self.core_container.register(StorageService, FileStorageAdapter, singleton=True)
        except ImportError:
            pass
                
    def _register_compute_port(self) -> None:
        """Register compute port with configured adapter."""
        # Register MLXComputeAdapter for low-level tensor operations only
        try:
            from ports.secondary.compute import ComputeBackend
            from adapters.secondary.compute.mlx.compute_adapter import MLXComputeAdapter
            
            # Register the compute adapter (tensor operations only)
            self.core_container.register(ComputeBackend, MLXComputeAdapter, singleton=True)
            
        except ImportError:
            pass
                
    def _register_tokenizer_port(self) -> None:
        """Register tokenizer port with configured adapter."""
        # Use existing tokenizer adapters to avoid import issues
        try:
            from ports.secondary.tokenizer import TokenizerPort
            from adapters.secondary.tokenizer.huggingface.tokenizer_adapter import HuggingFaceTokenizerAdapter
            self.core_container.register(TokenizerPort, HuggingFaceTokenizerAdapter, singleton=True)
        except ImportError:
            pass
                
    def _register_configuration_port(self) -> None:
        """Register configuration port."""
        # Use existing configuration port
        try:
            from ports.secondary.configuration import ConfigurationProvider
            from adapters.secondary.configuration.yaml_adapter import YamlConfigurationAdapter
            
            config_adapter = YamlConfigurationAdapter()
            self.core_container.register(
                ConfigurationProvider, 
                config_adapter, 
                instance=True
            )
        except ImportError:
            pass
    
    def _register_checkpointing_port(self) -> None:
        """Register checkpointing port with configured adapter."""
        try:
            from ports.secondary.checkpointing import CheckpointManager
            from adapters.secondary.checkpointing.filesystem_adapter import FilesystemCheckpointManager
            
            # Get checkpoint directory from config
            checkpoint_dir = self.config_manager.get("checkpoint.dir", "checkpoints")
            adapter = FilesystemCheckpointManager(checkpoint_dir)
            
            self.core_container.register(
                CheckpointManager,
                adapter,
                instance=True
            )
        except ImportError:
            pass
    
    def _register_optimization_port(self) -> None:
        """Register optimization port with configured adapter."""
        try:
            from ports.secondary.optimization import Optimizer, OptimizerConfig
            from adapters.secondary.optimization.mlx_optimizer import MLXOptimizerAdapter
            
            # Create factory for optimizer
            def optimizer_factory(container):
                config = OptimizerConfig(
                    learning_rate=container.config_manager.get("training.learning_rate", 1e-3),
                    weight_decay=container.config_manager.get("training.weight_decay", 0.0),
                )
                return MLXOptimizerAdapter(config)
            
            self.core_container.register_factory(Optimizer, optimizer_factory)
        except ImportError:
            pass
    
    def _register_plugins_port(self) -> None:
        """Register plugins port with configured adapter."""
        try:
            from adapters.secondary.plugins.loader_adapter import PluginLoaderAdapter
            from adapters.secondary.plugins.registry_adapter import PluginRegistryAdapter
            
            # Register plugin loader and registry
            self.core_container.register(PluginLoaderAdapter, singleton=True)
            self.core_container.register(PluginRegistryAdapter, singleton=True)
        except ImportError:
            pass
    
    def _register_neural_port(self) -> None:
        """Register neural port with configured adapter."""
        try:
            from ports.secondary.neural import NeuralBackend
            from adapters.secondary.neural.mlx_backend import MLXNeuralBackend
            from adapters.secondary.neural.mlx_adapter import MLXNeuralAdapter
            from ports.secondary.compute import ComputeBackend
            
            # Register the neural backend (low-level neural operations)
            self.core_container.register(NeuralBackend, MLXNeuralBackend, singleton=True)
            
            # Register the neural adapter with dependencies using factory pattern as singleton
            def neural_adapter_factory(container):
                neural_backend = container.resolve(NeuralBackend)
                compute_backend = container.resolve(ComputeBackend)
                return MLXNeuralAdapter(neural_backend, compute_backend)
            
            self.core_container.register_factory(MLXNeuralAdapter, neural_adapter_factory)
            self.core_container._singleton_types.add(MLXNeuralAdapter)
            
            # Register adapter info for monitoring
            self.adapter_registry.register_adapter("neural", "mlx", MLXNeuralAdapter)
            
        except ImportError:
            pass
    
    def _register_data_port(self) -> None:
        """Register data port with configured adapter."""
        try:
            from ports.secondary.data import DataLoaderPort
            from adapters.secondary.data.mlx.data_loader import MLXDataLoader
            
            self.core_container.register(DataLoaderPort, MLXDataLoader, singleton=True)
        except ImportError:
            pass
    
    def _register_metrics_port(self) -> None:
        """Register metrics port with configured adapter."""
        try:
            from ports.secondary.metrics import MetricsCalculator
            from adapters.secondary.metrics.mlx.metrics_calculator import MLXMetricsCalculator
            
            self.core_container.register(MetricsCalculator, MLXMetricsCalculator, singleton=True)
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