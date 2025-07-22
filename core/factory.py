"""Factory for creating and managing adapters with dependency injection."""

from pathlib import Path
from typing import Any, TypeVar, cast

from core.adapters import (
    ConfigRegistryImpl,
    FileStorageAdapter,
    LoguruMonitoringAdapter,
    MLflowExperimentTracker,
    MLXComputeAdapter,
    MLXNeuralOpsAdapter,
    ModelFileStorageAdapter,
    YAMLConfigAdapter,
)
from core.ports import (
    ComputeBackend,
    ConfigurationProvider,
    MonitoringService,
    StorageService,
)
from core.ports.compute import NeuralOps
from core.ports.config import ConfigRegistry
from core.ports.monitoring import ExperimentTracker
from core.ports.storage import ModelStorageService

T = TypeVar("T")


class AdapterFactory:
    """Factory for creating and managing adapter instances with dependency injection."""

    def __init__(self):
        """Initialize the factory with default adapters."""
        self._adapters: dict[type, Any] = {}
        self._config: dict[str, Any] = {}
        
        # Register default adapters
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default adapter implementations."""
        # Compute adapters
        self.register(ComputeBackend, MLXComputeAdapter())
        self.register(NeuralOps, MLXNeuralOpsAdapter())
        
        # Storage adapters
        self.register(StorageService, FileStorageAdapter())
        self.register(ModelStorageService, ModelFileStorageAdapter())
        
        # Config adapters
        self.register(ConfigurationProvider, YAMLConfigAdapter())
        self.register(ConfigRegistry, ConfigRegistryImpl())
        
        # Monitoring adapters
        self.register(MonitoringService, LoguruMonitoringAdapter())
        self.register(ExperimentTracker, MLflowExperimentTracker())

    def register(self, port_type: type[T], adapter: T) -> None:
        """Register an adapter for a port type.
        
        Args:
            port_type: The port interface type
            adapter: The adapter implementation
        """
        self._adapters[port_type] = adapter

    def get(self, port_type: type[T]) -> T:
        """Get an adapter for a port type.
        
        Args:
            port_type: The port interface type
            
        Returns:
            The registered adapter
            
        Raises:
            KeyError: If no adapter is registered for the port type
        """
        if port_type not in self._adapters:
            raise KeyError(f"No adapter registered for {port_type.__name__}")
        
        return cast(T, self._adapters[port_type])

    def configure(self, config: dict[str, Any]) -> None:
        """Configure adapters with settings.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Configure specific adapters based on config
        if "storage" in config:
            storage_config = config["storage"]
            if "base_path" in storage_config:
                base_path = Path(storage_config["base_path"])
                self.register(StorageService, FileStorageAdapter(base_path))
                self.register(ModelStorageService, ModelFileStorageAdapter(base_path))
        
        if "compute" in config:
            compute_config = config["compute"]
            # Could configure compute backend settings here
        
        if "monitoring" in config:
            monitoring_config = config["monitoring"]
            # Could configure monitoring settings here

    def create_context(self) -> "AdapterContext":
        """Create a context with all registered adapters.
        
        Returns:
            AdapterContext with easy access to all adapters
        """
        return AdapterContext(self)


class AdapterContext:
    """Context providing easy access to all registered adapters."""

    def __init__(self, factory: AdapterFactory):
        """Initialize with factory."""
        self._factory = factory

    @property
    def compute(self) -> ComputeBackend:
        """Get compute backend."""
        return self._factory.get(ComputeBackend)

    @property
    def neural_ops(self) -> NeuralOps:
        """Get neural operations."""
        return self._factory.get(NeuralOps)

    @property
    def storage(self) -> StorageService:
        """Get storage service."""
        return self._factory.get(StorageService)

    @property
    def model_storage(self) -> ModelStorageService:
        """Get model storage service."""
        return self._factory.get(ModelStorageService)

    @property
    def config(self) -> ConfigurationProvider:
        """Get configuration provider."""
        return self._factory.get(ConfigurationProvider)

    @property
    def config_registry(self) -> ConfigRegistry:
        """Get configuration registry."""
        return self._factory.get(ConfigRegistry)

    @property
    def monitoring(self) -> MonitoringService:
        """Get monitoring service."""
        return self._factory.get(MonitoringService)

    @property
    def experiment_tracker(self) -> ExperimentTracker:
        """Get experiment tracker."""
        return self._factory.get(ExperimentTracker)


# Global factory instance
_factory = AdapterFactory()


def get_factory() -> AdapterFactory:
    """Get the global adapter factory."""
    return _factory


def get_context() -> AdapterContext:
    """Get a context with all adapters."""
    return _factory.create_context()


def configure_adapters(config: dict[str, Any]) -> None:
    """Configure the global adapter factory.
    
    Args:
        config: Configuration dictionary
    """
    _factory.configure(config)


def register_adapter(port_type: type[T], adapter: T) -> None:
    """Register an adapter in the global factory.
    
    Args:
        port_type: The port interface type
        adapter: The adapter implementation
    """
    _factory.register(port_type, adapter)