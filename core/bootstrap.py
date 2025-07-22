"""
Application bootstrap for hexagonal architecture with dependency injection.

This module sets up the entire application dependency graph, configuring
all ports, adapters, and services.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from .di.container import Container, get_container
from .ports.compute import ComputeBackend
from .ports.storage import StorageService, ModelStorageService
from .ports.config import ConfigurationProvider
from .ports.monitoring import MonitoringService
from .ports.tokenizer import TokenizerPort, TokenizerFactory
from .adapters.mlx_adapter import MLXComputeAdapter
from .adapters.file_storage import FileStorageAdapter, ModelFileStorageAdapter
from .adapters.yaml_config import YAMLConfigAdapter
from .adapters.loguru_monitoring import LoguruMonitoringAdapter
from .adapters.huggingface_tokenizer import HuggingFaceTokenizerFactory
from .events.bus import EventBus, GlobalEventBus
from .plugins.registry import PluginRegistry
from .plugins.loader import PluginLoader


class ApplicationBootstrap:
    """Bootstrap class for setting up the entire application."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the bootstrap with optional configuration path.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.container = get_container()
        self._initialized = False
    
    def initialize(self) -> Container:
        """
        Initialize the entire application dependency graph.
        
        Returns:
            The configured container
        """
        if self._initialized:
            return self.container
            
        logger.info("Initializing k-bert application with hexagonal architecture")
        
        # 1. Setup core infrastructure
        self._setup_infrastructure()
        
        # 2. Setup ports and adapters
        self._setup_ports_and_adapters()
        
        # 3. Setup domain services
        self._setup_domain_services()
        
        # 4. Setup application services
        self._setup_application_services()
        
        # 5. Setup CLI and entry points
        self._setup_cli_layer()
        
        # 6. Load plugins
        self._setup_plugins()
        
        self._initialized = True
        logger.info("Application initialization complete")
        return self.container
    
    def _setup_infrastructure(self) -> None:
        """Setup core infrastructure components."""
        logger.debug("Setting up core infrastructure")
        
        # Event bus (singleton)
        event_bus = GlobalEventBus()
        self.container.register(EventBus, event_bus, instance=True)
        
        # Configuration
        if self.config_path and self.config_path.exists():
            config_adapter = YAMLConfigAdapter(str(self.config_path))
        else:
            config_adapter = YAMLConfigAdapter()
        self.container.register(ConfigurationProvider, config_adapter, singleton=True)
        
        logger.debug("Core infrastructure setup complete")
    
    def _setup_ports_and_adapters(self) -> None:
        """Setup ports and their corresponding adapters."""
        logger.debug("Setting up ports and adapters")
        
        # Compute backend
        self.container.register(ComputeBackend, MLXComputeAdapter, singleton=True)
        
        # Storage services
        self.container.register(StorageService, FileStorageAdapter, singleton=True)
        self.container.register(ModelStorageService, ModelFileStorageAdapter, singleton=True)
        
        # Monitoring
        self.container.register(MonitoringService, LoguruMonitoringAdapter, singleton=True)
        
        # Tokenizer
        self.container.register(TokenizerFactory, HuggingFaceTokenizerFactory, singleton=True)
        
        logger.debug("Ports and adapters setup complete")
    
    def _setup_domain_services(self) -> None:
        """Setup domain-specific services."""
        logger.debug("Setting up domain services")
        
        # Import domain services here to avoid circular imports
        from models.factory_facade import ModelFactory
        from data.factory import DatasetFactory
        
        # Register domain factories
        self.container.register(ModelFactory, singleton=True)
        self.container.register(DatasetFactory, singleton=True)
        
        logger.debug("Domain services setup complete")
    
    def _setup_application_services(self) -> None:
        """Setup application-level services."""
        logger.debug("Setting up application services")
        
        # Import application services
        from training.components.training_orchestrator import TrainingOrchestrator
        from training.components.training_loop import TrainingLoop
        from training.components.evaluation_loop import EvaluationLoop
        from training.components.checkpoint_manager import CheckpointManager
        from training.components.metrics_tracker import MetricsTracker
        
        # Register training components (transient - new instance per request)
        self.container.register(TrainingLoop)
        self.container.register(EvaluationLoop)
        self.container.register(CheckpointManager)
        self.container.register(MetricsTracker)
        self.container.register(TrainingOrchestrator)
        
        logger.debug("Application services setup complete")
    
    def _setup_cli_layer(self) -> None:
        """Setup CLI layer with dependency injection."""
        logger.debug("Setting up CLI layer")
        
        # Import CLI components
        from cli.factory import CommandFactory
        
        # Register CLI infrastructure
        self.container.register(CommandFactory, singleton=True)
        
        logger.debug("CLI layer setup complete")
    
    def _setup_plugins(self) -> None:
        """Setup plugin system."""
        logger.debug("Setting up plugin system")
        
        # Plugin infrastructure
        plugin_registry = PluginRegistry()
        plugin_loader = PluginLoader()
        
        self.container.register(PluginRegistry, plugin_registry, instance=True)
        self.container.register(PluginLoader, plugin_loader, instance=True)
        
        # Load and register plugins
        try:
            plugins = plugin_loader.discover_plugins()
            for plugin in plugins:
                plugin_registry.register(plugin)
                
            logger.info(f"Loaded {len(plugins)} plugins")
        except Exception as e:
            logger.warning(f"Plugin loading failed: {e}")
        
        logger.debug("Plugin system setup complete")
    
    def get_service(self, service_type):
        """
        Convenience method to get a service from the container.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            The resolved service instance
        """
        return self.container.resolve(service_type)


# Global bootstrap instance
_bootstrap: Optional[ApplicationBootstrap] = None


def get_bootstrap(config_path: Optional[Path] = None) -> ApplicationBootstrap:
    """
    Get the global bootstrap instance.
    
    Args:
        config_path: Optional configuration path
        
    Returns:
        The bootstrap instance
    """
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = ApplicationBootstrap(config_path)
    return _bootstrap


def initialize_application(config_path: Optional[Path] = None) -> Container:
    """
    Initialize the entire application.
    
    Args:
        config_path: Optional configuration path
        
    Returns:
        The configured container
    """
    bootstrap = get_bootstrap(config_path)
    return bootstrap.initialize()


def get_service(service_type):
    """
    Convenience function to get a service from the application.
    
    Args:
        service_type: The service type to resolve
        
    Returns:
        The resolved service instance
    """
    bootstrap = get_bootstrap()
    if not bootstrap._initialized:
        bootstrap.initialize()
    return bootstrap.get_service(service_type)