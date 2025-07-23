"""CLI Bootstrap - Dependency injection setup for the CLI.

This module handles the initialization of all adapters and services for the CLI,
following the hexagonal architecture pattern.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from infrastructure.bootstrap import ApplicationBootstrap, get_bootstrap
from infrastructure.di.container import InfrastructureContainer

# Import adapters
from adapters.secondary.compute.mlx.compute_adapter import MLXComputeAdapter
from adapters.secondary.monitoring.console.console_adapter import ConsoleMonitoringAdapter
from adapters.secondary.monitoring.composite.multi_monitor import MultiMonitorAdapter
from adapters.secondary.storage.filesystem import FilesystemStorageAdapter
from adapters.secondary.data.mlx.data_loader import MLXDataLoaderAdapter
from adapters.secondary.tokenizer.huggingface.tokenizer_adapter import HuggingFaceTokenizerAdapter

# Import application commands
from application.commands.train import TrainModelCommand
from application.commands.evaluate import EvaluateModelCommand
from application.commands.predict import PredictCommand
from application.commands.export import ExportModelCommand

# Import domain services
from domain.services.training import ModelTrainingService
from domain.services.tokenization import TokenizationService
from domain.services.checkpointing import CheckpointingService
from domain.services.evaluation_service import EvaluationService

# Import ports
from domain.ports import (
    ComputePort,
    MonitoringPort,
    StoragePort,
    DataLoaderPort,
    TokenizerPort,
    CheckpointPort,
    MetricsCalculatorPort,
)


class CLIBootstrap:
    """Bootstrap class for CLI application.
    
    This class sets up all the dependencies and adapters needed by the CLI,
    ensuring that the application layer commands have everything they need.
    """
    
    def __init__(self):
        """Initialize the CLI bootstrap."""
        self.bootstrap: Optional[ApplicationBootstrap] = None
        self.container: Optional[InfrastructureContainer] = None
        self._initialized = False
        
    def initialize(
        self,
        config_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
        project_config_path: Optional[Path] = None,
    ) -> None:
        """Initialize the CLI application with all dependencies.
        
        Args:
            config_path: Command-specific configuration file
            user_config_path: User configuration file (~/.k-bert/config.yaml)
            project_config_path: Project configuration file (k-bert.yaml)
        """
        if self._initialized:
            return
            
        # Initialize the application bootstrap
        self.bootstrap = get_bootstrap(
            config_path=config_path,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
        
        # Initialize the container
        self.container = self.bootstrap.initialize()
        
        # Register CLI-specific services
        self._register_cli_services()
        
        self._initialized = True
        
    def _register_cli_services(self) -> None:
        """Register CLI-specific services and adapters."""
        # The main adapters should already be registered by the infrastructure bootstrap
        # Here we can add any CLI-specific overrides or additional services
        
        # Register application commands
        self._register_application_commands()
        
    def _register_application_commands(self) -> None:
        """Register application layer commands."""
        # Register TrainModelCommand
        self.container.register_factory(
            TrainModelCommand,
            lambda c: TrainModelCommand(
                training_service=c.resolve(ModelTrainingService),
                tokenization_service=c.resolve(TokenizationService),
                checkpointing_service=c.resolve(CheckpointingService),
                data_loader_port=c.resolve(DataLoaderPort),
                compute_port=c.resolve(ComputePort),
                monitoring_port=c.resolve(MonitoringPort),
                storage_port=c.resolve(StoragePort),
                checkpoint_port=c.resolve(CheckpointPort),
                metrics_port=c.resolve(MetricsCalculatorPort),
            )
        )
        
        # Register EvaluateModelCommand
        self.container.register_factory(
            EvaluateModelCommand,
            lambda c: EvaluateModelCommand(
                evaluation_service=c.resolve(EvaluationService),
                tokenization_service=c.resolve(TokenizationService),
                data_loader_port=c.resolve(DataLoaderPort),
                compute_port=c.resolve(ComputePort),
                monitoring_port=c.resolve(MonitoringPort),
                storage_port=c.resolve(StoragePort),
                metrics_port=c.resolve(MetricsCalculatorPort),
            )
        )
        
        # Register PredictCommand
        self.container.register_factory(
            PredictCommand,
            lambda c: PredictCommand(
                tokenization_service=c.resolve(TokenizationService),
                data_loader_port=c.resolve(DataLoaderPort),
                compute_port=c.resolve(ComputePort),
                monitoring_port=c.resolve(MonitoringPort),
                storage_port=c.resolve(StoragePort),
            )
        )
        
        # Register ExportModelCommand
        self.container.register_factory(
            ExportModelCommand,
            lambda c: ExportModelCommand(
                storage_port=c.resolve(StoragePort),
                monitoring_port=c.resolve(MonitoringPort),
            )
        )
        
    def get_command(self, command_type: type) -> Any:
        """Get an application command instance.
        
        Args:
            command_type: Type of command to get
            
        Returns:
            Command instance
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("CLI not initialized. Call initialize() first.")
        return self.container.resolve(command_type)
        
    def get_service(self, service_type: type) -> Any:
        """Get a service instance.
        
        Args:
            service_type: Type of service to get
            
        Returns:
            Service instance
        """
        if not self._initialized:
            raise RuntimeError("CLI not initialized. Call initialize() first.")
        return self.container.resolve(service_type)
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self.bootstrap:
            return default
        return self.bootstrap.get_config(key, default)
        
    def shutdown(self) -> None:
        """Shutdown the CLI application gracefully."""
        if self.bootstrap:
            self.bootstrap.shutdown()
        self._initialized = False


# Global CLI bootstrap instance
_cli_bootstrap: Optional[CLIBootstrap] = None


def get_cli_bootstrap() -> CLIBootstrap:
    """Get the global CLI bootstrap instance.
    
    Returns:
        CLI bootstrap instance
    """
    global _cli_bootstrap
    if _cli_bootstrap is None:
        _cli_bootstrap = CLIBootstrap()
    return _cli_bootstrap


def initialize_cli(
    config_path: Optional[Path] = None,
    user_config_path: Optional[Path] = None,
    project_config_path: Optional[Path] = None,
) -> None:
    """Initialize the CLI application.
    
    Args:
        config_path: Command-specific configuration file
        user_config_path: User configuration file
        project_config_path: Project configuration file
    """
    bootstrap = get_cli_bootstrap()
    bootstrap.initialize(
        config_path=config_path,
        user_config_path=user_config_path,
        project_config_path=project_config_path,
    )


def get_command(command_type: type) -> Any:
    """Get an application command instance.
    
    Args:
        command_type: Type of command to get
        
    Returns:
        Command instance
    """
    bootstrap = get_cli_bootstrap()
    return bootstrap.get_command(command_type)


def get_service(service_type: type) -> Any:
    """Get a service instance.
    
    Args:
        service_type: Type of service to get
        
    Returns:
        Service instance
    """
    bootstrap = get_cli_bootstrap()
    return bootstrap.get_service(service_type)


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value.
    
    Args:
        key: Configuration key
        default: Default value
        
    Returns:
        Configuration value
    """
    bootstrap = get_cli_bootstrap()
    return bootstrap.get_config(key, default)


def shutdown_cli() -> None:
    """Shutdown the CLI application."""
    global _cli_bootstrap
    if _cli_bootstrap:
        _cli_bootstrap.shutdown()
        _cli_bootstrap = None