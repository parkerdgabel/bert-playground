"""Dependency injection setup for k-bert CLI.

This module configures the DI container with all CLI-specific services
and components.
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger
from rich.console import Console

from core.di import Container, get_container
from .config.loader import ConfigLoader, CachedConfigLoader, ConfigLoaderProtocol
from .config.validator import ConfigValidator, StrictConfigValidator, ConfigValidatorProtocol
from .config.merger import CliConfigMerger, ConfigMergerProtocol
from .config.resolver import EnvironmentConfigResolver, ConfigResolverProtocol
from .config.config_manager import ConfigManager
from .config.schemas import KBertConfig
from .commands.base import CommandMiddleware
from .middleware import (
    LoggingMiddleware,
    ErrorMiddleware,
    PerformanceMiddleware,
    ValidationMiddleware,
)


def setup_cli_container(
    container: Optional[Container] = None,
    strict_mode: bool = False,
    enable_cache: bool = True,
) -> Container:
    """Setup the CLI dependency injection container.
    
    Args:
        container: Base container to configure (creates new if None)
        strict_mode: Enable strict validation mode
        enable_cache: Enable configuration caching
        
    Returns:
        Configured container
    """
    if container is None:
        container = get_container()
    
    logger.debug("Setting up CLI dependency injection container")
    
    # Register console
    container.register(
        Console,
        instance=True,
        implementation=Console()
    )
    
    # Register configuration components
    _register_config_services(container, strict_mode, enable_cache)
    
    # Register middleware
    _register_middleware(container)
    
    # Register command factory
    _register_command_factory(container)
    
    logger.debug("CLI container setup complete")
    return container


def _register_config_services(
    container: Container,
    strict_mode: bool,
    enable_cache: bool
) -> None:
    """Register configuration-related services.
    
    Args:
        container: DI container
        strict_mode: Enable strict validation
        enable_cache: Enable caching
    """
    # Register ConfigLoader
    base_loader = ConfigLoader()
    if enable_cache:
        loader = CachedConfigLoader(base_loader)
    else:
        loader = base_loader
    
    container.register(
        ConfigLoaderProtocol,
        instance=True,
        implementation=loader
    )
    container.register(
        ConfigLoader,
        instance=True,
        implementation=loader
    )
    
    # Register ConfigValidator
    if strict_mode:
        validator = StrictConfigValidator()
    else:
        validator = ConfigValidator()
    
    container.register(
        ConfigValidatorProtocol,
        instance=True,
        implementation=validator
    )
    container.register(
        ConfigValidator,
        instance=True,
        implementation=validator
    )
    
    # Register ConfigMerger
    merger = CliConfigMerger()
    container.register(
        ConfigMergerProtocol,
        instance=True,
        implementation=merger
    )
    container.register(
        CliConfigMerger,
        instance=True,
        implementation=merger
    )
    
    # Register ConfigResolver
    resolver = EnvironmentConfigResolver()
    container.register(
        ConfigResolverProtocol,
        instance=True,
        implementation=resolver
    )
    container.register(
        EnvironmentConfigResolver,
        instance=True,
        implementation=resolver
    )
    
    # Register ConfigManager with injected dependencies
    container.register(
        ConfigManager,
        singleton=True,
        factory=True,
        implementation=lambda: _create_config_manager(
            loader=container.resolve(ConfigLoaderProtocol),
            validator=container.resolve(ConfigValidatorProtocol),
            merger=container.resolve(ConfigMergerProtocol),
            resolver=container.resolve(ConfigResolverProtocol),
        )
    )
    
    # Also register RefactoredConfigManager directly
    from .config.manager import RefactoredConfigManager
    container.register(
        RefactoredConfigManager,
        singleton=True,
        factory=True,
        implementation=lambda: RefactoredConfigManager(
            loader=container.resolve(ConfigLoaderProtocol),
            validator=container.resolve(ConfigValidatorProtocol),
            merger=container.resolve(ConfigMergerProtocol),
            resolver=container.resolve(ConfigResolverProtocol),
        )
    )


def _create_config_manager(
    loader: ConfigLoaderProtocol,
    validator: ConfigValidatorProtocol,
    merger: ConfigMergerProtocol,
    resolver: ConfigResolverProtocol,
) -> ConfigManager:
    """Create a ConfigManager with injected dependencies.
    
    This is a temporary factory function until ConfigManager is refactored
    to accept dependencies via constructor.
    
    Args:
        loader: Configuration loader
        validator: Configuration validator
        merger: Configuration merger
        resolver: Configuration resolver
        
    Returns:
        Configured ConfigManager
    """
    # For now, create standard ConfigManager
    # In the refactoring, we'll update ConfigManager to accept these dependencies
    manager = ConfigManager()
    
    # Inject dependencies via attributes (temporary solution)
    manager._loader = loader
    manager._validator = validator
    manager._merger = merger
    manager._resolver = resolver
    
    return manager


def _register_middleware(container: Container) -> None:
    """Register command middleware.
    
    Args:
        container: DI container
    """
    # Create middleware instances
    middleware = [
        ErrorMiddleware(),
        LoggingMiddleware(),
        ValidationMiddleware(),
        PerformanceMiddleware(),
    ]
    
    # Register as list
    container.register(
        List[CommandMiddleware],
        instance=True,
        implementation=middleware
    )
    
    # Also register individual middleware
    for mw in middleware:
        container.register(
            type(mw),
            instance=True,
            implementation=mw
        )


def _register_command_factory(container: Container) -> None:
    """Register command factory.
    
    Args:
        container: DI container
    """
    from .commands.factory import CommandFactory
    
    container.register(
        CommandFactory,
        singleton=True,
        implementation=CommandFactory
    )


def create_command_container(
    parent_container: Optional[Container] = None,
    config_overrides: Optional[dict] = None,
) -> Container:
    """Create a container for a specific command execution.
    
    Args:
        parent_container: Parent container to inherit from
        config_overrides: Configuration overrides for this command
        
    Returns:
        Command-specific container
    """
    if parent_container is None:
        parent_container = get_container()
    
    # Create child container
    container = parent_container.create_child()
    
    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            container.inject_config(key, value)
    
    return container


def inject_services(command_class: type) -> type:
    """Decorator to inject services into a command class.
    
    Args:
        command_class: Command class to enhance
        
    Returns:
        Enhanced command class
    """
    original_init = command_class.__init__
    
    def new_init(self, *args, **kwargs):
        # Get container
        container = kwargs.pop("container", None) or get_container()
        
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Initialize with container
        self.initialize(container)
    
    command_class.__init__ = new_init
    return command_class


class ConfiguredCommand:
    """Base class for commands that need configuration injection."""
    
    def __init__(self, container: Container):
        """Initialize with container.
        
        Args:
            container: DI container
        """
        self.container = container
        self._config: Optional[KBertConfig] = None
    
    @property
    def config(self) -> KBertConfig:
        """Get merged configuration."""
        if self._config is None:
            config_manager = self.container.resolve(ConfigManager)
            self._config = config_manager.get_merged_config()
        return self._config
    
    def get_service(self, service_type: type):
        """Get a service from the container.
        
        Args:
            service_type: Type of service to get
            
        Returns:
            Service instance
        """
        return self.container.resolve(service_type)