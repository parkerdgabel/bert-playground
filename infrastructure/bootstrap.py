"""Application bootstrap for the infrastructure layer.

This module provides the main application bootstrap process using the hexagonal
architecture pattern with proper dependency injection, configuration management,
and adapter registration.

The bootstrap process follows this order:
1. Configuration loading and validation
2. Infrastructure services setup  
3. Port and adapter registration
4. Domain services registration
5. Application services registration
6. Monitoring and logging setup
"""

from pathlib import Path
from typing import Optional, Dict, Any

from .config.manager import ConfigurationManager  
from .di.container import InfrastructureContainer


class ApplicationBootstrap:
    """Main application bootstrap class for infrastructure setup.
    
    This class handles the complete application initialization using
    hexagonal architecture principles with dependency injection.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
        project_config_path: Optional[Path] = None,
    ):
        """Initialize bootstrap.
        
        Args:
            config_path: Command-specific config file path
            user_config_path: User config file path  
            project_config_path: Project config file path
        """
        self.config_path = config_path
        self.user_config_path = user_config_path
        self.project_config_path = project_config_path
        
        self.config_manager: Optional[ConfigurationManager] = None
        self.container: Optional[InfrastructureContainer] = None
        self.monitoring: Optional[Any] = None
        
        self._initialized = False
        
    def initialize(self) -> InfrastructureContainer:
        """Initialize the complete application.
        
        Returns:
            Configured infrastructure container
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return self.container
            
        try:
            # 1. Setup configuration management
            self._setup_configuration()
            
            # 2. Setup dependency injection container
            self._setup_container()
            
            # 3. Setup monitoring (must be early for logging)
            self._setup_monitoring()
            
            # 4. Initialize all services and adapters
            self._initialize_services()
            
            # 5. Validate the complete setup
            self._validate_setup()
            
            self._initialized = True
            
            # Log successful initialization
            if self.monitoring:
                self.monitoring.info(
                    "Application initialization completed successfully",
                    services=len(self.container.list_services()),
                    adapters=self._count_adapters(),
                )
                
            return self.container
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.error(
                    "Application initialization failed", 
                    error=e,
                    stage=self._get_current_stage()
                )
            raise RuntimeError(f"Application initialization failed: {e}") from e
            
    def _setup_configuration(self) -> None:
        """Setup configuration management."""
        self.config_manager = ConfigurationManager(
            user_config_path=self.user_config_path,
            project_config_path=self.project_config_path,
            command_config_path=self.config_path,
        )
        
        # Load and validate configuration
        config = self.config_manager.load_configuration()
        
        # Setup environment-specific logging level
        log_level = config.get("logging", {}).get("level", "INFO")
        import os
        os.environ.setdefault("LOGURU_LEVEL", log_level)
        
    def _setup_container(self) -> None:
        """Setup dependency injection container."""
        self.container = InfrastructureContainer(self.config_manager)
        
    def _setup_monitoring(self) -> None:
        """Setup monitoring service early for logging."""
        # Register monitoring port first
        self.container._register_monitoring_port()
        
        # Resolve monitoring service
        try:
            from ports.secondary.monitoring import MonitoringService
            self.monitoring = self.container.resolve(MonitoringService)
            
            # Configure monitoring with application context
            self.monitoring.set_context(
                application="k-bert",
                version="0.1.0",
                environment=self._get_environment(),
            )
            
        except Exception:
            # Fallback to basic logging if monitoring setup fails
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Create a wrapper to match our monitoring interface
            class LoggingWrapper:
                def info(self, message, **kwargs):
                    logger.info(message)
                def error(self, message, error=None, **kwargs):
                    if error:
                        logger.error(f"{message}: {error}")
                    else:
                        logger.error(message)
                def debug(self, message, **kwargs):
                    logger.debug(message)
                def set_context(self, **kwargs):
                    pass
                def flush(self):
                    pass
                    
            self.monitoring = LoggingWrapper()
            
    def _initialize_services(self) -> None:
        """Initialize all services and adapters."""
        if self.monitoring:
            self.monitoring.info("Initializing services and adapters")
            
        # Initialize the container with all registrations
        self.container.initialize_all()
        
        if self.monitoring:
            self.monitoring.info(
                "Services and adapters initialized",
                services=len(self.container.list_services()),
                adapters=self._count_adapters(),
            )
            
    def _validate_setup(self) -> None:
        """Validate the complete application setup."""
        if self.monitoring:
            self.monitoring.info("Validating application setup")
            
        # Validate configuration
        config = self.config_manager.load_configuration()
        
        # Validate adapter configuration
        adapter_config = config.get("adapters", {})
        validation_errors = self.container.adapter_registry.validate_configuration(adapter_config)
        
        if validation_errors:
            error_msg = "Adapter configuration validation failed:\\n" + "\\n".join(validation_errors)
            raise ValueError(error_msg)
            
        # Validate critical services are available
        self._validate_critical_services()
        
        if self.monitoring:
            self.monitoring.info("Application setup validation completed")
            
    def _validate_critical_services(self) -> None:
        """Validate that critical services can be resolved."""
        critical_services = [
            # Always required
            ("ConfigurationManager", ConfigurationManager),
        ]
        
        # Add monitoring service if available
        try:
            from infrastructure.ports.monitoring import MonitoringService
            critical_services.append(("MonitoringService", MonitoringService))
        except ImportError:
            pass
            
        # Add other critical services (optional for now)
        # TODO: Re-enable when domain services are fully implemented
        # try:
        #     from data.factory import DatasetFactory
        #     critical_services.append(("DatasetFactory", DatasetFactory))
        # except ImportError:
        #     pass
        #     
        # try:
        #     from models.factory_facade import ModelFactory
        #     critical_services.append(("ModelFactory", ModelFactory))
        # except ImportError:
        #     pass
            
        # Validate each service can be resolved
        missing_services = []
        for service_name, service_type in critical_services:
            if not self.container.has(service_type):
                missing_services.append(service_name)
                
        if missing_services:
            raise RuntimeError(
                f"Critical services not available: {missing_services}"
            )
            
    def _get_environment(self) -> str:
        """Get current environment name."""
        import os
        return os.environ.get("K_BERT_ENV", "development")
        
    def _get_current_stage(self) -> str:
        """Get current initialization stage for error reporting."""
        if not self.config_manager:
            return "configuration"
        elif not self.container:
            return "container"
        elif not self.monitoring:
            return "monitoring"
        else:
            return "services"
            
    def _count_adapters(self) -> int:
        """Count registered adapters."""
        if not self.container:
            return 0
        all_adapters = self.container.adapter_registry.list_all_adapters()
        return sum(len(adapters) for adapters in all_adapters.values())
        
    def get_service(self, service_type):
        """Get a service from the container.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
            
        Raises:
            RuntimeError: If application not initialized
        """
        if not self._initialized:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        return self.container.resolve(service_type)
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        if not self.config_manager:
            return default
        return self.config_manager.get_config(key, default)
        
    def reload_configuration(self) -> None:
        """Reload configuration and reinitialize affected services."""
        if not self.config_manager:
            return
            
        if self.monitoring:
            self.monitoring.info("Reloading configuration")
            
        # Reload configuration
        self.config_manager.reload_configuration()
        
        # Re-register ports and adapters with new configuration
        if self.container:
            self.container.register_ports_and_adapters()
            
        if self.monitoring:
            self.monitoring.info("Configuration reloaded successfully")
            
    def shutdown(self) -> None:
        """Shutdown the application gracefully."""
        if self.monitoring:
            self.monitoring.info("Shutting down application")
            
        # Flush monitoring/logging
        if self.monitoring and hasattr(self.monitoring, 'flush'):
            self.monitoring.flush()
            
        # Clear container
        if self.container:
            self.container.clear()
            
        self._initialized = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get application status information.
        
        Returns:
            Status information dictionary
        """
        status = {
            "initialized": self._initialized,
            "environment": self._get_environment(),
        }
        
        if self.container:
            status.update({
                "services": len(self.container.list_services()),
                "adapters": self._count_adapters(),
                "adapter_info": {
                    port_type: len(adapters)
                    for port_type, adapters in self.container.adapter_registry.list_all_adapters().items()
                }
            })
            
        if self.config_manager:
            try:
                config = self.config_manager.load_configuration()
                status["config_sections"] = list(config.keys())
            except Exception:
                status["config_sections"] = []
                
        return status


# Global bootstrap instance
_bootstrap: Optional[ApplicationBootstrap] = None


def get_bootstrap(
    config_path: Optional[Path] = None,
    user_config_path: Optional[Path] = None,
    project_config_path: Optional[Path] = None,
) -> ApplicationBootstrap:
    """Get the global bootstrap instance.
    
    Args:
        config_path: Command-specific config path
        user_config_path: User config path
        project_config_path: Project config path
        
    Returns:
        Bootstrap instance
    """
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = ApplicationBootstrap(
            config_path=config_path,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
    return _bootstrap


def initialize_application(
    config_path: Optional[Path] = None,
    user_config_path: Optional[Path] = None, 
    project_config_path: Optional[Path] = None,
) -> InfrastructureContainer:
    """Initialize the complete application.
    
    Args:
        config_path: Command-specific config path
        user_config_path: User config path
        project_config_path: Project config path
        
    Returns:
        Configured container
    """
    bootstrap = get_bootstrap(config_path, user_config_path, project_config_path)
    return bootstrap.initialize()


def get_service(service_type):
    """Get a service from the application.
    
    Args:
        service_type: Service type to resolve
        
    Returns:
        Service instance
    """
    bootstrap = get_bootstrap()
    return bootstrap.get_service(service_type)


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from the application.
    
    Args:
        key: Configuration key (dot notation)
        default: Default value
        
    Returns:
        Configuration value
    """
    bootstrap = get_bootstrap()
    return bootstrap.get_config(key, default)


def shutdown_application() -> None:
    """Shutdown the application gracefully."""
    global _bootstrap
    if _bootstrap:
        _bootstrap.shutdown()
        _bootstrap = None