"""Application bootstrap for the infrastructure layer.

This module provides the main application bootstrap process using the hexagonal
architecture pattern with proper dependency injection and configuration management.

The bootstrap process follows this order:
1. Configuration loading and validation
2. Infrastructure container setup with auto-discovery
3. Service initialization and validation
"""

from pathlib import Path
from typing import Optional, Dict, Any, Set, List

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
            
            # 3. Initialize the container (auto-discovery happens here)
            self.container.initialize()
            
            # 4. Wire domain services into the container
            self._wire_domain_services()
            
            # 5. Validate the complete setup
            self._validate_setup()
            
            self._initialized = True
            
            return self.container
            
        except Exception as e:
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
            
    def _wire_domain_services(self) -> None:
        """Wire domain services into the DI container.
        
        This bridges the pure domain registry with the infrastructure DI system.
        """
        try:
            # Resolve the domain service wiring from application layer
            from application.services.domain_service_wiring import DomainServiceWiring
            
            # Check if it's already registered (from auto-discovery)
            if self.container.has(DomainServiceWiring):
                wiring_service = self.container.resolve(DomainServiceWiring)
            else:
                # Manually register and resolve it
                wiring_service = DomainServiceWiring()
                self.container.core_container.register(
                    DomainServiceWiring, 
                    wiring_service, 
                    instance=True
                )
            
            # Wire all domain services
            wiring_service.wire_domain_services(self.container.core_container)
            
        except ImportError as e:
            # Domain service wiring is optional - only needed if domain uses the registry
            pass
        except Exception as e:
            raise RuntimeError(f"Failed to wire domain services: {e}") from e
    
    def _validate_setup(self) -> None:
        """Validate the complete application setup."""
        # Validate configuration is loaded
        config = self.config_manager.load_configuration()
        
        # Validate critical service is available
        if not self.container.has(ConfigurationManager):
            raise RuntimeError("ConfigurationManager not available in container")
            
        # Basic health check
        health = self.container.health_check()
        if not health.get("initialized"):
            raise RuntimeError("Container failed to initialize properly")
            
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
        
    def shutdown(self) -> None:
        """Shutdown the application gracefully."""
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
        }
        
        if self.container:
            health = self.container.health_check()
            status.update({
                "container_initialized": health.get("initialized", False),
                "services_count": health.get("services_count", 0),
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
    bootstrap = get_bootstrap(
        config_path, user_config_path, project_config_path
    )
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