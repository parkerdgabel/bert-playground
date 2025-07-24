"""Dependency injection infrastructure for the hexagonal architecture.

This module provides:
- Enhanced container with port/adapter registration
- Auto-discovery decorator system
- Configuration-driven adapter selection
- Lifecycle management
"""

from .container import Container, InfrastructureContainer

# Import decorators
from .decorators import (
    # Core decorators
    component,
    service,
    application_service,
    use_case,
    port,
    adapter,
    repository,
    factory,
    handler,
    controller,
    gateway,
    
    # Qualifier decorators
    qualifier,
    primary,
    
    # Lifecycle decorators
    post_construct,
    pre_destroy,
    lazy,
    
    # Conditional decorators
    conditional,
    profile,
    profiles,
    
    # Configuration decorators
    value,
    
    # Types
    ComponentType,
    Scope,
    ComponentMetadata,
    
    # Utilities
    get_component_metadata,
    get_registered_components,
    clear_registry,
)

# Import scanner
from .scanner import (
    ComponentScanner,
    auto_discover_and_register,
)

# Global container instance
_container: InfrastructureContainer | None = None


def get_container() -> InfrastructureContainer:
    """Get the global container instance.
    
    Returns:
        The global InfrastructureContainer instance
        
    Raises:
        RuntimeError: If container has not been initialized
    """
    global _container
    if _container is None:
        _container = InfrastructureContainer()
    return _container


def reset_container() -> None:
    """Reset the global container instance."""
    global _container
    if _container:
        _container.core_container.clear()
    _container = None
    clear_registry()


__all__ = [
    # Containers
    "Container",
    "InfrastructureContainer",
    
    # Core decorators
    "component",
    "service",
    "application_service",
    "use_case",
    "port",
    "adapter",
    "repository",
    "factory",
    "handler",
    "controller",
    "gateway",
    
    # Qualifier decorators
    "qualifier",
    "primary",
    
    # Lifecycle decorators
    "post_construct",
    "pre_destroy",
    "lazy",
    
    # Conditional decorators
    "conditional",
    "profile",
    "profiles",
    
    # Configuration decorators
    "value",
    
    # Types
    "ComponentType",
    "Scope",
    "ComponentMetadata",
    
    # Scanner
    "ComponentScanner",
    "auto_discover_and_register",
    
    # Utilities
    "get_container",
    "reset_container",
    "get_component_metadata",
    "get_registered_components",
]