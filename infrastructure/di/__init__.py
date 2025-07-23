"""Dependency injection infrastructure for the hexagonal architecture.

This module provides:
- Enhanced container with port/adapter registration
- Service registration for all domains
- Configuration-driven adapter selection
- Lifecycle management
"""

from .container import Container, InfrastructureContainer
from .registry import ServiceRegistry, AdapterRegistry

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


__all__ = [
    "Container",
    "InfrastructureContainer",
    "ServiceRegistry", 
    "AdapterRegistry",
    "get_container",
]