"""Domain-internal service registry.

This module provides a pure domain registry for services without any
infrastructure dependencies. The actual wiring happens in the application layer.
"""

from typing import Dict, Type, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ServiceScope(Enum):
    """Service lifecycle scope."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"


@dataclass
class ServiceMetadata:
    """Metadata for registered domain services."""
    service_type: Type
    implementation: Optional[Type] = None
    scope: ServiceScope = ServiceScope.TRANSIENT
    factory: Optional[Callable] = None
    dependencies: Dict[str, Type] = field(default_factory=dict)


class DomainServiceRegistry:
    """Registry for domain services.
    
    This is a pure domain construct that tracks service metadata.
    The actual dependency injection happens in the application layer.
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceMetadata] = {}
        
    def register(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        scope: ServiceScope = ServiceScope.TRANSIENT,
        factory: Optional[Callable] = None
    ) -> None:
        """Register a domain service.
        
        Args:
            service_type: The service interface/type
            implementation: The implementation class (defaults to service_type)
            scope: Service lifecycle scope
            factory: Optional factory function
        """
        self._services[service_type] = ServiceMetadata(
            service_type=service_type,
            implementation=implementation or service_type,
            scope=scope,
            factory=factory
        )
        
    def get_metadata(self, service_type: Type) -> Optional[ServiceMetadata]:
        """Get metadata for a registered service."""
        return self._services.get(service_type)
        
    def list_services(self) -> Dict[Type, ServiceMetadata]:
        """List all registered services."""
        return self._services.copy()
        
    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()


# Global registry instance
_registry = DomainServiceRegistry()


def domain_service(
    scope: ServiceScope = ServiceScope.TRANSIENT,
    implementation: Optional[Type] = None
) -> Callable[[Type], Type]:
    """Decorator for domain services.
    
    This is a pure domain decorator that doesn't depend on infrastructure.
    
    Args:
        scope: Service lifecycle scope
        implementation: Optional implementation class
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type) -> Type:
        _registry.register(
            service_type=cls,
            implementation=implementation,
            scope=scope
        )
        return cls
    return decorator


def get_domain_registry() -> DomainServiceRegistry:
    """Get the global domain registry."""
    return _registry