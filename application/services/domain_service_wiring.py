"""Application service for wiring domain services into the DI container.

This bridges the pure domain registry with the infrastructure DI system,
maintaining the hexagonal architecture boundary.
"""

from typing import Type
from infrastructure.di import service, Scope as DIScope
from infrastructure.di.container import Container
from domain.registry import get_domain_registry, ServiceScope


@service(scope="singleton")
class DomainServiceWiring:
    """Wires domain services into the infrastructure DI container.
    
    This is an application-layer concern that bridges the domain
    registry with the infrastructure DI system.
    """
    
    def __init__(self):
        self.domain_registry = get_domain_registry()
        
    def wire_domain_services(self, container: Container) -> None:
        """Wire all registered domain services into the DI container.
        
        Args:
            container: The DI container to wire services into
        """
        for service_type, metadata in self.domain_registry.list_services().items():
            # Convert domain scope to DI scope
            di_scope = (
                DIScope.SINGLETON 
                if metadata.scope == ServiceScope.SINGLETON 
                else DIScope.TRANSIENT
            )
            
            # Register in DI container with proper scope
            # The container will handle the actual instantiation and dependency injection
            container.register(
                service_type=service_type,
                implementation=metadata.implementation,
                factory=metadata.factory,
                singleton=(di_scope == DIScope.SINGLETON)
            )
            
    def register_domain_service_manually(
        self, 
        service_type: Type,
        container: Container
    ) -> None:
        """Register a specific domain service in the DI container.
        
        Args:
            service_type: The domain service to register
            container: The DI container
        """
        metadata = self.domain_registry.get_metadata(service_type)
        if not metadata:
            raise ValueError(f"Domain service {service_type} not found in registry")
            
        di_scope = (
            DIScope.SINGLETON 
            if metadata.scope == ServiceScope.SINGLETON 
            else DIScope.TRANSIENT
        )
        
        container.register(
            service_type=service_type,
            implementation=metadata.implementation,
            factory=metadata.factory,
            singleton=(di_scope == DIScope.SINGLETON)
        )