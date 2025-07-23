"""Service and adapter registry for infrastructure DI.

This module provides registry classes to track registered services and adapters,
enabling runtime introspection and management.
"""

from typing import Any, Dict, Type, List, Optional


class ServiceRegistry:
    """Registry for tracking registered services."""
    
    def __init__(self):
        """Initialize service registry."""
        self._services: Dict[str, Type] = {}
        self._service_metadata: Dict[str, Dict[str, Any]] = {}
        
    def register_service(
        self, 
        name: str, 
        service_type: Type,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service with the registry.
        
        Args:
            name: Service name
            service_type: Service class/type
            metadata: Optional service metadata
        """
        self._services[name] = service_type
        self._service_metadata[name] = metadata or {}
        
    def get_service_type(self, name: str) -> Optional[Type]:
        """Get service type by name.
        
        Args:
            name: Service name
            
        Returns:
            Service type or None if not found
        """
        return self._services.get(name)
        
    def get_service_metadata(self, name: str) -> Dict[str, Any]:
        """Get service metadata by name.
        
        Args:
            name: Service name
            
        Returns:
            Service metadata
        """
        return self._service_metadata.get(name, {})
        
    def list_services(self) -> Dict[str, Type]:
        """List all registered services.
        
        Returns:
            Service name -> type mapping
        """
        return self._services.copy()
        
    def list_services_by_domain(self, domain: str) -> Dict[str, Type]:
        """List services by domain.
        
        Args:
            domain: Domain name (e.g., "training", "data", "model")
            
        Returns:
            Services in the domain
        """
        domain_services = {}
        for name, service_type in self._services.items():
            if name.startswith(domain) or domain in name:
                domain_services[name] = service_type
        return domain_services
        
    def has_service(self, name: str) -> bool:
        """Check if service is registered.
        
        Args:
            name: Service name
            
        Returns:
            True if registered
        """
        return name in self._services
        
    def unregister_service(self, name: str) -> bool:
        """Unregister a service.
        
        Args:
            name: Service name
            
        Returns:
            True if service was registered and removed
        """
        if name in self._services:
            del self._services[name]
            del self._service_metadata[name]
            return True
        return False
        
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._service_metadata.clear()


class AdapterRegistry:
    """Registry for tracking registered adapters."""
    
    def __init__(self):
        """Initialize adapter registry."""
        self._adapters: Dict[str, Dict[str, Type]] = {}
        self._adapter_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
    def register_adapter(
        self,
        port_type: str,
        implementation: str, 
        adapter_type: Type,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an adapter for a port.
        
        Args:
            port_type: Port type (e.g., "monitoring", "storage")
            implementation: Implementation name (e.g., "loguru", "filesystem")
            adapter_type: Adapter class/type
            metadata: Optional adapter metadata
        """
        if port_type not in self._adapters:
            self._adapters[port_type] = {}
            self._adapter_metadata[port_type] = {}
            
        self._adapters[port_type][implementation] = adapter_type
        self._adapter_metadata[port_type][implementation] = metadata or {}
        
    def get_adapter_type(self, port_type: str, implementation: str) -> Optional[Type]:
        """Get adapter type for a port implementation.
        
        Args:
            port_type: Port type
            implementation: Implementation name
            
        Returns:
            Adapter type or None if not found
        """
        port_adapters = self._adapters.get(port_type, {})
        return port_adapters.get(implementation)
        
    def get_adapter_metadata(
        self, 
        port_type: str, 
        implementation: str
    ) -> Dict[str, Any]:
        """Get adapter metadata.
        
        Args:
            port_type: Port type
            implementation: Implementation name
            
        Returns:
            Adapter metadata
        """
        port_metadata = self._adapter_metadata.get(port_type, {})
        return port_metadata.get(implementation, {})
        
    def list_adapters_for_port(self, port_type: str) -> Dict[str, Type]:
        """List all adapters for a port type.
        
        Args:
            port_type: Port type
            
        Returns:
            Implementation -> adapter type mapping
        """
        return self._adapters.get(port_type, {}).copy()
        
    def list_all_adapters(self) -> Dict[str, Dict[str, Type]]:
        """List all registered adapters.
        
        Returns:
            Port type -> implementation -> adapter type mapping
        """
        return {
            port_type: adapters.copy()
            for port_type, adapters in self._adapters.items()
        }
        
    def list_port_types(self) -> List[str]:
        """List all registered port types.
        
        Returns:
            List of port type names
        """
        return list(self._adapters.keys())
        
    def has_adapter(self, port_type: str, implementation: str) -> bool:
        """Check if adapter is registered.
        
        Args:
            port_type: Port type
            implementation: Implementation name
            
        Returns:
            True if registered
        """
        return (port_type in self._adapters and 
                implementation in self._adapters[port_type])
        
    def get_default_adapter(self, port_type: str) -> Optional[Type]:
        """Get the first registered adapter for a port (as default).
        
        Args:
            port_type: Port type
            
        Returns:
            Default adapter type or None
        """
        port_adapters = self._adapters.get(port_type, {})
        if port_adapters:
            return next(iter(port_adapters.values()))
        return None
        
    def get_adapter_info(self, port_type: str) -> Dict[str, Any]:
        """Get comprehensive information about adapters for a port.
        
        Args:
            port_type: Port type
            
        Returns:
            Adapter information including types and metadata
        """
        port_adapters = self._adapters.get(port_type, {})
        port_metadata = self._adapter_metadata.get(port_type, {})
        
        info = {
            "port_type": port_type,
            "implementations": {},
            "count": len(port_adapters)
        }
        
        for impl_name, adapter_type in port_adapters.items():
            info["implementations"][impl_name] = {
                "type": adapter_type,
                "module": adapter_type.__module__,
                "name": adapter_type.__name__, 
                "metadata": port_metadata.get(impl_name, {})
            }
            
        return info
        
    def unregister_adapter(self, port_type: str, implementation: str) -> bool:
        """Unregister an adapter.
        
        Args:
            port_type: Port type
            implementation: Implementation name
            
        Returns:
            True if adapter was registered and removed
        """
        if self.has_adapter(port_type, implementation):
            del self._adapters[port_type][implementation]
            del self._adapter_metadata[port_type][implementation]
            
            # Clean up empty port types
            if not self._adapters[port_type]:
                del self._adapters[port_type]
                del self._adapter_metadata[port_type]
                
            return True
        return False
        
    def clear_port(self, port_type: str) -> None:
        """Clear all adapters for a port type.
        
        Args:
            port_type: Port type to clear
        """
        if port_type in self._adapters:
            del self._adapters[port_type]
            del self._adapter_metadata[port_type]
            
    def clear(self) -> None:
        """Clear all registered adapters."""
        self._adapters.clear()
        self._adapter_metadata.clear()
        
    def validate_configuration(self, config: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate adapter configuration against registered adapters.
        
        Args:
            config: Adapter configuration
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for port_type, port_config in config.items():
            implementation = port_config.get("implementation")
            
            if not implementation:
                errors.append(f"No implementation specified for port: {port_type}")
                continue
                
            if not self.has_adapter(port_type, implementation):
                available = list(self._adapters.get(port_type, {}).keys())
                errors.append(
                    f"Unknown adapter '{implementation}' for port '{port_type}'. "
                    f"Available: {available}"
                )
                
        return errors