"""Dependency injection container implementation for k-bert.

This module provides two container classes:
1. Container - Core DI functionality
2. InfrastructureContainer - Application-level container with auto-discovery
"""

from typing import (
    Any, Callable, Dict, List, Optional, Type, TypeVar, Union,
    get_origin, get_args
)
import inspect
from pathlib import Path
from collections import defaultdict

from ..config.manager import ConfigurationManager
from .decorators import (
    ComponentMetadata, ComponentType, Scope, 
    get_component_metadata, qualifier, value
)

T = TypeVar("T")


class Container:
    """Basic dependency injection container.
    
    Provides core DI functionality including:
    - Service registration
    - Dependency resolution
    - Lifecycle management (singleton/transient)
    """
    
    def __init__(self):
        """Initialize the container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._singleton_types: set = set()
        
        # Enhanced metadata tracking
        self._metadata: Dict[Type, ComponentMetadata] = {}
        self._qualifiers: Dict[str, Dict[Type, Type]] = defaultdict(dict)
        self._primary_implementations: Dict[Type, Type] = {}
        self._config_values: Dict[str, Any] = {}
        
    def register(
        self, 
        service_type: Type[T], 
        implementation: Any = None,
        factory: Optional[Callable] = None,
        instance: bool = False,
        singleton: bool = False,
        metadata: Optional[ComponentMetadata] = None
    ) -> None:
        """Register a service in the container.
        
        Args:
            service_type: The service interface/type
            implementation: The implementation class
            factory: Factory function to create instances
            instance: Whether implementation is already an instance
            singleton: Whether to use singleton lifecycle
            metadata: Optional component metadata
        """
        # Store metadata if provided
        if metadata:
            self._metadata[service_type] = metadata
            
            # Handle qualifiers
            for qualifier_name, qualifier_value in metadata.qualifiers.items():
                if qualifier_name == "primary" and qualifier_value:
                    self._primary_implementations[service_type] = implementation or service_type
                elif isinstance(qualifier_name, str):
                    self._qualifiers[qualifier_name][service_type] = implementation or service_type
        
        # Auto-detect metadata from decorated classes
        if not metadata and implementation and hasattr(implementation, '__class__'):
            metadata = get_component_metadata(implementation)
            if metadata:
                self._metadata[service_type] = metadata
                
                # Handle singleton scope from metadata
                if metadata.scope == Scope.SINGLETON:
                    singleton = True
        
        if instance:
            # Register an existing instance
            self._services[service_type] = implementation
            self._singleton_types.add(service_type)
        elif factory:
            # Register a factory function
            self._factories[service_type] = factory
            if singleton:
                self._singleton_types.add(service_type)
        else:
            # Register a class
            self._services[service_type] = implementation or service_type
            if singleton:
                self._singleton_types.add(service_type)
    
    def register_factory(self, service_type: Type[T], factory: Callable[["Container"], T]) -> None:
        """Register a factory function for a service type.
        
        Args:
            service_type: The service type to register
            factory: Factory function that takes container and returns instance
        """
        self._factories[service_type] = factory
        
    def resolve(self, service_type: Type[T], qualifier_name: Optional[str] = None) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The type to resolve
            qualifier_name: Optional qualifier for specific implementation
            
        Returns:
            The resolved service instance
            
        Raises:
            ValueError: If service is not registered or cannot be resolved
        """
        # Handle qualified resolution
        if qualifier_name:
            qualified_implementations = self._qualifiers.get(qualifier_name, {})
            if service_type in qualified_implementations:
                actual_type = qualified_implementations[service_type]
                return self._resolve_type(actual_type)
        
        # Handle primary implementation
        if service_type in self._primary_implementations:
            actual_type = self._primary_implementations[service_type]
            return self._resolve_type(actual_type)
        
        # Standard resolution
        return self._resolve_type(service_type)
    
    def _resolve_type(self, service_type: Type[T]) -> T:
        """Internal method to resolve a specific type."""
        # Check for singleton instance
        if service_type in self._singleton_types and service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check for factory
        if service_type in self._factories:
            factory = self._factories[service_type]
            instance = factory(self)
            
            # Cache if singleton
            if service_type in self._singleton_types:
                self._singletons[service_type] = instance
            
            return instance
        
        # Check for registered service
        if service_type in self._services:
            implementation = self._services[service_type]
            
            # If it's already an instance, return it
            if not inspect.isclass(implementation):
                return implementation
            
            # Create new instance
            instance = self._create_instance(implementation)
            
            # Cache if singleton
            if service_type in self._singleton_types:
                self._singletons[service_type] = instance
            
            return instance
        
        # Try to auto-wire if it's a class
        if inspect.isclass(service_type):
            instance = self._create_instance(service_type)
            return instance
        
        raise ValueError(f"Cannot resolve service: {service_type}")
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with dependency injection."""
        return self.auto_wire(implementation)
    
    def auto_wire(self, implementation: Type[T]) -> T:
        """Auto-wire dependencies for a class constructor.
        
        Args:
            implementation: The class to instantiate
            
        Returns:
            Instance with dependencies injected
        """
        if not inspect.isclass(implementation):
            return implementation
        
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = param.annotation
            
            # Skip if no type annotation
            if param_type == inspect.Parameter.empty:
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                continue
            
            # Handle Optional types
            origin = get_origin(param_type)
            if origin is Union:
                args = get_args(param_type)
                if len(args) == 2 and type(None) in args:
                    # This is Optional[T]
                    actual_type = args[0] if args[1] is type(None) else args[1]
                    try:
                        kwargs[param_name] = self.resolve(actual_type)
                    except ValueError:
                        # Optional dependency not available
                        kwargs[param_name] = None
                    continue
            
            # Handle List types
            if origin is list:
                args = get_args(param_type)
                if args:
                    element_type = args[0]
                    # Find all implementations of this type
                    implementations = []
                    for registered_type, registered_impl in self._services.items():
                        if self._is_assignable(registered_impl, element_type):
                            implementations.append(self.resolve(registered_type))
                    kwargs[param_name] = implementations
                    continue
            
            # Handle Set types
            if origin is set:
                args = get_args(param_type)
                if args:
                    element_type = args[0]
                    # Find all implementations of this type
                    implementations = set()
                    for registered_type, registered_impl in self._services.items():
                        if self._is_assignable(registered_impl, element_type):
                            implementations.add(self.resolve(registered_type))
                    kwargs[param_name] = implementations
                    continue
            
            # Try to resolve the dependency
            try:
                kwargs[param_name] = self.resolve(param_type)
            except ValueError:
                # Check if parameter has default value
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                else:
                    raise ValueError(f"Cannot resolve dependency '{param_name}' of type {param_type} for {implementation}")
        
        return implementation(**kwargs)
    
    def _is_assignable(self, implementation: Type, interface: Type) -> bool:
        """Check if implementation is assignable to interface."""
        if inspect.isclass(implementation):
            return issubclass(implementation, interface)
        return isinstance(implementation, interface)
    
    def has(self, service_type: Type) -> bool:
        """Check if a service type is registered.
        
        Args:
            service_type: The service type to check
            
        Returns:
            True if the service is registered
        """
        return (service_type in self._services or 
                service_type in self._factories or
                service_type in self._singletons)
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._singleton_types.clear()
        self._metadata.clear()
        self._qualifiers.clear()
        self._primary_implementations.clear()
        self._config_values.clear()
    
    def list_services(self) -> list:
        """List all registered service types.
        
        Returns:
            List of registered service types
        """
        all_types = set()
        all_types.update(self._services.keys())
        all_types.update(self._factories.keys())
        all_types.update(self._singletons.keys())
        return list(all_types)
    
    def inject_config(self, key: str, value: Any) -> None:
        """Inject a configuration value.
        
        Args:
            key: Configuration key (dot-separated)
            value: Configuration value
        """
        self._config_values[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config_values.get(key, default)
    
    def register_decorator(self, cls: Type[T]) -> Type[T]:
        """Register a decorated class with the container.
        
        Args:
            cls: The decorated class to register
            
        Returns:
            The class (for chaining)
        """
        metadata = get_component_metadata(cls)
        if not metadata:
            raise ValueError(f"Class {cls} is not decorated with a DI decorator")
        
        # Determine what to register this class as
        if metadata.component_type == ComponentType.ADAPTER:
            # Register as adapter for its port
            if metadata.port_type:
                self.register(
                    metadata.port_type, 
                    cls,
                    singleton=(metadata.scope == Scope.SINGLETON),
                    metadata=metadata
                )
        else:
            # Register as itself
            self.register(
                cls,
                cls, 
                singleton=(metadata.scope == Scope.SINGLETON),
                metadata=metadata
            )
        
        return cls
    
    def create_child(self) -> "Container":
        """Create a child container that inherits from this one.
        
        Returns:
            New child container
        """
        child = Container()
        
        # Copy services but not singletons
        child._services = self._services.copy()
        child._factories = self._factories.copy()
        child._singleton_types = self._singleton_types.copy()
        child._metadata = self._metadata.copy()
        child._qualifiers = {k: v.copy() for k, v in self._qualifiers.items()}
        child._primary_implementations = self._primary_implementations.copy()
        child._config_values = self._config_values.copy()
        return child


class InfrastructureContainer:
    """Enhanced DI container for hexagonal architecture infrastructure.
    
    This container extends the core DI functionality with:
    - Automatic component discovery via decorators
    - Configuration-driven adapter selection
    - Application lifecycle management
    - Health checks and validation
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize infrastructure container.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.core_container = Container()
        self.config_manager = config_manager or ConfigurationManager()
        self._initialized = False
        
        # Register the configuration manager itself
        self.core_container.register(
            ConfigurationManager, 
            self.config_manager, 
            instance=True
        )
        
    def initialize(self) -> None:
        """Initialize the container with auto-discovery and configuration.
        
        This replaces the previous manual registration methods with a clean
        auto-discovery approach using decorators.
        """
        if self._initialized:
            return
            
        try:
            # Auto-discover all decorated components
            self._auto_discover_components()
            
            # Apply configuration-driven adapter selection
            self._configure_adapters()
            
            # Validate required components are present
            self._validate_setup()
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InfrastructureContainer: {e}") from e
    
    def _auto_discover_components(self) -> None:
        """Auto-discover and register all decorated components."""
        from .scanner import auto_discover_and_register
        
        # Package paths to scan for decorated components
        package_paths = [
            "domain",
            "application",  # This includes application.ports now
            "infrastructure.adapters",
            "adapters"
            # "ports" removed - now in application.ports
        ]
        
        # Auto-discover and register components
        discovered_count = auto_discover_and_register(
            self.core_container,
            package_paths=package_paths,
            profiles=None,  # Use default profile for now
            validate=True  # Re-enabled after fixing dictionary iteration bug
        )
        
        if discovered_count == 0:
            # This might be normal in some environments, so just log a warning
            pass
    
    def _configure_adapters(self) -> None:
        """Configure adapters based on configuration settings."""
        # Get adapter configuration
        config = self.config_manager.load_configuration()
        adapter_config = config.get("adapters", {})
        
        if not adapter_config:
            # No adapter configuration, use defaults
            return
        
        from .decorators import ComponentType, get_component_metadata
        
        # Process each configured adapter selection
        for port_name, adapter_spec in adapter_config.items():
            if isinstance(adapter_spec, dict) and "implementation" in adapter_spec:
                impl_name = adapter_spec["implementation"]
                
                # Find the adapter class by name
                for service_type in self.core_container.list_services():
                    metadata = self.core_container._metadata.get(service_type)
                    if (metadata and 
                        metadata.component_type == ComponentType.ADAPTER and
                        service_type.__name__.lower() == impl_name.lower()):
                        
                        # Found the adapter, update its priority to make it primary
                        metadata.priority = 1000  # High priority for configured adapters
                        
                        # If qualifiers are specified, add them
                        if "qualifiers" in adapter_spec:
                            metadata.qualifiers.update(adapter_spec["qualifiers"])
    
    def _validate_setup(self) -> None:
        """Validate that required components are properly registered."""
        # Basic validation - ensure core container is functional
        if not self.core_container.has(ConfigurationManager):
            raise RuntimeError("ConfigurationManager not registered")
        
        # Additional validation can be added here for required ports/adapters
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            The resolved service instance
        """
        # Ensure container is initialized
        if not self._initialized:
            self.initialize()
            
        return self.core_container.resolve(service_type)
    
    def has(self, service_type: Type) -> bool:
        """Check if a service type is registered.
        
        Args:
            service_type: The service type to check
            
        Returns:
            True if the service is registered
        """
        return self.core_container.has(service_type)
    
    def create_child(self) -> "InfrastructureContainer":
        """Create a child container.
        
        Returns:
            New child InfrastructureContainer
        """
        child = InfrastructureContainer(self.config_manager)
        child.core_container = self.core_container.create_child()
        return child
    
    def get_adapter_info(self, port_type: str) -> Dict[str, Any]:
        """Get information about registered adapters for a port.
        
        Args:
            port_type: The port type name
            
        Returns:
            Adapter information dictionary
        """
        from .decorators import ComponentType, get_component_metadata
        
        implementations = {}
        
        # Search through all registered services for adapters of this port
        for service_type in self.core_container.list_services():
            metadata = self.core_container._metadata.get(service_type)
            if metadata and metadata.component_type == ComponentType.ADAPTER:
                # Check if this adapter implements the specified port
                if metadata.port_type and metadata.port_type.__name__ == port_type:
                    implementations[service_type.__name__] = {
                        "type": service_type,
                        "scope": metadata.scope.value,
                        "priority": metadata.priority,
                        "qualifiers": dict(metadata.qualifiers),
                        "profiles": list(metadata.profiles) if metadata.profiles else []
                    }
        
        return {
            "port_type": port_type,
            "implementations": implementations,
            "count": len(implementations)
        }
    
    def list_services(self) -> Dict[str, Type]:
        """List all registered services.
        
        Returns:
            Dictionary of service names to types
        """
        services = {}
        
        # Get all registered services from core container
        for service_type in self.core_container.list_services():
            # Use the service's class name as key
            services[service_type.__name__] = service_type
            
        return services
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the container.
        
        Returns:
            Health check results
        """
        from .decorators import ComponentType
        
        health = {
            "initialized": self._initialized,
            "services_count": len(self.core_container.list_services()),
            "config_manager_available": self.has(ConfigurationManager),
            "components_by_type": {},
            "singleton_count": 0,
            "transient_count": 0,
            "adapters": {},
            "ports_with_adapters": []
        }
        
        # Count components by type and scope
        for service_type in self.core_container.list_services():
            metadata = self.core_container._metadata.get(service_type)
            if metadata:
                # Count by component type
                comp_type = metadata.component_type.value
                health["components_by_type"][comp_type] = health["components_by_type"].get(comp_type, 0) + 1
                
                # Count by scope
                if metadata.scope.value == "singleton":
                    health["singleton_count"] += 1
                else:
                    health["transient_count"] += 1
                
                # Track adapters and their ports
                if metadata.component_type == ComponentType.ADAPTER and metadata.port_type:
                    port_name = metadata.port_type.__name__
                    if port_name not in health["adapters"]:
                        health["adapters"][port_name] = []
                        health["ports_with_adapters"].append(port_name)
                    health["adapters"][port_name].append(service_type.__name__)
        
        # Add container state info
        health["total_metadata_entries"] = len(self.core_container._metadata)
        health["total_qualifiers"] = len(self.core_container._qualifiers)
        
        return health
    
    def clear(self) -> None:
        """Clear all registrations."""
        self.core_container.clear()
        self._initialized = False
    
    def get_service_metadata(self, service_type: Type) -> Optional[ComponentMetadata]:
        """Get metadata for a registered service.
        
        Args:
            service_type: The service type
            
        Returns:
            Component metadata or None if not found
        """
        return self.core_container._metadata.get(service_type)
    
    def get_services_by_type(self, component_type: ComponentType) -> List[Type]:
        """Get all services of a specific component type.
        
        Args:
            component_type: The component type to filter by
            
        Returns:
            List of service types
        """
        from .decorators import ComponentType
        
        services = []
        for service_type in self.core_container.list_services():
            metadata = self.core_container._metadata.get(service_type)
            if metadata and metadata.component_type == component_type:
                services.append(service_type)
        return services
    
    def get_primary_implementation(self, interface: Type) -> Optional[Type]:
        """Get the primary implementation for an interface.
        
        Args:
            interface: The interface type
            
        Returns:
            Primary implementation type or None
        """
        return self.core_container._primary_implementations.get(interface)