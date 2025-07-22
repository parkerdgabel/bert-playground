"""Main dependency injection container for k-bert.

This module provides the core container implementation with support for:
- Protocol-based registration and resolution
- Hierarchical containers with parent/child relationships
- Auto-wiring based on type hints
- Configuration injection
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, get_type_hints
from kink import Container as KinkContainer, di
import inspect
from loguru import logger

from .providers import Provider, SingletonProvider, TransientProvider

T = TypeVar("T")

# Global container instance
_container: Optional["Container"] = None


class Container:
    """Enhanced dependency injection container built on top of Kink.
    
    Provides additional features:
    - Protocol-based service registration
    - Auto-wiring capabilities
    - Configuration injection support
    - Lifecycle management
    """
    
    def __init__(self, parent: Optional["Container"] = None):
        """Initialize container with optional parent container.
        
        Args:
            parent: Parent container for hierarchical resolution
        """
        self._kink_container = KinkContainer()
        self._parent = parent
        self._providers: Dict[Type, Provider] = {}
        self._protocols: Dict[Type, Type] = {}  # Protocol -> Implementation mapping
        
        # Configure Kink to use our container
        di._initialized = True
        di._container = self._kink_container
        
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], T]] = None,
        *,
        singleton: bool = False,
        factory: bool = False,
        instance: bool = False,
    ) -> None:
        """Register a service with the container.
        
        Args:
            service_type: The type/protocol to register
            implementation: The implementation (class, instance, or factory)
            singleton: Whether to use singleton lifecycle
            factory: Whether implementation is a factory function
            instance: Whether implementation is an instance
        """
        if implementation is None:
            implementation = service_type
            
        # Store protocol mapping if service_type is a Protocol
        if hasattr(service_type, "__protocol__"):
            self._protocols[service_type] = implementation
            
        # Create appropriate provider
        if instance:
            # Direct instance registration
            self._kink_container[service_type] = implementation
            logger.debug(f"Registered instance: {service_type.__name__}")
        elif factory:
            # Factory function registration
            # Use Kink's factory mechanism
            self._kink_container.factories[service_type] = lambda _: implementation()
            logger.debug(f"Registered factory: {service_type.__name__}")
        elif singleton:
            # Singleton registration
            provider = SingletonProvider(implementation)
            self._providers[service_type] = provider
            self._kink_container[service_type] = lambda _: provider.get(self)
            logger.debug(f"Registered singleton: {service_type.__name__}")
        else:
            # Transient registration - use factories for fresh instances
            provider = TransientProvider(implementation)
            self._providers[service_type] = provider
            self._kink_container.factories[service_type] = lambda _: provider.get(self)
            logger.debug(f"Registered transient: {service_type.__name__}")
            
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The type to resolve
            
        Returns:
            The resolved instance
            
        Raises:
            KeyError: If the service is not registered
        """
        # Check if it's a protocol and we have a mapping
        if hasattr(service_type, "__protocol__") and service_type in self._protocols:
            implementation_type = self._protocols[service_type]
            return self.resolve(implementation_type)
            
        # Try to resolve from current container
        if service_type in self._kink_container:
            # Kink returns the resolved value directly
            return self._kink_container[service_type]
            
        # Try parent container if available
        if self._parent:
            return self._parent.resolve(service_type)
            
        raise KeyError(f"Service {service_type.__name__} not registered")
        
    def has(self, service_type: Type) -> bool:
        """Check if a service is registered.
        
        Args:
            service_type: The type to check
            
        Returns:
            True if the service is registered
        """
        if service_type in self._kink_container:
            return True
        if hasattr(service_type, "__protocol__") and service_type in self._protocols:
            return True
        if self._parent:
            return self._parent.has(service_type)
        return False
        
    def create_child(self) -> "Container":
        """Create a child container.
        
        Returns:
            A new container with this container as parent
        """
        return Container(parent=self)
        
    def auto_wire(self, cls: Type[T]) -> T:
        """Create an instance of a class by auto-wiring its dependencies.
        
        Args:
            cls: The class to instantiate
            
        Returns:
            An instance with dependencies injected
        """
        # Get constructor parameters
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            # Get type hint
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                continue
                
            # Try to resolve the dependency
            if self.has(param_type):
                kwargs[param_name] = self.resolve(param_type)
            elif param.default != inspect.Parameter.empty:
                # Use default value if available
                kwargs[param_name] = param.default
            else:
                raise ValueError(
                    f"Cannot auto-wire parameter '{param_name}' of type "
                    f"'{param_type}' for class '{cls.__name__}'"
                )
                
        return cls(**kwargs)
        
    def inject_config(self, config_key: str, config_value: Any) -> None:
        """Inject a configuration value into the container.
        
        Args:
            config_key: The configuration key
            config_value: The configuration value
        """
        self._kink_container[f"config:{config_key}"] = config_value
        
    def get_config(self, config_key: str, default: Any = None) -> Any:
        """Get a configuration value from the container.
        
        Args:
            config_key: The configuration key
            default: Default value if not found
            
        Returns:
            The configuration value
        """
        full_key = f"config:{config_key}"
        if full_key in self._kink_container:
            return self._kink_container[full_key]
        if self._parent:
            return self._parent.get_config(config_key, default)
        return default
        
    def clear(self) -> None:
        """Clear all registrations from the container."""
        self._kink_container._services.clear()
        self._kink_container._factories.clear()
        self._kink_container._memoized_services.clear()
        self._providers.clear()
        self._protocols.clear()
        logger.debug("Container cleared")


def get_container() -> Container:
    """Get the global container instance.
    
    Returns:
        The global container
    """
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Reset the global container."""
    global _container
    if _container:
        _container.clear()
    _container = None
    logger.debug("Global container reset")