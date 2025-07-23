"""Provider implementations for different object lifecycle strategies.

This module contains providers that manage how objects are created and
their lifecycle within the dependency injection container.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
import inspect
from threading import Lock
from loguru import logger

T = TypeVar("T")


class Provider(ABC):
    """Base provider interface for object creation strategies."""
    
    @abstractmethod
    def get(self, container: "Container") -> Any:
        """Get an instance from the provider.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            The created instance
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset the provider state."""
        pass


class SingletonProvider(Provider):
    """Provider that ensures only one instance is created (singleton pattern)."""
    
    def __init__(self, implementation: Union[Type[T], Callable[[], T]]):
        """Initialize singleton provider.
        
        Args:
            implementation: Class or factory function to create the instance
        """
        self._implementation = implementation
        self._instance: Optional[T] = None
        self._lock = Lock()
        
    def get(self, container: "Container") -> T:
        """Get the singleton instance, creating it if necessary.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            The singleton instance
        """
        if self._instance is None:
            with self._lock:
                # Double-check locking
                if self._instance is None:
                    self._instance = self._create_instance(container)
                    logger.debug(
                        f"Created singleton instance of "
                        f"{self._implementation.__name__}"
                    )
        return self._instance
        
    def reset(self) -> None:
        """Reset the singleton instance."""
        with self._lock:
            self._instance = None
            
    def _create_instance(self, container: "Container") -> T:
        """Create an instance using the implementation.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            The created instance
        """
        if inspect.isclass(self._implementation):
            # Auto-wire dependencies for classes
            return container.auto_wire(self._implementation)
        else:
            # Call factory function
            return self._implementation()


class TransientProvider(Provider):
    """Provider that creates a new instance every time (transient lifecycle)."""
    
    def __init__(self, implementation: Union[Type[T], Callable[[], T]]):
        """Initialize transient provider.
        
        Args:
            implementation: Class or factory function to create instances
        """
        self._implementation = implementation
        
    def get(self, container: "Container") -> T:
        """Create a new instance.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            A new instance
        """
        if inspect.isclass(self._implementation):
            # Auto-wire dependencies for classes
            instance = container.auto_wire(self._implementation)
        else:
            # Call factory function
            instance = self._implementation()
            
        logger.debug(f"Created transient instance of {self._implementation.__name__}")
        return instance
        
    def reset(self) -> None:
        """No-op for transient provider."""
        pass


class FactoryProvider(Provider):
    """Provider that delegates creation to a factory function."""
    
    def __init__(self, factory: Callable[..., T]):
        """Initialize factory provider.
        
        Args:
            factory: Factory function to create instances
        """
        self._factory = factory
        
    def get(self, container: "Container") -> T:
        """Call the factory function to create an instance.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            The created instance
        """
        # Inspect factory parameters and auto-wire if needed
        sig = inspect.signature(self._factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                continue
                
            # Try to resolve from container
            if container.has(param_type):
                kwargs[param_name] = container.resolve(param_type)
            elif param.default != inspect.Parameter.empty:
                kwargs[param_name] = param.default
                
        instance = self._factory(**kwargs)
        logger.debug(f"Created instance via factory {self._factory.__name__}")
        return instance
        
    def reset(self) -> None:
        """No-op for factory provider."""
        pass


class ConfigurationProvider(Provider):
    """Provider that creates instances from configuration data.
    
    Useful for creating configured instances of services based on
    configuration files or environment variables.
    """
    
    def __init__(
        self,
        implementation: Type[T],
        config_key: str,
        config_factory: Optional[Callable[[Dict[str, Any]], T]] = None,
    ):
        """Initialize configuration provider.
        
        Args:
            implementation: The class to instantiate
            config_key: Configuration key to look up
            config_factory: Optional factory to create instance from config
        """
        self._implementation = implementation
        self._config_key = config_key
        self._config_factory = config_factory
        self._instance: Optional[T] = None
        self._lock = Lock()
        
    def get(self, container: "Container") -> T:
        """Get configured instance.
        
        Args:
            container: The DI container for resolving dependencies
            
        Returns:
            The configured instance
        """
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    # Get configuration
                    config = container.get_config(self._config_key, {})
                    
                    if self._config_factory:
                        # Use custom factory
                        self._instance = self._config_factory(config)
                    else:
                        # Create instance with config as kwargs
                        self._instance = self._implementation(**config)
                        
                    logger.debug(
                        f"Created configured instance of "
                        f"{self._implementation.__name__} with config key "
                        f"'{self._config_key}'"
                    )
                    
        return self._instance
        
    def reset(self) -> None:
        """Reset the configured instance."""
        with self._lock:
            self._instance = None