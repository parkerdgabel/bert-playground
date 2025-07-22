"""Service registration helpers and decorators for dependency injection.

This module provides convenient decorators and functions for registering
services with the DI container.
"""

from typing import Any, Callable, Optional, Type, TypeVar, Union, overload
from functools import wraps
from loguru import logger

from .container import get_container
from .providers import FactoryProvider

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# Decorators

def injectable(
    cls: Optional[Type[T]] = None,
    *,
    bind_to: Optional[Type] = None,
    singleton: bool = False,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorator to mark a class as injectable.
    
    Can be used with or without parameters:
    - @injectable: Register as transient
    - @injectable(singleton=True): Register as singleton
    - @injectable(bind_to=Protocol): Register implementation for protocol
    
    Args:
        cls: The class to register (when used without parentheses)
        bind_to: Protocol/interface to bind the implementation to
        singleton: Whether to use singleton lifecycle
        
    Returns:
        The decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        service_type = bind_to or cls
        container.register(service_type, cls, singleton=singleton)
        return cls
        
    if cls is None:
        # Called with parameters: @injectable(...)
        return decorator
    else:
        # Called without parameters: @injectable
        return decorator(cls)


def singleton(
    cls: Optional[Type[T]] = None,
    *,
    bind_to: Optional[Type] = None,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorator to register a class as a singleton.
    
    Shorthand for @injectable(singleton=True).
    
    Args:
        cls: The class to register
        bind_to: Protocol/interface to bind the implementation to
        
    Returns:
        The decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return injectable(cls, bind_to=bind_to, singleton=True)
        
    if cls is None:
        return decorator
    else:
        return decorator(cls)


def provider(
    func: Optional[F] = None,
    *,
    bind_to: Optional[Type] = None,
    singleton: bool = False,
) -> Union[F, Callable[[F], F]]:
    """Decorator to register a factory function as a provider.
    
    The function will be called to create instances when the service is resolved.
    
    Args:
        func: The factory function
        bind_to: Type to register the factory for
        singleton: Whether to cache the result
        
    Returns:
        The decorated function
    """
    def decorator(func: F) -> F:
        container = get_container()
        
        # Determine the service type from return annotation or bind_to
        if bind_to:
            service_type = bind_to
        else:
            import inspect
            sig = inspect.signature(func)
            if sig.return_annotation == inspect.Parameter.empty:
                raise ValueError(
                    f"Provider function '{func.__name__}' must have a return "
                    "type annotation or specify bind_to parameter"
                )
            service_type = sig.return_annotation
            
        # Create factory provider
        if singleton:
            # Wrap in a singleton provider
            _instance = None
            
            @wraps(func)
            def singleton_factory():
                nonlocal _instance
                if _instance is None:
                    _instance = func()
                return _instance
                
            container.register(service_type, singleton_factory, factory=True)
        else:
            container.register(service_type, func, factory=True)
            
        return func
        
    if func is None:
        return decorator
    else:
        return decorator(func)


# Registration functions

def register_service(
    service_type: Type[T],
    implementation: Type[T],
    *,
    singleton: bool = False,
) -> None:
    """Register a service with the container.
    
    Args:
        service_type: The type/protocol to register
        implementation: The implementation class
        singleton: Whether to use singleton lifecycle
    """
    container = get_container()
    container.register(service_type, implementation, singleton=singleton)


def register_singleton(
    service_type: Type[T],
    implementation: Type[T],
) -> None:
    """Register a singleton service with the container.
    
    Args:
        service_type: The type/protocol to register
        implementation: The implementation class
    """
    register_service(service_type, implementation, singleton=True)


def register_factory(
    service_type: Type[T],
    factory: Callable[[], T],
    *,
    singleton: bool = False,
) -> None:
    """Register a factory function with the container.
    
    Args:
        service_type: The type to register
        factory: The factory function
        singleton: Whether to cache the result
    """
    container = get_container()
    container.register(service_type, factory, factory=True)


def register_instance(
    service_type: Type[T],
    instance: T,
) -> None:
    """Register an existing instance with the container.
    
    Args:
        service_type: The type to register
        instance: The instance to register
    """
    container = get_container()
    container.register(service_type, instance, instance=True)