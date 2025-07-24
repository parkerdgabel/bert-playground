"""Service registration helpers for dependency injection.

This module provides convenient functions for registering services with the DI container.
Legacy decorators have been removed in favor of the new decorator system in decorators.py.
"""

from typing import Any, Callable, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .container import InfrastructureContainer

T = TypeVar("T")


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
    from . import get_container
    container = get_container()
    container.core_container.register(service_type, implementation, singleton=singleton)


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
    from . import get_container
    container = get_container()
    container.core_container.register(service_type, factory, factory=True)


def register_instance(
    service_type: Type[T],
    instance: T,
) -> None:
    """Register an existing instance with the container.
    
    Args:
        service_type: The type to register
        instance: The instance to register
    """
    from . import get_container
    container = get_container()
    container.core_container.register(service_type, instance, instance=True)