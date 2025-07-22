"""Dependency injection infrastructure for k-bert using Kink.

This module provides a clean, protocol-based dependency injection system
with support for:
- Protocol-based service registration
- Configuration injection
- Singleton and transient lifecycle management
- Auto-wiring capabilities
- Factory providers
"""

from .container import Container, get_container, reset_container
from .services import (
    injectable,
    singleton,
    provider,
    register_service,
    register_singleton,
    register_factory,
    register_instance,
)
from .providers import (
    Provider,
    SingletonProvider,
    TransientProvider,
    FactoryProvider,
    ConfigurationProvider,
)

__all__ = [
    # Container
    "Container",
    "get_container",
    "reset_container",
    # Decorators
    "injectable",
    "singleton",
    "provider",
    # Registration functions
    "register_service",
    "register_singleton",
    "register_factory",
    "register_instance",
    # Providers
    "Provider",
    "SingletonProvider",
    "TransientProvider",
    "FactoryProvider",
    "ConfigurationProvider",
]