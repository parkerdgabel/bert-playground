"""Dependency injection infrastructure for the hexagonal architecture.

This module provides:
- Enhanced container with port/adapter registration
- Service registration for all domains
- Configuration-driven adapter selection
- Lifecycle management
"""

from .container import InfrastructureContainer
from .registry import ServiceRegistry, AdapterRegistry

__all__ = [
    "InfrastructureContainer",
    "ServiceRegistry", 
    "AdapterRegistry",
]