"""Infrastructure package for k-bert hexagonal architecture.

This package contains:
- Configuration management system
- Dependency injection container setup  
- Application bootstrap logic
- Infrastructure adapters for cross-cutting concerns

The infrastructure layer serves as the foundation for the hexagonal architecture,
providing the plumbing that wires together all the ports and adapters.
"""

__version__ = "0.1.0"

from .bootstrap import ApplicationBootstrap, initialize_application
from .config.manager import ConfigurationManager
from .di.container import InfrastructureContainer

__all__ = [
    "ApplicationBootstrap",
    "initialize_application", 
    "ConfigurationManager",
    "InfrastructureContainer",
]