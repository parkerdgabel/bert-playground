"""Port interfaces for hexagonal architecture.

Ports define the boundaries between the core domain logic and external systems.
All external dependencies should be accessed through these port interfaces.
"""

from .compute import ComputeBackend
from .config import ConfigurationProvider
from .monitoring import MonitoringService
from .storage import StorageService

__all__ = [
    "ComputeBackend",
    "ConfigurationProvider", 
    "MonitoringService",
    "StorageService",
]