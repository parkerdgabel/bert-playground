"""Base compute adapter with common functionality."""

from abc import ABC
from typing import Any, Dict, Optional
from ports.secondary.compute import ComputeBackend


class BaseComputeAdapter(ABC, ComputeBackend):
    """Base implementation of ComputeBackend with common functionality."""
    
    def __init__(self):
        """Initialize base compute adapter."""
        self._device_cache: Optional[Dict[str, Any]] = None
        
    
    def get_device_memory(self) -> Dict[str, int]:
        """Get device memory information.
        
        Returns:
            Dictionary with 'total' and 'available' memory in bytes
        """
        device_info = self.get_device_info()
        return {
            "total": device_info.get("memory_total", 0),
            "available": device_info.get("memory_available", 0),
        }
    
    def clear_cache(self) -> None:
        """Clear any cached data."""
        self._device_cache = None