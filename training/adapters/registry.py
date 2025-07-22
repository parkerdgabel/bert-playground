"""Framework adapter registry.

This module manages registration and retrieval of framework adapters.
"""

from typing import Any

from .base import FrameworkAdapter


class AdapterRegistry:
    """Registry for framework adapters."""
    
    def __init__(self):
        """Initialize registry."""
        self._adapters: dict[str, FrameworkAdapter] = {}
        self._default_adapter: str | None = None
    
    def register(
        self,
        adapter: FrameworkAdapter,
        set_as_default: bool = False
    ) -> None:
        """Register a framework adapter.
        
        Args:
            adapter: Adapter to register
            set_as_default: Whether to set as default adapter
        """
        self._adapters[adapter.name.lower()] = adapter
        
        if set_as_default or self._default_adapter is None:
            if adapter.available:
                self._default_adapter = adapter.name.lower()
    
    def get(self, name: str | None = None) -> FrameworkAdapter:
        """Get adapter by name.
        
        Args:
            name: Adapter name (if None, returns default)
            
        Returns:
            Framework adapter
            
        Raises:
            KeyError: If adapter not found
            RuntimeError: If no adapters available
        """
        if name is None:
            if self._default_adapter is None:
                raise RuntimeError("No default adapter available")
            name = self._default_adapter
        
        name = name.lower()
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found. Available: {list(self._adapters.keys())}")
        
        adapter = self._adapters[name]
        if not adapter.available:
            raise RuntimeError(f"Adapter '{name}' is not available")
        
        return adapter
    
    def list_adapters(self) -> dict[str, dict[str, Any]]:
        """List all registered adapters.
        
        Returns:
            Dictionary of adapter info
        """
        return {
            name: {
                "available": adapter.available,
                "is_default": name == self._default_adapter,
            }
            for name, adapter in self._adapters.items()
        }
    
    def get_available_adapters(self) -> list[str]:
        """Get list of available adapter names."""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.available
        ]
    
    def auto_detect(self) -> str | None:
        """Auto-detect best available adapter.
        
        Returns:
            Name of best adapter or None if none available
        """
        # Priority order for auto-detection
        priority_order = ["mlx", "pytorch", "tensorflow", "jax"]
        
        for framework in priority_order:
            if framework in self._adapters and self._adapters[framework].available:
                return framework
        
        # Fallback to first available
        available = self.get_available_adapters()
        return available[0] if available else None
    
    def set_default(self, name: str) -> None:
        """Set default adapter.
        
        Args:
            name: Adapter name
            
        Raises:
            KeyError: If adapter not found
            RuntimeError: If adapter not available
        """
        name = name.lower()
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found")
        
        if not self._adapters[name].available:
            raise RuntimeError(f"Adapter '{name}' is not available")
        
        self._default_adapter = name


# Global registry instance
_registry = AdapterRegistry()

# Initialize with available adapters
def _initialize_registry():
    """Initialize registry with available adapters."""
    # Register MLX adapter
    try:
        from .mlx_adapter import MLXFrameworkAdapter
        mlx_adapter = MLXFrameworkAdapter()
        _registry.register(mlx_adapter, set_as_default=True)
    except ImportError:
        pass
    
    # Register PyTorch adapter (placeholder)
    # try:
    #     from .pytorch_adapter import PyTorchFrameworkAdapter
    #     pytorch_adapter = PyTorchFrameworkAdapter()
    #     _registry.register(pytorch_adapter)
    # except ImportError:
    #     pass

_initialize_registry()


def register_adapter(
    adapter: FrameworkAdapter,
    set_as_default: bool = False
) -> None:
    """Register a framework adapter globally.
    
    Args:
        adapter: Adapter to register
        set_as_default: Whether to set as default
    """
    _registry.register(adapter, set_as_default)


def get_framework_adapter(name: str | None = None) -> FrameworkAdapter:
    """Get framework adapter by name.
    
    Args:
        name: Adapter name (if None, returns default)
        
    Returns:
        Framework adapter
    """
    return _registry.get(name)


def list_adapters() -> dict[str, dict[str, Any]]:
    """List all registered adapters."""
    return _registry.list_adapters()


def auto_detect_framework() -> str | None:
    """Auto-detect best available framework."""
    return _registry.auto_detect()


def set_default_adapter(name: str) -> None:
    """Set default framework adapter.
    
    Args:
        name: Adapter name
    """
    _registry.set_default(name)


def get_available_frameworks() -> list[str]:
    """Get list of available framework names."""
    return _registry.get_available_adapters()