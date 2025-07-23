"""Factory for creating framework adapters.

This module provides a factory pattern for creating framework adapters,
allowing easy switching between different ML frameworks.
"""

from typing import Dict, Type, Optional
from functools import lru_cache

from core.protocols.training import FrameworkAdapter as IFrameworkAdapter
from training.adapters.framework_adapter import FrameworkAdapter
from training.adapters.mlx_adapter import MLXFrameworkAdapter
from loguru import logger


class FrameworkAdapterFactory:
    """Factory for creating framework adapters."""
    
    # Registry of available adapters
    _adapters: Dict[str, Type[IFrameworkAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[IFrameworkAdapter]) -> None:
        """Register a framework adapter.
        
        Args:
            name: Name of the framework
            adapter_class: Adapter class
        """
        cls._adapters[name.lower()] = adapter_class
        logger.debug(f"Registered framework adapter: {name}")
    
    @classmethod
    @lru_cache(maxsize=None)
    def create(cls, framework: str = "mlx") -> IFrameworkAdapter:
        """Create a framework adapter.
        
        Args:
            framework: Name of the framework
            
        Returns:
            Framework adapter instance
            
        Raises:
            ValueError: If framework is not supported
        """
        framework_lower = framework.lower()
        
        # Use the comprehensive adapter by default
        if framework_lower in ["mlx", "auto"]:
            adapter = FrameworkAdapter(backend=framework_lower)
            logger.info(f"Created {framework} framework adapter")
            return adapter
        
        # Check registry for custom adapters
        if framework_lower in cls._adapters:
            adapter_class = cls._adapters[framework_lower]
            adapter = adapter_class()
            logger.info(f"Created {framework} framework adapter from registry")
            return adapter
        
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Available: {list(cls._adapters.keys()) + ['mlx', 'auto']}"
        )
    
    @classmethod
    def get_available_frameworks(cls) -> list[str]:
        """Get list of available frameworks.
        
        Returns:
            List of framework names
        """
        return ["mlx", "auto"] + list(cls._adapters.keys())
    
    @classmethod
    def is_framework_available(cls, framework: str) -> bool:
        """Check if a framework is available.
        
        Args:
            framework: Framework name
            
        Returns:
            True if framework is available
        """
        try:
            adapter = cls.create(framework)
            return adapter.available
        except ValueError:
            return False


# Singleton instance
_factory = FrameworkAdapterFactory()


def get_framework_adapter(framework: str = "mlx") -> IFrameworkAdapter:
    """Get a framework adapter.
    
    Args:
        framework: Framework name
        
    Returns:
        Framework adapter instance
    """
    return _factory.create(framework)


def register_adapter(name: str, adapter_class: Type[IFrameworkAdapter]) -> None:
    """Register a custom framework adapter.
    
    Args:
        name: Framework name
        adapter_class: Adapter class
    """
    _factory.register(name, adapter_class)


def get_available_frameworks() -> list[str]:
    """Get list of available frameworks.
    
    Returns:
        List of framework names
    """
    return _factory.get_available_frameworks()


def is_framework_available(framework: str) -> bool:
    """Check if a framework is available.
    
    Args:
        framework: Framework name
        
    Returns:
        True if framework is available
    """
    return _factory.is_framework_available(framework)