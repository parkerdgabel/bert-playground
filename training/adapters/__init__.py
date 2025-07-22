"""Framework adapters for training components.

This module provides adapters that abstract framework-specific implementations
(MLX, PyTorch, etc.) behind common interfaces.
"""

from .base import FrameworkAdapter
from .mlx_adapter import MLXFrameworkAdapter
from .registry import get_framework_adapter, register_adapter, list_adapters

__all__ = [
    "FrameworkAdapter",
    "MLXFrameworkAdapter",
    "get_framework_adapter",
    "register_adapter", 
    "list_adapters",
]