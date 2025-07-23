"""Framework adapters for training components.

This module provides adapters that abstract framework-specific implementations
(MLX, PyTorch, etc.) behind common interfaces.
"""

from .base import FrameworkAdapter as BaseFrameworkAdapter
from .mlx_adapter import MLXFrameworkAdapter
from .framework_adapter import FrameworkAdapter
from .factory import (
    get_framework_adapter,
    register_adapter,
    get_available_frameworks,
    is_framework_available,
)

__all__ = [
    "BaseFrameworkAdapter",
    "FrameworkAdapter",
    "MLXFrameworkAdapter",
    "get_framework_adapter",
    "register_adapter",
    "get_available_frameworks",
    "is_framework_available",
]