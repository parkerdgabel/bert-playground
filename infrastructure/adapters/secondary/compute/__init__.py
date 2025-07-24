"""
Compute backend adapters for neural network frameworks.

This package contains implementations of compute backends:
- MLX: Apple's ML framework for Apple Silicon
- PyTorch: Support for PyTorch models (future)
"""

from .base import BaseComputeAdapter
from .mlx import (
    MLXComputeAdapter,
    MLXModelAdapter,
    MLXOptimizer,
)

# Keep backward compatibility
from .mlx import MLXComputeAdapter as MLXNeuralBackend

__all__ = [
    "BaseComputeAdapter",
    "MLXComputeAdapter",
    "MLXModelAdapter", 
    "MLXOptimizer",
    "MLXNeuralBackend",  # For backward compatibility
]