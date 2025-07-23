"""
Compute backend adapters for neural network frameworks.

This package contains implementations of compute backends:
- MLX: Apple's ML framework for Apple Silicon
- PyTorch: Support for PyTorch models (future)
"""

from .mlx import MLXComputeAdapter, MLXNeuralBackend

__all__ = ["MLXComputeAdapter", "MLXNeuralBackend"]