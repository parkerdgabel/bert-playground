"""Neural network adapters for different ML frameworks."""

from .mlx_backend import MLXNeuralBackend
from .mlx_adapter import MLXNeuralAdapter

__all__ = ["MLXNeuralBackend", "MLXNeuralAdapter"]