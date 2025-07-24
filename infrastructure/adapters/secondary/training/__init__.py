"""Training execution adapters.

This module contains infrastructure adapters for training execution,
implementing the TrainingExecutor port with framework-specific code.
"""

from .mlx_training_executor import MLXTrainingExecutor

__all__ = [
    "MLXTrainingExecutor",
]