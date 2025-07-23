"""Domain-specific exceptions."""

from .base import DomainException, ValidationException
from .training import (
    TrainingError,
    ModelNotInitializedError,
    CheckpointError,
    InvalidConfigurationError,
    DataError,
    MetricsError,
    EarlyStoppingError,
)

__all__ = [
    # Base exceptions
    "DomainException",
    "ValidationException",
    
    # Training exceptions
    "TrainingError",
    "ModelNotInitializedError",
    "CheckpointError",
    "InvalidConfigurationError",
    "DataError",
    "MetricsError",
    "EarlyStoppingError",
]