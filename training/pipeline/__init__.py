"""Training pipeline with middleware support.

This module provides a flexible pipeline architecture for training,
allowing middleware to modify behavior without changing core logic.
"""

from .base import Pipeline, Middleware, PipelineBuilder
from .middleware import (
    TimingMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    ValidationMiddleware,
)
from .executor import PipelineExecutor

__all__ = [
    "Pipeline",
    "Middleware",
    "PipelineBuilder",
    "PipelineExecutor",
    "TimingMiddleware",
    "ErrorHandlingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "ValidationMiddleware",
]