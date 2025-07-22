"""CLI Middleware System.

Provides a flexible middleware pipeline for command execution with support for:
- Logging enhancement
- Validation
- Error handling
- Performance monitoring
- Custom middleware extensions
"""

from cli.middleware.base import (
    Middleware,
    MiddlewarePipeline,
    CommandContext,
    MiddlewareResult,
)
from cli.middleware.logging import LoggingMiddleware
from cli.middleware.validation import ValidationMiddleware
from cli.middleware.error import ErrorMiddleware
from cli.middleware.monitoring import PerformanceMiddleware

__all__ = [
    "Middleware",
    "MiddlewarePipeline",
    "CommandContext",
    "MiddlewareResult",
    "LoggingMiddleware",
    "ValidationMiddleware",
    "ErrorMiddleware",
    "PerformanceMiddleware",
]