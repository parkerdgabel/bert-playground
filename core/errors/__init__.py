"""Comprehensive error handling framework for k-bert.

This module provides:
- Rich error types with context and recovery suggestions
- Error handler registry for custom error handling
- Recovery strategies for automatic error recovery
- Integration with logging and CLI output
"""

# Base error classes
from .base import ErrorContext, ErrorGroup, KBertError

# Error handlers
from .handlers import (
    CLIErrorHandler,
    ErrorHandler,
    ErrorHandlerRegistry,
    ErrorHandlingContext,
    RecoverableErrorHandler,
    RecoveryStrategies,
    handle_error,
    register_handler,
    register_type_handler,
    setup_default_handlers,
    with_error_handling,
)

# Recovery strategies
from .recovery import (
    CheckpointRecoveryStrategy,
    CompositeStrategy,
    FallbackStrategy,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    ResourceReductionStrategy,
    RetryStrategy,
    attempt_recovery,
    register_recovery_strategy,
    setup_default_recovery,
    with_recovery,
)

# Specific error types
from .types import (
    CLIError,
    ConfigurationError,
    DataError,
    ModelError,
    PluginError,
    TrainingError,
    ValidationError,
)

__all__ = [
    # Base classes
    "KBertError",
    "ErrorContext",
    "ErrorGroup",
    # Specific error types
    "ConfigurationError",
    "ModelError",
    "DataError",
    "TrainingError",
    "ValidationError",
    "PluginError",
    "CLIError",
    # Handlers
    "ErrorHandler",
    "ErrorHandlerRegistry",
    "CLIErrorHandler",
    "RecoverableErrorHandler",
    "ErrorHandlingContext",
    "RecoveryStrategies",
    "handle_error",
    "register_handler",
    "register_type_handler",
    "setup_default_handlers",
    "with_error_handling",
    # Recovery
    "RecoveryStrategy",
    "RecoveryResult",
    "RecoveryManager",
    "RetryStrategy",
    "ResourceReductionStrategy",
    "CheckpointRecoveryStrategy",
    "FallbackStrategy",
    "CompositeStrategy",
    "attempt_recovery",
    "register_recovery_strategy",
    "setup_default_recovery",
    "with_recovery",
]


# Initialize default handlers and recovery on import
def _initialize_error_system():
    """Initialize the error handling system with defaults."""
    setup_default_handlers()
    setup_default_recovery()


_initialize_error_system()