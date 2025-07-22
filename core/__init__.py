"""Core infrastructure modules for k-bert."""

# Error handling framework
from .errors import (
    # Base classes
    KBertError,
    ErrorContext,
    ErrorGroup,
    # Specific error types
    ConfigurationError,
    ModelError,
    DataError,
    TrainingError,
    ValidationError,
    PluginError,
    CLIError,
    # Handlers
    handle_error,
    register_handler,
    register_type_handler,
    setup_default_handlers,
    with_error_handling,
    # Recovery
    attempt_recovery,
    register_recovery_strategy,
    setup_default_recovery,
    with_recovery,
)

__all__ = [
    # Error types
    "KBertError",
    "ErrorContext", 
    "ErrorGroup",
    "ConfigurationError",
    "ModelError",
    "DataError",
    "TrainingError",
    "ValidationError",
    "PluginError",
    "CLIError",
    # Error handling
    "handle_error",
    "register_handler",
    "register_type_handler",
    "setup_default_handlers",
    "with_error_handling",
    # Recovery
    "attempt_recovery",
    "register_recovery_strategy",
    "setup_default_recovery",
    "with_recovery",
]