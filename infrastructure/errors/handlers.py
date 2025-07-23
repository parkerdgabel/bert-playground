"""Error handler registry and implementations."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .base import ErrorContext, KBertError
from .types import (
    CLIError,
    ConfigurationError,
    DataError,
    ModelError,
    PluginError,
    TrainingError,
    ValidationError,
)

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
HandlerFunc = Callable[[E], Optional[Any]]


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can handle the given error."""
        pass

    @abstractmethod
    def handle(self, error: Exception) -> Optional[Any]:
        """Handle the error and optionally return a result."""
        pass


class BaseErrorHandler(ErrorHandler):
    """Base implementation for type-based error handlers."""

    def __init__(self, error_type: Type[Exception]):
        """Initialize handler for specific error type."""
        self.error_type = error_type

    def can_handle(self, error: Exception) -> bool:
        """Check if error is of handled type."""
        return isinstance(error, self.error_type)


class CLIErrorHandler(BaseErrorHandler):
    """Handler for CLI errors with rich output."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize CLI error handler."""
        super().__init__(CLIError)
        self.console = console or Console(stderr=True)

    def handle(self, error: Exception) -> Optional[Any]:
        """Handle CLI error with formatted output."""
        if not isinstance(error, CLIError):
            return None

        # Create error panel
        error_text = Text(error.message, style="red bold")
        panel = Panel(
            error_text,
            title=f"[red]Error: {error.error_code}[/red]",
            border_style="red",
            expand=False,
        )
        self.console.print(panel)

        # Print suggestions
        if error.context.suggestions:
            self.console.print("\n[yellow]Suggestions:[/yellow]")
            for suggestion in error.context.suggestions:
                self.console.print(f"  • {suggestion}")

        # Exit with error code
        sys.exit(error.exit_code)


class RecoverableErrorHandler(BaseErrorHandler):
    """Handler that attempts recovery for recoverable errors."""

    def __init__(
        self,
        error_type: Type[KBertError],
        recovery_strategies: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize recoverable error handler."""
        super().__init__(error_type)
        self.recovery_strategies = recovery_strategies or {}

    def handle(self, error: Exception) -> Optional[Any]:
        """Attempt to recover from the error."""
        if not isinstance(error, KBertError) or not error.recoverable:
            return None

        # Try recovery actions
        for action in error.context.recovery_actions:
            if action in self.recovery_strategies:
                try:
                    logger.info(f"Attempting recovery: {action}")
                    result = self.recovery_strategies[action](error)
                    if result is not None:
                        logger.success(f"Recovery successful: {action}")
                        return result
                except Exception as e:
                    logger.warning(f"Recovery failed: {action} - {e}")

        return None


class ErrorHandlerRegistry:
    """Registry for error handlers with priority support."""

    def __init__(self):
        """Initialize error handler registry."""
        self.handlers: List[tuple[int, ErrorHandler]] = []
        self.type_handlers: Dict[Type[Exception], List[HandlerFunc]] = {}
        self.default_handler: Optional[ErrorHandler] = None

    def register_handler(
        self,
        handler: ErrorHandler,
        priority: int = 0,
    ) -> None:
        """Register an error handler with priority."""
        self.handlers.append((priority, handler))
        self.handlers.sort(key=lambda x: x[0], reverse=True)

    def register_type_handler(
        self,
        error_type: Type[E],
        handler: HandlerFunc[E],
    ) -> None:
        """Register a handler function for specific error type."""
        if error_type not in self.type_handlers:
            self.type_handlers[error_type] = []
        self.type_handlers[error_type].append(handler)

    def set_default_handler(self, handler: ErrorHandler) -> None:
        """Set default handler for unhandled errors."""
        self.default_handler = handler

    def handle_error(self, error: Exception) -> Optional[Any]:
        """Handle an error using registered handlers."""
        # Try specific type handlers first
        for error_type, handlers in self.type_handlers.items():
            if isinstance(error, error_type):
                for handler_func in handlers:
                    result = handler_func(error)
                    if result is not None:
                        return result

        # Try registered handlers by priority
        for _, handler in self.handlers:
            if handler.can_handle(error):
                result = handler.handle(error)
                if result is not None:
                    return result

        # Use default handler if available
        if self.default_handler:
            return self.default_handler.handle(error)

        # Re-raise if no handler found
        raise error


# Global registry instance
_registry = ErrorHandlerRegistry()


def register_handler(handler: ErrorHandler, priority: int = 0) -> None:
    """Register an error handler."""
    _registry.register_handler(handler, priority)


def register_type_handler(
    error_type: Type[E],
) -> Callable[[HandlerFunc[E]], HandlerFunc[E]]:
    """Decorator to register a handler for specific error type."""
    def decorator(func: HandlerFunc[E]) -> HandlerFunc[E]:
        _registry.register_type_handler(error_type, func)
        return func
    return decorator


def handle_error(error: Exception) -> Optional[Any]:
    """Handle an error using registered handlers."""
    return _registry.handle_error(error)


def with_error_handling(
    recoverable: bool = True,
    error_type: Type[KBertError] = KBertError,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add error handling to functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except KBertError:
                # Re-raise k-bert errors as-is
                raise
            except Exception as e:
                # Wrap other exceptions
                message = error_message or f"Error in {func.__name__}: {str(e)}"
                wrapped_error = error_type.from_exception(
                    e,
                    message=message,
                    recoverable=recoverable,
                )
                raise wrapped_error from e
        return wrapper
    return decorator


# Recovery strategies
class RecoveryStrategies:
    """Common recovery strategies."""

    @staticmethod
    def retry_with_backoff(
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
    ) -> Callable[[Exception], Optional[Any]]:
        """Create a retry strategy with exponential backoff."""
        def strategy(error: Exception) -> Optional[Any]:
            import time
            
            for attempt in range(max_attempts):
                wait_time = backoff_factor ** attempt
                logger.info(f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait_time)
                
                # Return None to indicate retry should happen
                # Actual retry logic should be implemented by caller
                if attempt < max_attempts - 1:
                    return None
            
            # All attempts failed
            return None
        
        return strategy

    @staticmethod
    def fallback_to_default(default_value: Any) -> Callable[[Exception], Any]:
        """Create a fallback strategy that returns a default value."""
        def strategy(error: Exception) -> Any:
            logger.warning(f"Using default value due to error: {error}")
            return default_value
        return strategy

    @staticmethod
    def reduce_resource_usage(
        resource_type: str = "memory",
        reduction_factor: float = 0.5,
    ) -> Callable[[Exception], Dict[str, Any]]:
        """Create a strategy to reduce resource usage."""
        def strategy(error: Exception) -> Dict[str, Any]:
            if resource_type == "memory":
                return {
                    "batch_size_reduction": reduction_factor,
                    "gradient_accumulation": True,
                }
            elif resource_type == "compute":
                return {
                    "num_workers_reduction": reduction_factor,
                    "prefetch_reduction": reduction_factor,
                }
            return {}
        return strategy


# Default handlers for common error types
def setup_default_handlers(console: Optional[Console] = None) -> None:
    """Set up default error handlers."""
    # CLI errors
    register_handler(CLIErrorHandler(console), priority=100)
    
    # Configuration errors
    @register_type_handler(ConfigurationError)
    def handle_config_error(error: ConfigurationError) -> Optional[Any]:
        if error.error_code == "CONFIG_MISSING_FIELD":
            # Could return a default value or prompt user
            return None
        return None
    
    # Training errors with recovery
    training_handler = RecoverableErrorHandler(
        TrainingError,
        recovery_strategies={
            "Resume from last checkpoint with lower learning rate": 
                RecoveryStrategies.reduce_resource_usage("compute"),
            "Retry with halved batch size": 
                RecoveryStrategies.reduce_resource_usage("memory"),
        }
    )
    register_handler(training_handler, priority=50)
    
    # Data errors
    @register_type_handler(DataError)
    def handle_data_error(error: DataError) -> Optional[Any]:
        if error.error_code == "DATA_FILE_NOT_FOUND":
            # Could prompt for alternative path
            return None
        return None


# Context manager for temporary error handling
class ErrorHandlingContext:
    """Context manager for temporary error handling configuration."""

    def __init__(
        self,
        handlers: Optional[List[ErrorHandler]] = None,
        suppress: Optional[List[Type[Exception]]] = None,
        transform: Optional[Dict[Type[Exception], Type[Exception]]] = None,
    ):
        """Initialize error handling context."""
        self.handlers = handlers or []
        self.suppress = suppress or []
        self.transform = transform or {}
        self._original_handlers = []

    def __enter__(self) -> "ErrorHandlingContext":
        """Enter context and register handlers."""
        for handler in self.handlers:
            register_handler(handler, priority=1000)  # High priority for context handlers
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle exceptions."""
        # Remove context handlers
        # Note: In production, we'd need to track and remove specific handlers
        
        if exc_type is None:
            return False
        
        # Check if we should suppress
        if any(issubclass(exc_type, suppressed) for suppressed in self.suppress):
            logger.debug(f"Suppressing {exc_type.__name__}: {exc_val}")
            return True
        
        # Check if we should transform
        for source_type, target_type in self.transform.items():
            if issubclass(exc_type, source_type):
                logger.debug(f"Transforming {exc_type.__name__} to {target_type.__name__}")
                raise target_type(str(exc_val)) from exc_val
        
        return False