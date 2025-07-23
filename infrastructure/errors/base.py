"""Base error classes with rich context for k-bert."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel, Field


class ErrorContext(BaseModel):
    """Rich context information for errors."""

    timestamp: datetime = Field(default_factory=datetime.now)
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    user_message: Optional[str] = None
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    stack_trace: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    recovery_actions: List[str] = Field(default_factory=list)
    related_errors: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_exception(cls, exc: Exception) -> "ErrorContext":
        """Create context from an exception."""
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last_frame = tb[-1]
            return cls(
                module=last_frame.name,
                function=last_frame.name,
                line_number=last_frame.lineno,
                file_path=str(last_frame.filename),
                stack_trace=traceback.format_tb(exc.__traceback__),
            )
        return cls(stack_trace=traceback.format_tb(exc.__traceback__))

    def add_suggestion(self, suggestion: str) -> None:
        """Add a helpful suggestion for resolving the error."""
        self.suggestions.append(suggestion)

    def add_recovery_action(self, action: str) -> None:
        """Add a recovery action that could resolve the error."""
        self.recovery_actions.append(action)

    def add_technical_detail(self, key: str, value: Any) -> None:
        """Add technical debugging information."""
        self.technical_details[key] = value

    def add_related_error(self, error: Exception) -> None:
        """Add a related error for context."""
        self.related_errors.append({
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception_only(type(error), error),
        })


T = TypeVar("T", bound="KBertError")


class KBertError(Exception):
    """Base exception class for k-bert with rich context support."""

    def __init__(
        self,
        message: str,
        *,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[str] = None,
        recoverable: bool = True,
    ):
        """Initialize k-bert error with context.

        Args:
            message: Human-readable error message
            context: Rich error context
            cause: Original exception that caused this error
            error_code: Unique error code for programmatic handling
            recoverable: Whether this error can be recovered from
        """
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext.from_exception(self)
        self.cause = cause
        self.error_code = error_code or self._generate_error_code()
        self.recoverable = recoverable
        
        # Enhance context with error information
        self.context.user_message = message
        if cause:
            self.context.add_related_error(cause)
        
        # Log the error
        self._log_error()

    def _generate_error_code(self) -> str:
        """Generate a unique error code based on error type."""
        class_name = self.__class__.__name__
        # Convert CamelCase to UPPER_SNAKE_CASE
        code = ""
        for i, char in enumerate(class_name):
            if i > 0 and char.isupper() and class_name[i-1].islower():
                code += "_"
            code += char.upper()
        return code.replace("_ERROR", "")

    def _log_error(self) -> None:
        """Log the error with appropriate severity."""
        log_data = {
            "error_code": self.error_code,
            "recoverable": self.recoverable,
            "module": self.context.module,
            "function": self.context.function,
            "line": self.context.line_number,
            "file": self.context.file_path,
        }
        
        if self.context.technical_details:
            log_data["details"] = self.context.technical_details
        
        if self.recoverable:
            logger.error(f"{self.message}", **log_data)
        else:
            logger.critical(f"{self.message}", **log_data)

    @classmethod
    def from_exception(
        cls: Type[T],
        exc: Exception,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Create error from another exception."""
        error_message = message or str(exc)
        context = ErrorContext.from_exception(exc)
        return cls(error_message, context=context, cause=exc, **kwargs)

    def with_context(self: T, **kwargs: Any) -> T:
        """Add context information to the error."""
        for key, value in kwargs.items():
            self.context.add_technical_detail(key, value)
        return self

    def with_suggestion(self: T, suggestion: str) -> T:
        """Add a suggestion for resolving the error."""
        self.context.add_suggestion(suggestion)
        return self

    def with_recovery(self: T, action: str) -> T:
        """Add a recovery action."""
        self.context.add_recovery_action(action)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context.model_dump(mode="json"),
            "cause": str(self.cause) if self.cause else None,
        }

    def format_for_cli(self, verbose: bool = False) -> str:
        """Format error for CLI output."""
        lines = [
            f"[red]Error[/red]: {self.message}",
            f"[dim]Code: {self.error_code}[/dim]",
        ]
        
        if self.context.suggestions:
            lines.append("\n[yellow]Suggestions:[/yellow]")
            for suggestion in self.context.suggestions:
                lines.append(f"  • {suggestion}")
        
        if self.context.recovery_actions:
            lines.append("\n[green]Recovery Actions:[/green]")
            for action in self.context.recovery_actions:
                lines.append(f"  • {action}")
        
        if verbose and self.context.technical_details:
            lines.append("\n[dim]Technical Details:[/dim]")
            for key, value in self.context.technical_details.items():
                lines.append(f"  {key}: {value}")
        
        if verbose and self.context.stack_trace:
            lines.append("\n[dim]Stack Trace:[/dim]")
            lines.extend(self.context.stack_trace)
        
        return "\n".join(lines)


class ErrorGroup(KBertError):
    """Group multiple errors together."""

    def __init__(self, message: str, errors: List[Exception], **kwargs: Any):
        """Initialize error group."""
        super().__init__(message, **kwargs)
        self.errors = errors
        
        # Add all errors to context
        for error in errors:
            self.context.add_related_error(error)

    def __iter__(self):
        """Iterate over grouped errors."""
        return iter(self.errors)

    def __len__(self):
        """Get number of errors in group."""
        return len(self.errors)