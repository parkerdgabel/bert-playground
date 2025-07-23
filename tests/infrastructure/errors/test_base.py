"""Tests for base error classes."""

import pytest
from datetime import datetime

from infrastructure.errors import ErrorContext, ErrorGroup, KBertError


class TestErrorContext:
    """Test ErrorContext functionality."""

    def test_create_empty_context(self):
        """Test creating empty context."""
        context = ErrorContext()
        assert isinstance(context.timestamp, datetime)
        assert context.module is None
        assert context.technical_details == {}
        assert context.suggestions == []

    def test_from_exception(self):
        """Test creating context from exception."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContext.from_exception(e)
            assert context.stack_trace
            assert len(context.stack_trace) > 0

    def test_add_methods(self):
        """Test adding information to context."""
        context = ErrorContext()
        
        context.add_suggestion("Try this")
        assert "Try this" in context.suggestions
        
        context.add_recovery_action("Do that")
        assert "Do that" in context.recovery_actions
        
        context.add_technical_detail("key", "value")
        assert context.technical_details["key"] == "value"


class TestKBertError:
    """Test KBertError base class."""

    def test_basic_error(self):
        """Test creating basic error."""
        error = KBertError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.recoverable is True
        assert error.error_code == "KBERT"

    def test_error_with_context(self):
        """Test error with custom context."""
        context = ErrorContext()
        context.add_suggestion("Fix it")
        
        error = KBertError("Test error", context=context)
        assert error.context.suggestions == ["Fix it"]

    def test_error_code_generation(self):
        """Test automatic error code generation."""
        class CustomError(KBertError):
            pass
        
        error = CustomError("Test")
        assert error.error_code == "CUSTOM"

    def test_from_exception(self):
        """Test creating from another exception."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = KBertError.from_exception(e, "Wrapped error")
            assert error.message == "Wrapped error"
            assert error.cause == e
            assert len(error.context.related_errors) == 1

    def test_fluent_interface(self):
        """Test fluent interface for adding context."""
        error = (KBertError("Test")
                .with_context(key="value")
                .with_suggestion("Try this")
                .with_recovery("Do that"))
        
        assert error.context.technical_details["key"] == "value"
        assert "Try this" in error.context.suggestions
        assert "Do that" in error.context.recovery_actions

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = KBertError("Test error", error_code="TEST_CODE")
        data = error.to_dict()
        
        assert data["error_type"] == "KBertError"
        assert data["error_code"] == "TEST_CODE"
        assert data["message"] == "Test error"
        assert data["recoverable"] is True

    def test_format_for_cli(self):
        """Test CLI formatting."""
        error = (KBertError("Test error")
                .with_suggestion("Try this")
                .with_recovery("Do that"))
        
        output = error.format_for_cli(verbose=False)
        assert "Test error" in output
        assert "Try this" in output
        assert "Do that" in output


class TestErrorGroup:
    """Test ErrorGroup functionality."""

    def test_error_group(self):
        """Test grouping multiple errors."""
        errors = [
            ValueError("Error 1"),
            TypeError("Error 2"),
            KBertError("Error 3"),
        ]
        
        group = ErrorGroup("Multiple errors occurred", errors)
        assert len(group) == 3
        assert list(group) == errors
        assert len(group.context.related_errors) == 3