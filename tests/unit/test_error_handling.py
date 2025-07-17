"""Tests for error handling system."""

import json
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.error_handling import (
    ErrorCategory,
    ErrorHandler,
    ErrorHandlingConfig,
    ErrorSeverity,
    TrainingError,
    with_error_handling,
)


class TestTrainingError:
    """Test TrainingError class."""
    
    def test_error_creation(self):
        """Test creating training error."""
        error = TrainingError(
            error_type="ValueError",
            error_message="Invalid value",
            error_category=ErrorCategory.USER,
            severity=ErrorSeverity.ERROR,
            timestamp=time.time(),
            step=100,
            epoch=5,
            traceback="Traceback...",
            context={"batch_size": 32},
            recoverable=True,
            retry_count=1,
        )
        
        assert error.error_type == "ValueError"
        assert error.error_message == "Invalid value"
        assert error.error_category == ErrorCategory.USER
        assert error.severity == ErrorSeverity.ERROR
        assert error.step == 100
        assert error.epoch == 5
        assert error.context["batch_size"] == 32
        assert error.recoverable is True
        assert error.retry_count == 1
    
    def test_error_serialization(self):
        """Test error to_dict conversion."""
        error = TrainingError(
            error_type="RuntimeError",
            error_message="Runtime issue",
            error_category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            timestamp=1234567890.0,
            step=50,
            epoch=2,
        )
        
        data = error.to_dict()
        assert isinstance(data, dict)
        assert data["error_type"] == "RuntimeError"
        assert data["error_message"] == "Runtime issue"
        assert data["error_category"] == "system"
        assert data["severity"] == "critical"
        assert data["timestamp"] == 1234567890.0
        assert data["step"] == 50
        assert data["epoch"] == 2


class TestErrorHandlingConfig:
    """Test ErrorHandlingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ErrorHandlingConfig()
        
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 5.0
        assert config.exponential_backoff is True
        assert config.on_memory_error == "reduce_batch"
        assert config.memory_reduction_factor == 0.5
        assert config.min_batch_size == 1
        assert config.enable_graceful_shutdown is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ErrorHandlingConfig(
            max_retries=5,
            retry_delay_seconds=10.0,
            on_memory_error="checkpoint",
            log_errors_to_file=True,
            error_log_file="./custom_errors.log",
        )
        
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 10.0
        assert config.on_memory_error == "checkpoint"
        assert config.log_errors_to_file is True
        assert config.error_log_file == "./custom_errors.log"


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def error_handler(self, temp_dir):
        """Create error handler for tests."""
        config = ErrorHandlingConfig(
            log_errors_to_file=False,
            save_error_diagnostics=True,
            diagnostics_dir=str(Path(temp_dir) / "diagnostics"),
            enable_graceful_shutdown=False,
        )
        return ErrorHandler(config)
    
    def test_initialization(self, error_handler):
        """Test error handler initialization."""
        assert error_handler.config.max_retries == 3
        assert len(error_handler.error_history) == 0
        assert all(count == 0 for count in error_handler.error_counts.values())
        assert error_handler.interrupted is False
    
    def test_classify_error_memory(self, error_handler):
        """Test classifying memory errors."""
        # Test various memory error patterns
        memory_errors = [
            MemoryError("Out of memory"),
            RuntimeError("CUDA out of memory"),
            Exception("OOM error occurred"),
            Exception("Insufficient memory"),
        ]
        
        for error in memory_errors:
            category = error_handler._classify_error(error)
            assert category == ErrorCategory.MEMORY
    
    def test_classify_error_compute(self, error_handler):
        """Test classifying compute errors."""
        compute_errors = [
            ValueError("Gradient contains NaN"),
            RuntimeError("Loss is inf"),
            Exception("Numerical overflow detected"),
        ]
        
        for error in compute_errors:
            category = error_handler._classify_error(error)
            assert category == ErrorCategory.COMPUTE
    
    def test_classify_error_io(self, error_handler):
        """Test classifying IO errors."""
        io_errors = [
            IOError("File not found"),
            OSError("Permission denied"),
            Exception("Cannot read file"),
        ]
        
        for error in io_errors:
            category = error_handler._classify_error(error)
            assert category == ErrorCategory.IO
    
    def test_classify_error_data(self, error_handler):
        """Test classifying data errors."""
        data_errors = [
            Exception("Dataset is empty"),
            ValueError("Batch size mismatch"),
            RuntimeError("Invalid shape for input"),
        ]
        
        for error in data_errors:
            category = error_handler._classify_error(error)
            assert category == ErrorCategory.DATA
    
    def test_determine_severity(self, error_handler):
        """Test determining error severity."""
        # Fatal errors
        assert error_handler._determine_severity(
            SystemExit(), ErrorCategory.SYSTEM
        ) == ErrorSeverity.FATAL
        
        assert error_handler._determine_severity(
            KeyboardInterrupt(), ErrorCategory.SYSTEM
        ) == ErrorSeverity.FATAL
        
        # Critical errors
        assert error_handler._determine_severity(
            MemoryError(), ErrorCategory.MEMORY
        ) == ErrorSeverity.CRITICAL
        
        # Regular errors
        assert error_handler._determine_severity(
            ValueError(), ErrorCategory.COMPUTE
        ) == ErrorSeverity.ERROR
        
        # Warnings
        assert error_handler._determine_severity(
            Exception(), ErrorCategory.DATA
        ) == ErrorSeverity.WARNING
    
    def test_is_recoverable(self, error_handler):
        """Test determining if error is recoverable."""
        # Non-recoverable
        assert error_handler._is_recoverable(
            SystemExit(), ErrorCategory.SYSTEM
        ) is False
        
        assert error_handler._is_recoverable(
            KeyboardInterrupt(), ErrorCategory.SYSTEM
        ) is False
        
        # Recoverable
        assert error_handler._is_recoverable(
            MemoryError(), ErrorCategory.MEMORY
        ) is True
        
        assert error_handler._is_recoverable(
            ValueError(), ErrorCategory.COMPUTE
        ) is True
    
    def test_handle_error(self, error_handler):
        """Test handling an error."""
        error = ValueError("Test error")
        should_continue, training_error = error_handler.handle_error(
            exception=error,
            step=100,
            epoch=5,
            context={"batch_size": 32},
        )
        
        assert isinstance(training_error, TrainingError)
        assert training_error.error_type == "ValueError"
        assert training_error.error_message == "Test error"
        assert training_error.step == 100
        assert training_error.epoch == 5
        assert len(error_handler.error_history) == 1
        assert error_handler.error_counts[training_error.error_category] == 1
    
    def test_handle_memory_error_reduce_batch(self, error_handler):
        """Test handling memory error with batch reduction."""
        error_handler.config.on_memory_error = "reduce_batch"
        
        context = {"batch_size": 64}
        should_continue, _ = error_handler.handle_error(
            exception=MemoryError("Out of memory"),
            context=context,
        )
        
        # Should reduce batch size
        assert context["batch_size"] == 32
        assert should_continue is True
    
    def test_handle_memory_error_min_batch_size(self, error_handler):
        """Test memory error handling at minimum batch size."""
        error_handler.config.on_memory_error = "reduce_batch"
        error_handler.config.min_batch_size = 4
        
        context = {"batch_size": 4}
        should_continue, _ = error_handler.handle_error(
            exception=MemoryError("Out of memory"),
            context=context,
        )
        
        # Should not reduce below minimum
        assert context["batch_size"] == 4
        assert should_continue is False
    
    def test_retry_delay(self, error_handler):
        """Test retry delay calculation."""
        # Without exponential backoff
        error_handler.config.exponential_backoff = False
        assert error_handler._get_retry_delay(0) == 5.0
        assert error_handler._get_retry_delay(1) == 5.0
        assert error_handler._get_retry_delay(2) == 5.0
        
        # With exponential backoff
        error_handler.config.exponential_backoff = True
        assert error_handler._get_retry_delay(0) == 5.0
        assert error_handler._get_retry_delay(1) == 10.0
        assert error_handler._get_retry_delay(2) == 20.0
    
    def test_recovery_callbacks(self, error_handler):
        """Test recovery callback mechanism."""
        callback_called = False
        
        def recovery_callback(error, context):
            nonlocal callback_called
            callback_called = True
            return True
        
        error_handler.register_recovery_callback(
            ErrorCategory.COMPUTE, recovery_callback
        )
        
        should_continue, _ = error_handler.handle_error(
            exception=ValueError("NaN in loss"),
            context={},
        )
        
        assert callback_called is True
        assert should_continue is True
    
    def test_shutdown_callbacks(self, error_handler):
        """Test shutdown callback mechanism."""
        callback_called = False
        
        def shutdown_callback():
            nonlocal callback_called
            callback_called = True
        
        error_handler.register_shutdown_callback(shutdown_callback)
        
        # Simulate signal handler
        error_handler.interrupted = True
        for callback in error_handler.shutdown_callbacks:
            callback()
        
        assert callback_called is True
    
    def test_error_diagnostics(self, error_handler, temp_dir):
        """Test saving error diagnostics."""
        error_handler.config.save_error_diagnostics = True
        
        _, training_error = error_handler.handle_error(
            exception=RuntimeError("Critical error"),
            step=100,
            epoch=5,
        )
        
        # Check diagnostics file was created
        diagnostics_dir = Path(temp_dir) / "diagnostics"
        assert diagnostics_dir.exists()
        
        diagnostic_files = list(diagnostics_dir.glob("error_*.json"))
        assert len(diagnostic_files) == 1
        
        # Check file content
        with open(diagnostic_files[0]) as f:
            data = json.load(f)
        
        assert data["error_type"] == "RuntimeError"
        assert data["error_message"] == "Critical error"
        assert data["step"] == 100
        assert data["epoch"] == 5
    
    def test_get_error_summary(self, error_handler):
        """Test getting error summary."""
        # Add some errors
        for i in range(5):
            error_handler.handle_error(
                exception=ValueError(f"Error {i}"),
                step=i * 100,
            )
        
        summary = error_handler.get_error_summary()
        
        assert summary["total_errors"] == 5
        assert summary["recoverable_errors"] == 5
        assert len(summary["recent_errors"]) == 5
        assert "error_counts_by_category" in summary
        assert "error_counts_by_severity" in summary


class TestErrorHandlingDecorator:
    """Test with_error_handling decorator."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for tests."""
        config = ErrorHandlingConfig(
            max_retries=2,
            retry_delay_seconds=0.1,
            exponential_backoff=False,
        )
        return ErrorHandler(config)
    
    def test_successful_function(self, error_handler):
        """Test decorator with successful function."""
        @with_error_handling(error_handler)
        def successful_func(x, y):
            return x + y
        
        result = successful_func(2, 3)
        assert result == 5
        assert len(error_handler.error_history) == 0
    
    def test_function_with_retries(self, error_handler):
        """Test decorator with function that fails then succeeds."""
        call_count = 0
        
        @with_error_handling(error_handler)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 2
    
    def test_function_exceeds_retries(self, error_handler):
        """Test decorator when function exceeds max retries."""
        @with_error_handling(error_handler)
        def always_fails():
            raise RuntimeError("Permanent error")
        
        with pytest.raises(RuntimeError, match="Permanent error"):
            always_fails()
        
        assert len(error_handler.error_history) >= 1
    
    def test_decorator_with_category_override(self, error_handler):
        """Test decorator with category override."""
        @with_error_handling(error_handler, category=ErrorCategory.MODEL)
        def model_func():
            raise ValueError("Model error")
        
        try:
            model_func()
        except ValueError:
            pass
        
        assert error_handler.error_history[-1].error_category == ErrorCategory.MODEL
    
    def test_decorator_with_retries_override(self, error_handler):
        """Test decorator with retries override."""
        call_count = 0
        
        @with_error_handling(error_handler, retries=1)
        def limited_retries():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            limited_retries()
        
        # Should only try twice (initial + 1 retry)
        assert call_count == 2
    
    def test_decorator_preserves_function_info(self, error_handler):
        """Test that decorator preserves function metadata."""
        @with_error_handling(error_handler)
        def documented_func():
            """This is a documented function."""
            return 42
        
        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."