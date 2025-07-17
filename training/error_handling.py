"""Comprehensive error handling system for production MLX training.

This module provides robust error handling, recovery mechanisms, and
diagnostic tools for production training runs.
"""

import functools
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""
    
    MEMORY = "memory"
    COMPUTE = "compute"
    IO = "io"
    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    CHECKPOINT = "checkpoint"
    SYSTEM = "system"
    USER = "user"
    UNKNOWN = "unknown"


@dataclass
class TrainingError:
    """Represents a training error with context."""
    
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    step: Optional[int] = None
    epoch: Optional[int] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "epoch": self.epoch,
            "traceback": self.traceback,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
        }


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""
    
    # Recovery settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    exponential_backoff: bool = True
    
    # Memory error handling
    on_memory_error: str = "reduce_batch"  # reduce_batch, checkpoint, fail
    memory_reduction_factor: float = 0.5
    min_batch_size: int = 1
    
    # Checkpoint error handling
    on_checkpoint_error: str = "continue"  # continue, retry, fail
    checkpoint_retry_delay: float = 10.0
    
    # System error handling
    on_system_error: str = "retry"  # retry, checkpoint_and_exit, fail
    system_health_check_interval: int = 100  # steps
    
    # Error logging
    log_errors_to_file: bool = True
    error_log_file: str = "./errors.log"
    save_error_diagnostics: bool = True
    diagnostics_dir: str = "./error_diagnostics"
    
    # Graceful shutdown
    enable_graceful_shutdown: bool = True
    shutdown_timeout_seconds: float = 60.0
    save_on_interrupt: bool = True


class ErrorHandler:
    """Comprehensive error handler for training."""
    
    def __init__(self, config: ErrorHandlingConfig):
        """Initialize error handler.
        
        Args:
            config: Error handling configuration
        """
        self.config = config
        self.error_history: List[TrainingError] = []
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.recovery_callbacks: Dict[ErrorCategory, List[Callable]] = {
            cat: [] for cat in ErrorCategory
        }
        
        # Setup error logging
        if config.log_errors_to_file:
            self._setup_error_logging()
        
        # Setup diagnostics directory
        if config.save_error_diagnostics:
            self.diagnostics_dir = Path(config.diagnostics_dir)
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        # Signal handling for graceful shutdown
        if config.enable_graceful_shutdown:
            self._setup_signal_handlers()
        
        self.interrupted = False
        self.shutdown_callbacks: List[Callable] = []
        
        logger.info(
            f"Error Handler initialized:\n"
            f"  Max retries: {config.max_retries}\n"
            f"  Memory error strategy: {config.on_memory_error}\n"
            f"  Graceful shutdown: {config.enable_graceful_shutdown}"
        )
    
    def _setup_error_logging(self) -> None:
        """Setup dedicated error logging."""
        error_logger = logger.bind(name="error_handler")
        error_logger.add(
            self.config.error_log_file,
            format="{time} | {level} | {message}",
            level="ERROR",
            rotation="10 MB",
        )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self.interrupted = True
            
            # Execute shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Shutdown callback failed: {e}")
        
        # Register handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_recovery_callback(
        self, category: ErrorCategory, callback: Callable
    ) -> None:
        """Register a recovery callback for an error category.
        
        Args:
            category: Error category
            callback: Recovery callback function
        """
        self.recovery_callbacks[category].append(callback)
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a shutdown callback.
        
        Args:
            callback: Shutdown callback function
        """
        self.shutdown_callbacks.append(callback)
    
    def handle_error(
        self,
        exception: Exception,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[TrainingError]]:
        """Handle a training error.
        
        Args:
            exception: The exception that occurred
            step: Current training step
            epoch: Current epoch
            context: Additional context
            
        Returns:
            (should_continue, error_record)
        """
        # Classify the error
        category = self._classify_error(exception)
        severity = self._determine_severity(exception, category)
        
        # Create error record
        error = TrainingError(
            error_type=type(exception).__name__,
            error_message=str(exception),
            error_category=category,
            severity=severity,
            timestamp=time.time(),
            step=step,
            epoch=epoch,
            traceback=traceback.format_exc(),
            context=context or {},
            recoverable=self._is_recoverable(exception, category),
        )
        
        # Log the error
        self._log_error(error)
        
        # Save to history
        self.error_history.append(error)
        self.error_counts[category] += 1
        
        # Save diagnostics if configured
        if self.config.save_error_diagnostics:
            self._save_error_diagnostics(error)
        
        # Determine if we should continue
        should_continue = False
        
        if error.recoverable:
            # Try recovery callbacks
            for callback in self.recovery_callbacks[category]:
                try:
                    if callback(error, context):
                        should_continue = True
                        break
                except Exception as e:
                    logger.error(f"Recovery callback failed: {e}")
            
            # Apply category-specific handling
            if not should_continue:
                should_continue = self._apply_error_handling(error, context)
        
        return should_continue, error
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            exception: The exception
            
        Returns:
            Error category
        """
        error_type = type(exception).__name__
        error_msg = str(exception).lower()
        
        # Memory errors
        if "memory" in error_msg or "oom" in error_msg or error_type == "MemoryError":
            return ErrorCategory.MEMORY
        
        # Compute errors
        elif "nan" in error_msg or "inf" in error_msg or "overflow" in error_msg:
            return ErrorCategory.COMPUTE
        
        # IO errors
        elif error_type in ["IOError", "OSError"] or "file" in error_msg:
            return ErrorCategory.IO
        
        # Network errors
        elif "connection" in error_msg or "timeout" in error_msg:
            return ErrorCategory.NETWORK
        
        # Data errors
        elif "dataset" in error_msg or "batch" in error_msg or "shape" in error_msg:
            return ErrorCategory.DATA
        
        # Model errors
        elif "model" in error_msg or "layer" in error_msg or "parameter" in error_msg:
            return ErrorCategory.MODEL
        
        # Optimizer errors
        elif "optimizer" in error_msg or "gradient" in error_msg:
            return ErrorCategory.OPTIMIZER
        
        # Checkpoint errors
        elif "checkpoint" in error_msg or "save" in error_msg or "load" in error_msg:
            return ErrorCategory.CHECKPOINT
        
        # System errors
        elif error_type in ["SystemError", "RuntimeError"]:
            return ErrorCategory.SYSTEM
        
        # User errors
        elif error_type in ["ValueError", "TypeError", "KeyError"]:
            return ErrorCategory.USER
        
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity(
        self, exception: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity.
        
        Args:
            exception: The exception
            category: Error category
            
        Returns:
            Error severity
        """
        # Fatal errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.FATAL
        
        # Critical errors
        if category in [ErrorCategory.MEMORY, ErrorCategory.SYSTEM]:
            return ErrorSeverity.CRITICAL
        
        # Regular errors
        if category in [ErrorCategory.COMPUTE, ErrorCategory.MODEL]:
            return ErrorSeverity.ERROR
        
        # Warnings
        if category in [ErrorCategory.DATA, ErrorCategory.CHECKPOINT]:
            return ErrorSeverity.WARNING
        
        return ErrorSeverity.ERROR
    
    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable.
        
        Args:
            exception: The exception
            category: Error category
            
        Returns:
            True if recoverable
        """
        # Non-recoverable errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return False
        
        # Category-based determination
        if category in [ErrorCategory.MEMORY, ErrorCategory.COMPUTE]:
            return True
        
        if category == ErrorCategory.SYSTEM:
            return "fatal" not in str(exception).lower()
        
        return True
    
    def _log_error(self, error: TrainingError) -> None:
        """Log an error with appropriate level.
        
        Args:
            error: Training error
        """
        log_message = (
            f"Training Error: {error.error_type}\n"
            f"  Category: {error.error_category.value}\n"
            f"  Message: {error.error_message}\n"
            f"  Step: {error.step}, Epoch: {error.epoch}\n"
            f"  Recoverable: {error.recoverable}"
        )
        
        if error.severity == ErrorSeverity.FATAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.CRITICAL:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _apply_error_handling(
        self, error: TrainingError, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply category-specific error handling.
        
        Args:
            error: Training error
            context: Additional context
            
        Returns:
            True if training should continue
        """
        if error.error_category == ErrorCategory.MEMORY:
            return self._handle_memory_error(error, context)
        elif error.error_category == ErrorCategory.CHECKPOINT:
            return self._handle_checkpoint_error(error, context)
        elif error.error_category == ErrorCategory.SYSTEM:
            return self._handle_system_error(error, context)
        else:
            # Default retry logic
            if error.retry_count < self.config.max_retries:
                delay = self._get_retry_delay(error.retry_count)
                logger.info(f"Retrying after {delay:.1f} seconds...")
                time.sleep(delay)
                return True
            return False
    
    def _handle_memory_error(
        self, error: TrainingError, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Handle memory errors.
        
        Args:
            error: Training error
            context: Additional context
            
        Returns:
            True if training should continue
        """
        if self.config.on_memory_error == "reduce_batch":
            if context and "batch_size" in context:
                current_batch_size = context["batch_size"]
                new_batch_size = max(
                    int(current_batch_size * self.config.memory_reduction_factor),
                    self.config.min_batch_size,
                )
                
                if new_batch_size < current_batch_size:
                    logger.info(
                        f"Reducing batch size: {current_batch_size} -> {new_batch_size}"
                    )
                    context["batch_size"] = new_batch_size
                    return True
        
        elif self.config.on_memory_error == "checkpoint":
            logger.info("Saving checkpoint before memory error recovery...")
            # Checkpoint saving would be handled by caller
            return True
        
        return False
    
    def _handle_checkpoint_error(
        self, error: TrainingError, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Handle checkpoint errors.
        
        Args:
            error: Training error
            context: Additional context
            
        Returns:
            True if training should continue
        """
        if self.config.on_checkpoint_error == "continue":
            logger.warning("Continuing training despite checkpoint error")
            return True
        elif self.config.on_checkpoint_error == "retry":
            if error.retry_count < self.config.max_retries:
                time.sleep(self.config.checkpoint_retry_delay)
                return True
        
        return False
    
    def _handle_system_error(
        self, error: TrainingError, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Handle system errors.
        
        Args:
            error: Training error
            context: Additional context
            
        Returns:
            True if training should continue
        """
        if self.config.on_system_error == "retry":
            if error.retry_count < self.config.max_retries:
                delay = self._get_retry_delay(error.retry_count)
                logger.info(f"System error recovery, retrying after {delay:.1f}s...")
                time.sleep(delay)
                return True
        
        return False
    
    def _get_retry_delay(self, retry_count: int) -> float:
        """Get retry delay with optional exponential backoff.
        
        Args:
            retry_count: Number of retries
            
        Returns:
            Delay in seconds
        """
        base_delay = self.config.retry_delay_seconds
        
        if self.config.exponential_backoff:
            return base_delay * (2 ** retry_count)
        else:
            return base_delay
    
    def _save_error_diagnostics(self, error: TrainingError) -> None:
        """Save error diagnostics to file.
        
        Args:
            error: Training error
        """
        try:
            timestamp = int(error.timestamp)
            diagnostic_file = self.diagnostics_dir / f"error_{timestamp}_{error.error_type}.json"
            
            import json
            
            with open(diagnostic_file, "w") as f:
                json.dump(error.to_dict(), f, indent=2)
            
            # Save additional system info if critical
            if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                self._save_system_diagnostics(timestamp)
                
        except Exception as e:
            logger.warning(f"Failed to save error diagnostics: {e}")
    
    def _save_system_diagnostics(self, timestamp: int) -> None:
        """Save system diagnostics.
        
        Args:
            timestamp: Error timestamp
        """
        try:
            import platform
            import psutil
            
            system_info = {
                "timestamp": timestamp,
                "platform": platform.platform(),
                "python_version": sys.version,
                "mlx_available": mx.metal.is_available(),
                "memory": {
                    "total_gb": psutil.virtual_memory().total / (1024**3),
                    "available_gb": psutil.virtual_memory().available / (1024**3),
                    "percent": psutil.virtual_memory().percent,
                },
                "cpu": {
                    "count": psutil.cpu_count(),
                    "percent": psutil.cpu_percent(interval=1),
                },
            }
            
            diagnostic_file = self.diagnostics_dir / f"system_{timestamp}.json"
            
            import json
            
            with open(diagnostic_file, "w") as f:
                json.dump(system_info, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save system diagnostics: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors.
        
        Returns:
            Error summary
        """
        summary = {
            "total_errors": len(self.error_history),
            "error_counts_by_category": dict(self.error_counts),
            "error_counts_by_severity": {},
            "recoverable_errors": sum(1 for e in self.error_history if e.recoverable),
            "recent_errors": [],
        }
        
        # Count by severity
        for severity in ErrorSeverity:
            count = sum(1 for e in self.error_history if e.severity == severity)
            if count > 0:
                summary["error_counts_by_severity"][severity.value] = count
        
        # Recent errors
        for error in self.error_history[-5:]:
            summary["recent_errors"].append({
                "type": error.error_type,
                "message": error.error_message,
                "category": error.error_category.value,
                "severity": error.severity.value,
                "step": error.step,
            })
        
        return summary


def with_error_handling(
    error_handler: ErrorHandler,
    category: Optional[ErrorCategory] = None,
    retries: Optional[int] = None,
):
    """Decorator for error handling.
    
    Args:
        error_handler: Error handler instance
        category: Override error category
        retries: Override max retries
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            max_retries = retries or error_handler.config.max_retries
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get context from kwargs
                    context = {
                        "function": func.__name__,
                        "attempt": attempt,
                    }
                    if "step" in kwargs:
                        context["step"] = kwargs["step"]
                    if "epoch" in kwargs:
                        context["epoch"] = kwargs["epoch"]
                    
                    # Handle the error
                    should_continue, error = error_handler.handle_error(
                        e, context=context
                    )
                    
                    if error and category:
                        error.error_category = category
                    
                    if should_continue and attempt < max_retries:
                        if error:
                            error.retry_count = attempt + 1
                        continue
                    else:
                        last_error = e
                        break
            
            if last_error:
                raise last_error
        
        return wrapper
    return decorator